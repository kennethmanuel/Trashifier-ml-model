#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pathlib
import pandas as pd
import tensorflow as tf
tf.random.set_seed(42)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2


def main():

    # independent variable
    learning_rate = 0.001
    dropout_rate = 0.2

    data_dir = "../../dataset/HouseHoldGarbage"
    data_dir = pathlib.Path(data_dir)

    # Controlled Variable
    img_height = 150
    img_width = 150
    num_channels = 3
    epochs = 10
    batch_size = 32
    fine_tune_epoch = 15

    def load_dataset():
        ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        class_names = ds.class_names
        return ds, class_names

    def preprocess_ds(ds):
        def preprocess_image(images, labels):
            preprocessed_images = tf.keras.applications.inception_resnet_v2.preprocess_input(
                images)
            return preprocessed_images, labels

        preprocessed_ds = ds.map(preprocess_image)
        return preprocessed_ds

    def get_dataset_partitions(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=15515):
        assert (train_split + test_split + val_split) == 1
        ds_size = tf.data.experimental.cardinality(ds).numpy()

        if shuffle:
            ds = ds.shuffle(shuffle_size, seed=123)

        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)

        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)

        return train_ds, val_ds, test_ds

    def configure_performance(train_ds, val_ds, test_ds):
        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        return train_ds, val_ds, test_ds

    def create_model(num_class, learning_rate, dropout_rate):
        # Load the pre-trained model without the top classification layer
        base_model = InceptionResNetV2(
            weights='imagenet', include_top=False, input_shape=(150, 150, 3))

        # Freeze the layers of the base model
        for layer in base_model.layers:
            layer.trainable = False

        # Custom classification layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(num_class, activation='softmax')(x)

        # Create the transfer learning model
        model = Model(inputs=base_model.input, outputs=predictions)

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

        return model

    def fit(model, train_ds, val_ds, epochs):
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
        return history

    def plot_loss(history, save_path):
        plt.plot(range(epochs), history.history['loss'], label='Training Loss')
        plt.plot(range(epochs),
                 history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def plot_accuracy(history, save_path):
        plt.plot(range(epochs),
                 history.history['accuracy'], label='Training Accuracy')
        plt.plot(range(epochs),
                 history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def get_predictions_labels(ds, model):
        predictions = np.array([])
        labels = np.array([])
        for x, y in test_ds:
            predictions = np.concatenate(
                [predictions, np.argmax(model.predict(x), axis=-1)])
            labels = np.concatenate([labels, y.numpy()])

        return predictions, labels

    def generate_classification_report(labels, predictions, save_path):
        report = classification_report(labels, predictions, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(save_path, index=False)

    def plot_confusion_matrix(labels, predictions, class_names, save_path):
        confusion_matrix = tf.math.confusion_matrix(
            labels=labels, predictions=predictions).numpy()

        # Plot the confusion matrix
        fig, ax = plt.subplots()
        im = ax.imshow(confusion_matrix, cmap='Blues')

        # Customize the plot
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        # Rotate x-axis labels vertically
        ax.set_xticklabels(class_names, rotation=90)
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title('Confusion Matrix')

        # Add the value annotations to the plot
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = ax.text(
                    j, i, confusion_matrix[i, j], ha='center', va='center', color='black')

        # Display the colorbar
        cbar = ax.figure.colorbar(im, ax=ax)

        # Automatically adjust subplot parameters
        plt.tight_layout()

        # Save the plot
        plt.savefig(save_path)

    def fine_tune(model, num_layers_to_freeze):
        # Freeze layers except for the last `num_layers_to_freeze` layers
        num_layers = len(model.layers)
        for layer in model.layers[:num_layers - num_layers_to_freeze]:
            layer.trainable = False
        for layer in model.layers[num_layers - num_layers_to_freeze:]:
            layer.trainable = True

        # Recompile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate/10),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

        return model

    def fine_tune_fit(model, train_ds, val_ds, epochs, history):
        history = model.fit(train_ds, validation_data=val_ds,
                            epochs=epochs, initial_epoch=history.epoch[-1])
        return history

    def plot_acc_loss_fine_tune(history, fine_tuned_history, save_path):
        initial_epochs = len(history.history['loss'])

        acc = history.history['accuracy'] + \
            fine_tuned_history.history['accuracy']
        val_acc = history.history['val_accuracy'] + \
            fine_tuned_history.history['val_accuracy']
        loss = history.history['loss'] + fine_tuned_history.history['loss']
        val_loss = history.history['val_loss'] + \
            fine_tuned_history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.ylim([0.8, 1])
        plt.plot([initial_epochs-1, initial_epochs-1],
                 plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.ylim([0, 1.0])
        plt.plot([initial_epochs-1, initial_epochs-1],
                 plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.savefig(save_path)

    # ACTUAL CODE STARTS HERE
    ds, class_names = load_dataset()
    ds = preprocess_ds(ds)
    train_ds, val_ds, test_ds = get_dataset_partitions(
        ds, train_split=0.8, val_split=0.1, test_split=0.1)
    train_ds, val_ds, test_ds = configure_performance(
        train_ds, val_ds, test_ds)
    model = create_model(len(class_names), learning_rate, dropout_rate)
    history = fit(model, train_ds, val_ds, epochs)
    plot_loss(history, 'loss_plot.png')
    plot_accuracy(history, 'accuracy_plot.png')
    predictions, labels = get_predictions_labels(test_ds, model)
    generate_classification_report(
        labels, predictions, 'classification_report.csv')
    plot_confusion_matrix(labels, predictions,
                          class_names, 'confusion_matrix.png')
    model.save('model.h5')
    fine_tuned_model = fine_tune(model, 36)
    fine_tuned_history = fine_tune_fit(
        fine_tuned_model, train_ds, val_ds, fine_tune_epoch, history)
    plot_acc_loss_fine_tune(history, fine_tuned_history,
                            'fine_tune_acc_loss.png')
    predictions, labels = get_predictions_labels(test_ds, fine_tuned_model)
    generate_classification_report(
        labels, predictions, 'fine_tune_classification_report.csv')
    plot_confusion_matrix(labels, predictions, class_names,
                          'fine_tune_confusion_matrix.png')
    model.save('fine_tune_model.h5')


if __name__ == '__main__':
    main()
