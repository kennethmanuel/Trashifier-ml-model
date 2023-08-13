#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide TensorFlow Warning due to using GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Disable GPU
import tensorflow as tf
import time
import numpy as np
from keras.applications.mobilenet_v2 import preprocess_input


def main():
    model_path = 'model.h5'  # Replace with the actual path to your first pretrained model
    fine_tune_model_path = 'fine_tune_model.h5'  # Replace with the actual path to your second pretrained model

    # Load model 1
    model = tf.keras.models.load_model(model_path)

    # Load model 2
    fine_tune_model = tf.keras.models.load_model(fine_tune_model_path)

    image_dir = '../trash_iftest'

    image_files = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]

    inference_times_pft = []
    inference_times_ft = []

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = tf.keras.utils.load_img(image_path, target_size=(150, 150))
        image_array = np.expand_dims(image, axis=0)
        image_array = preprocess_input(image_array)

        start_time_pft = time.time()
        predictions2 = model.predict(image_array)
        end_time_pft = time.time()

        start_time_ft = time.time()
        predictions2 = fine_tune_model.predict(image_array)
        end_time_ft = time.time()

        inference_time_pft = end_time_pft - start_time_pft
        inference_times_pft.append(inference_time_pft)

        inference_time2 = end_time_ft - start_time_ft
        inference_times_ft.append(inference_time2)

    average_inference_time_pft = sum(inference_times_pft) / len(inference_times_pft)
    average_inference_time_ft = sum(inference_times_ft) / len(inference_times_ft)

    # Save average inference times to a file
    with open('average_inference_times.txt', 'w') as f:
        f.write(f"Average inference pre fine tune: {average_inference_time_pft} seconds\n")
        f.write(f"Average inference fine tuned: {average_inference_time_ft} seconds\n")
    
    print("Save success!")

if __name__ == '__main__':
    main()

