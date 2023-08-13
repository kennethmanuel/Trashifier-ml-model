# Import necessary modules
import argparse
import pathlib
import os
from matplotlib import pyplot as plt

# Define command-line arguments
# def parse_args():
#     parser = argparse.ArgumentParser(description='Your app description')
#     # Add your command-line arguments here
#     # parser.add_argument('--argument', type=int, help='Help message')
#     return parser.parse_args()

import os
import matplotlib.pyplot as plt

import os
import matplotlib.pyplot as plt

def generate_plot_class_distribution(dataset_dir_path, save_path):
    # Directory path containing the image dataset
    dataset_dir = dataset_dir_path
    
    # Get the list of subdirectories (classes)
    classes = os.listdir(dataset_dir)
    
    # Initialize a dictionary to store the class names and image counts
    class_counts = {}
    
    # Iterate over the subdirectories
    for class_name in classes:
        # Get the path to the current class directory
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            # Get the count of images in the class directory
            image_count = len(os.listdir(class_dir))
            class_counts[class_name] = image_count
    
    # Extract the class names and image counts for plotting
    class_names = list(class_counts.keys())
    image_counts = list(class_counts.values())
    
    # Set the figure size to accommodate the plot
    plt.figure(figsize=(6, 6))
    
    # Create a pie chart
    wedges, text_labels, autotexts = plt.pie(image_counts, labels=class_names, autopct='%1.1f%%', textprops={'fontsize': 12})
    
    # Modify the title style
    title_font = {'weight': 'bold', 'size': 16}
    plt.title("Class Distribution", fontdict=title_font)
    
    # Modify the size of all text elements
    for label in text_labels:
        label.set_fontsize(12)
    
    for autotext in autotexts:
        autotext.set_fontsize(12)
    
    # Save the plot to a file
    plt.savefig(save_path)
    
    # Close the plot to release resources
    plt.close()

# Main function
def main():

    hhg_data_dir = "../dataset/HouseHoldGarbage"
    hhg_data_dir = pathlib.Path(hhg_data_dir) #pathlib.Path respect different semantics appropriate for different operating systems
    hhg_save_path = "hhg_dist.png"
    hhg_image_count = len(list(hhg_data_dir.glob('*/*.jpg')))

    wadaba_data_dir = "../dataset/WaDaBa"
    wadaba_data_dir = pathlib.Path(wadaba_data_dir) #pathlib.Path respect different semantics appropriate for different operating systems
    wadaba_save_path = "wadaba_dist.png"
    wadaba_image_count = len(list(wadaba_data_dir.glob('*/*.jpg')))
    
    plastic_data_dir = "../dataset/Extended WaDaBa"
    plastic_data_dir = pathlib.Path(plastic_data_dir) #pathlib.Path respect different semantics appropriate for different operating systems
    plastic_save_path = "ext_wadaba_dist.png"
    plastic_image_count = len(list(plastic_data_dir.glob('*/*.jpg')))
    

    generate_plot_class_distribution(hhg_data_dir, hhg_save_path)
    generate_plot_class_distribution(plastic_data_dir, plastic_save_path)
    generate_plot_class_distribution(wadaba_data_dir, wadaba_save_path)


# Entry point of the application
if __name__ == '__main__':
    main()
