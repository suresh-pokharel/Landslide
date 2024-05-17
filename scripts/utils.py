from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, jaccard_score
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from keras_unet_collection import models,losses
from tensorflow.keras import backend as K
from keras.losses import Loss
from tensorflow.keras.losses import BinaryCrossentropy
import datetime, os, sys, glob, h5py, math


def create_output_folder():
    # Get the current time in seconds
    current_time_seconds = int(datetime.datetime.now().timestamp())

    # Create a folder with the current time in seconds
    folder_name = f"folder_{current_time_seconds}"

    # Specify the path where you want to create the folder
    full_path = os.path.join(os.getcwd(), "outputs", folder_name)

    # Create the folder
    os.makedirs(full_path)
    return full_path

def load_h5_data(paths):
    paths = glob.glob(paths)
    features = []
    for path in paths:
        # Open the HDF5 file in read mode
        with h5py.File(path, 'r') as file:
            # Get the first item (assume there is only one item at the top level)
            item = list(file.values())[0]
            features.append(np.array(item))
    return np.array(features)

# Load your list of images/h5 and corresponding masks
def load_image_data(image_paths):
    image_paths = glob.glob(image_paths)
    images = []
    for path in image_paths:
        img = Image.open(path)
        
	# Preprocess your images as needed, e.g., resizing, normalization, etc.
        img = np.array(img) / 255.0  # normalize to [0, 1]
        images.append(img)
    return np.array(images)
    
def z_score_normalization(images, means, stds):
    # z-scaling the entire dataset
    scaled_image = (images - means) / stds
    
    # Clip values to ensure they are in the range [0, 1]
    #normalized_images = np.clip(normalized_images, 0, 1)
    return scaled_image
    
def z_score_normalization_double_sd(images, means, stds):
    # z-scaling the entire dataset
    scaled_image = (images - means) / (2* stds) + 0.5 
    
    # Clip values to ensure they are in the range [0, 1]
    clipped_images = np.clip(normalized_images, 0, 1)
    return clipped_images

def min_max_scaling(images):
    # Calculating the minimum and maximum values of the entire dataset
    mins = np.min(images)
    maxs = np.max(images)
    
    # Scaling the entire dataset to the range [0, 1]
    scaled_images = (images - mins) / (maxs - mins)
    
    # Clip values to ensure they are in the range [0, 1]
    clipped_images = np.clip(scaled_images, 0, 1)
    
    return clipped_images
    
def calculate_means_stds(image_data):
    # Assuming image_data has shape (n, 128, 128, 14)
    
    # Calculate mean and standard deviation across all images and channels
    means = np.mean(image_data, axis=(0, 1, 2))
    stds = np.std(image_data, axis=(0, 1, 2))

    return means, stds

def normalize(images):
    normalized_image = (images) / 255
    return normalized_image
    
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


