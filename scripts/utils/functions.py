from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, jaccard_score
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from keras_unet_collection import models,losses
from tensorflow.keras import backend as K
#from keras.losses import Loss
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

def prepare_dataset(DATASET_TYPE, DATASET_FOLDER):
    # ### Prepare Datasets
    if DATASET_TYPE == 'LANDSLIDE4SENSE':
        dataset_path = DATASET_FOLDER + "Landslide4Sense_Dataset/"
        X_train = load_h5_data(dataset_path + "TrainData/img/*.h5")
        y_train = load_h5_data(dataset_path + "TrainData/mask/*.h5")
        X_val = load_h5_data(dataset_path + "ValidData/img/*.h5")
        y_val = load_h5_data(dataset_path + "ValidData/mask/*.h5")
        X_test = load_h5_data(dataset_path + "TestData/img/*.h5")
        y_test = load_h5_data(dataset_path + "TestData/mask/*.h5")
    elif DATASET_TYPE == 'KERELA':
        dataset_path = DATASET_FOLDER + "new_dataset/new_dataset_h5/"
        X = load_h5_data(dataset_path + "images/*.h5")
        y = load_h5_data(dataset_path + "masks/*.h5")

        # Split the data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    elif DATASET_TYPE == 'ITALY':
        dataset_path = DATASET_FOLDER + "new_dataset_Italy/new_dataset_Italy_h5/"
        X = load_h5_data(dataset_path + "images/*.h5")
        y = load_h5_data(dataset_path + "masks/*.h5")

        # Split the data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    elif DATASET_TYPE == 'SKIN_LESION':

        # Define dataset path
        dataset_path = DATASET_FOLDER + "skin_lesion/"

        # Define patterns for image and mask files
        x_train_pattern = dataset_path + "ISBI2016_ISIC_Part1_Training_Data/*.jpg"
        y_train_pattern = dataset_path + "ISBI2016_ISIC_Part1_Training_GroundTruth/*_Segmentation.png"

        # Get the list of image and mask files
        x_train_files = glob.glob(x_train_pattern)
        y_train_files = glob.glob(y_train_pattern)

        # Extract base file names without extensions
        x_train_names = [os.path.basename(file).replace('.jpg', '') for file in x_train_files]
        y_train_names = [os.path.basename(file).replace('_Segmentation.png', '') for file in y_train_files]

        # Find common file names between X_train and y_train
        common_names = set(x_train_names).intersection(y_train_names)

        # Filter X_train and y_train to keep only the common images
        X_train = [file for file in x_train_files if os.path.basename(file).replace('.jpg', '') in common_names]
        y_train = [file for file in y_train_files if os.path.basename(file).replace('_Segmentation.png', '') in common_names]

        # Check the number of matched images
        print(f"Number of matched images: {len(common_names)}")
        
        # Split the data into training, validation, and test sets
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    	
     
        X_test = load_image_data(dataset_path + "ISBI2016_ISIC_Part1_Test_Data/*.jpg")
        y_test = load_image_data(dataset_path + "ISBI2016_ISIC_Part1_Test_GroundTruth/*.png")
        
    else:
        print("No dataset found!")
        return 0
    
    # return
    return X_train, X_val, X_test, y_train, y_val, y_test


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
    clipped_images = np.clip(scaled_image, 0, 1)
    return clipped_images
    
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

def f1_score_custom(y_true, y_pred):
    # Cast both y_true and y_pred to float32
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')

    # Ensure y_pred has the same shape as y_true
    if K.ndim(y_pred) > K.ndim(y_true):
        y_pred = K.squeeze(y_pred, axis=-1)
    
    # Calculate true positives, false positives, and false negatives
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    FP = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)))
    FN = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))
    
    # Calculate precision and recall
    precision = TP / (TP + FP + K.epsilon())
    recall = TP / (TP + FN + K.epsilon())
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_score
    
