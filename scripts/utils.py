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

def eval_image(predict,label,num_classes):
    index = np.where((label>=0) & (label<num_classes))
    predict = predict[index]
    label = label[index] 
    
    TP = np.zeros((num_classes, 1))
    FP = np.zeros((num_classes, 1))
    TN = np.zeros((num_classes, 1))
    FN = np.zeros((num_classes, 1))
    
    for i in range(0,num_classes):
        TP[i] = np.sum(label[np.where(predict==i)]==i)
        FP[i] = np.sum(label[np.where(predict==i)]!=i)
        TN[i] = np.sum(label[np.where(predict!=i)]!=i)
        FN[i] = np.sum(label[np.where(predict!=i)]==i)        
    
    return int(TP[0]), int(FP[0]), int(TN[0]), int(FN[0])

def jaccard_coef(y_true, y_pred):
    # Cast tensors to float32
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), tf.float32)
    y_pred_f = tf.cast(tf.keras.backend.flatten(y_pred), tf.float32)
    
    # Calculate the intersection
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    
    # Calculate the sum of both predictions and true values, then subtract intersection to get union
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    
    # Smooth term to avoid division by zero
    smooth = 1.0
    
    # Compute Jaccard index
    jaccard_index = (intersection + smooth) / (union + smooth)
    return jaccard_index

def jaccard_coef_loss(y_true, y_pred):
    return -jaccard_coef(y_true, y_pred)

def compute_iou(predictions, masks):
    intersection = np.logical_and(predictions, masks)
    union = np.logical_or(predictions, masks)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def compute_mean_iou(predictions_list, masks_list):
    num_samples = len(predictions_list)
    total_iou = 0.0

    for i in range(num_samples):
        predictions = predictions_list[i]
        masks = masks_list[i]

        iou = compute_iou(predictions, masks)
        total_iou += iou

    mean_iou = total_iou / num_samples
    return mean_iou


def f1_score(y_true, y_pred):
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
    
def z_scale(images, means, stds):
    # z-scaling the entire dataset
    scaled_image = (images - means) / stds
    
    # Clip values to ensure they are in the range [0, 1]
    #normalized_images = np.clip(normalized_images, 0, 1)
    return scaled_image

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


