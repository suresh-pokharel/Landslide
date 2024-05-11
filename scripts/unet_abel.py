# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 01:45:17 2024

@author: Abel
"""
#%%
import numpy as np
from natsort import natsorted
import pandas as pd    
import matplotlib.pyplot as plt
from datetime import datetime 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras_unet_collection import models,losses
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, jaccard_score
import os, sys, glob, h5py, math
from datetime import datetime


# Get the current time in seconds
current_time_seconds = int(datetime.now().timestamp())

# Create a folder with the current time in seconds
folder_name = f"folder_{current_time_seconds}"

# Specify the path where you want to create the folder
full_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "outputs", folder_name)

# Create the folder
os.makedirs(full_path)
print("-----------------PATH--------------------")
print(full_path)


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


#%%
dataset_path="/home/sureshp/Landslide/Suresh/datasets/Landslide4Sense_Dataset/"
root = "/home/sureshp/Landslide/Suresh/datasets/Landslide4Sense_Dataset/"
dataset_name="landslide4sense"
batch_size=4##Change this
num_epochs=2## and this


norm = True

if norm:
    norm_name='norm'
else:
    norm_name='no_norm'
    
    
#%% LOADING DATA
################################################################
imgs_ar = load_h5_data(dataset_path + "TrainData/img/*.h5")
masks_ar = load_h5_data(dataset_path + "TrainData/mask/*.h5")
################################################################

means=[]
stds=[]
if norm:  

    for i in range(0,14):
        print(np.mean(imgs_ar[:,:,:,i]), np.max(imgs_ar[:,:,:,i]), np.min(imgs_ar[:,:,:,i]), np.std(imgs_ar[:,:,:,i]))
        means.append(np.mean(imgs_ar[:,:,:,i]))
        stds.append(np.std(imgs_ar[:,:,:,i]))
        
    imgs_norm=[]
    for img in imgs_ar:
    #    img_std=min_max_scaler_image(img)
        img_std=scale_normalize_image(img, means, stds)
        imgs_norm.append(img_std)  
    imgs_norm=np.array(imgs_norm)
    X_train=imgs_norm
else:
    X_train=imgs_ar


y_train=masks_ar.astype(np.float32)

#%% TESTING DATASET

################################################################
imgs_ts = load_h5_data(dataset_path + "TestData/img/*.h5")
masks_ts = load_h5_data(dataset_path + "TestData/mask/*.h5")
################################################################

if norm:
    imgs_st_norm=[]
    for img in imgs_ts:
    #    img_std=min_max_scaler_image(img)
        img_std=scale_normalize_image(img, means, stds)
        imgs_st_norm.append(img_std)  
    imgs_st_norm=np.array(imgs_st_norm)
    X_test=imgs_st_norm
else:
    X_test=imgs_ts #imgs_st_norm

y_test=masks_ts.astype(np.float32)

#%% VALIDATION DATASET

################################################################
imgs_val = load_h5_data(dataset_path + "ValidData/img/*.h5")
masks_val = load_h5_data(dataset_path + "ValidData/mask/*.h5")
################################################################

#means=[]
#stds=[]
#for i in range(0,14):
#    print(np.mean(imgs_val_ar[:,:,:,i]), np.max(imgs_val_ar[:,:,:,i]), np.min(imgs_val_ar[:,:,:,i]), np.std(imgs_val_ar[:,:,:,i]))
#    means.append(np.mean(imgs_val_ar[:,:,:,i]))
#    stds.append(np.std(imgs_val_ar[:,:,:,i]))
if norm:
    imgs_val_norm=[]
    for img in imgs_val:
    #    img_std=min_max_scaler_image(img)
        img_std=scale_normalize_image(img, means, stds)
        imgs_val_norm.append(img_std)  
    imgs_val_norm=np.array(imgs_val_norm)
    X_val=imgs_val_norm
else:
    X_val=imgs_val
y_val=masks_val.astype(np.float32)

#%%
print((X_train.shape, y_train.shape), (X_test.shape, y_test.shape), (X_val.shape, y_val.shape))
#%%Data Augmentation



#%%
IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]


num_labels = 1  #Binary
input_shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)

#%%
model = models.unet_2d(input_shape, [64, 128, 256, 512, 1024], n_labels=1,
                      stack_num_down=2, stack_num_up=1,
                      activation='ReLU', output_activation='Sigmoid', 
                      batch_norm=True, pool='max', unpool='nearest', name='unet_ABEL')
model.summary()

# Define call backs
filepath = (full_path+"/"+model.name+"_best-model.keras")

checkpoint = ModelCheckpoint(filepath, monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max')
lr_shceduler = LearningRateScheduler(lambda _, lr: lr * np.exp(-0.1), verbose=1)
#callbacks_list = [lr_shceduler, checkpoint]
callbacks_list = [checkpoint]
#%%
model.compile(optimizer= Adam(lr = 0.001), 
           # loss= [lovasz_softmax],
            loss=  jaccard_coef_loss,
           # loss= lovasz_loss,
            metrics=[losses.dice_coef, 'accuracy'])
#%%
start1 = datetime.now() 
model_history = model.fit(X_train, y_train, 
                    batch_size = batch_size, 
                    verbose=1, 
                    epochs=num_epochs, 
                    validation_data=(X_val, y_val),
                    callbacks=[callbacks_list],
                    shuffle=True)
stop1 = datetime.now()
execution_time_model = stop1-start1
print(model.name+" execution time is: ", execution_time_model)
#%% Saving history
model_history_df = pd.DataFrame(model_history.history) 

#%% Loading best model
# model.load_weights(filepath)
#%%

# Convert to appropriate type and check shapes
y_test = y_test.astype(np.int8)
predictions_test = (model.predict(X_test, batch_size=4) > 0.5).astype(np.int8)
y_val = y_val.astype(np.int8)
predictions_val = (model.predict(X_val, batch_size=4) > 0.5).astype(np.int8)

# Flatten the arrays
y_test_flat = y_test.reshape(-1)
predictions_test_flat = predictions_test.reshape(-1)
y_val_flat = y_val.reshape(-1)
predictions_val_flat = predictions_val.reshape(-1)
#%%
# Compute metrics for the Test set
precision_test = precision_score(y_test_flat, predictions_test_flat)
recall_test = recall_score(y_test_flat, predictions_test_flat)
accuracy_test = accuracy_score(y_test_flat, predictions_test_flat)
dice_coefficient_test = f1_score(y_test_flat, predictions_test_flat)  # Dice coefficient is equivalent to F1 Score
jaccard_index_test = jaccard_score(y_test_flat, predictions_test_flat)

# Compute metrics for the Validation set
precision_val = precision_score(y_val_flat, predictions_val_flat)
recall_val = recall_score(y_val_flat, predictions_val_flat)
accuracy_val = accuracy_score(y_val_flat, predictions_val_flat)
dice_coefficient_val = f1_score(y_val_flat, predictions_val_flat)  # Dice coefficient is equivalent to F1 Score
jaccard_index_val = jaccard_score(y_val_flat, predictions_val_flat)

print("Test Set Metrics:")
print("Precision:", precision_test)
print("Recall:", recall_test)
print("Accuracy:", accuracy_test)
print("Dice Coefficient (F1 Score):", dice_coefficient_test)
print("Jaccard Index:", jaccard_index_test)

print("\nValidation Set Metrics:")
print("Precision:", precision_val)
print("Recall:", recall_val)
print("Accuracy:", accuracy_val)
print("Dice Coefficient (F1 Score):", dice_coefficient_val)
print("Jaccard Index:", jaccard_index_val)


# Save model
model.save(f"{full_path}/{dataset_name}_{model.name}_model.keras")
np.save(f"{full_path}/{dataset_name}_{model.name}_history.npy", model_history.history)
    
