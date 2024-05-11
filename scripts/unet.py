#!/usr/bin/env python
# coding: utf-8

# In[61]:
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

# import custom functions
from utils import eval_image, jaccard_coef, jaccard_coef_loss, compute_iou, compute_mean_iou, f1_score, load_h5_data, z_scale, calculate_means_stds, normalize, create_output_folder

full_path = create_output_folder()
print("Output Path: " + full_path)

# ### Prepare Datasets

# dataset folder path
DATASET_FOLDER = "/home/sureshp/Landslide/Suresh/datasets/"
DATASET_TYPE = 'KERELA' # LANDSLIDE4SENSE or KERELA or ITALY

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
else:
    print("No dataset found!")

# Expand dimensions for y-train, y_val, y_test to make similar dimension with output of model
y_train = np.expand_dims(y_train, axis=-1)
y_val = np.expand_dims(y_val, axis=-1)
y_test = np.expand_dims(y_test, axis=-1)


# Print shapes of dataset splits
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# Scale the images
# Find mean and standard dev from training set 
means, stds = calculate_means_stds(X_train)

# From LandSlide4Sense
# means = np.array([-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819])
# stds = np.array([0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913])

# Scale X-train, X_val, X_test with respect to means/stds from X_train
X_train = z_scale(X_train, means, stds)
X_val = z_scale(X_val, means, stds)
X_test = z_scale(X_test, means, stds)

#X_train = normalize(X_train)
#X_val = normalize(X_val)
#X_test = normalize(X_test)

model = models.unet_2d((128, 128, 14), [64, 128, 256, 512, 1024], n_labels=1,
                               stack_num_down=2, stack_num_up=1,
                               activation='ReLU', output_activation='Sigmoid', 
                               batch_norm=True, pool='max', unpool='nearest', name='unet_2d'
                              )

# Define call backs
filepath = (full_path+"/"+model.name+"_"+DATASET_TYPE+"_best-model.keras")
checkpoint = ModelCheckpoint(filepath, monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max')

# Compile
model.compile(
optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
loss = jaccard_coef_loss, #BinaryCrossentropy() #losses.dice, # jaccard_coef_loss
metrics=[
	 f1_score,
         losses.dice_coef,
         tf.keras.metrics.Recall(), 
         tf.keras.metrics.Precision(),
         tf.keras.metrics.MeanIoU(num_classes=2)
        ]
)

# Train the Model with Early Stopping
history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_val, y_val), shuffle=False, callbacks=[checkpoint])

# Save model
model.save(f"{full_path}/{DATASET_TYPE}_{model.name}.keras")
np.save(f"{full_path}/{DATASET_TYPE}_{model.name}_history.npy", history.history)

# Convert to appropriate type and check shapes
y_test = y_test.astype(np.int8)
predictions_test = (model.predict(X_test, batch_size=64) > 0.5).astype(np.int8)
y_val = y_val.astype(np.int8)
predictions_val = (model.predict(X_val, batch_size=64) > 0.5).astype(np.int8)

# Flatten the arrays
y_test_flat = y_test.reshape(-1)
predictions_test_flat = predictions_test.reshape(-1)

y_val_flat = y_val.reshape(-1)
predictions_val_flat = predictions_val.reshape(-1)

#%%
# Compute metrics for the Test set
precision_test = precision_score(y_test_flat, predictions_test_flat)
recall_test = recall_score(y_test_flat, predictions_test_flat)
dice_coefficient_test = f1_score(y_test_flat, predictions_test_flat)
jaccard_index_test = jaccard_score(y_test_flat, predictions_test_flat)

# Compute metrics for the Validation set
precision_val = precision_score(y_val_flat, predictions_val_flat)
recall_val = recall_score(y_val_flat, predictions_val_flat)
dice_coefficient_val = f1_score(y_val_flat, predictions_val_flat)
jaccard_index_val = jaccard_score(y_val_flat, predictions_val_flat)

print("Test Set Metrics:")
print("Precision:", precision_test)
print("Recall:", recall_test)
print("Dice Coefficient (F1 Score):", dice_coefficient_test)
print("Jaccard Index:", jaccard_index_test)

print("\nValidation Set Metrics:")
print("Precision:", precision_val)
print("Recall:", recall_val)
print("Dice Coefficient (F1 Score):", dice_coefficient_val)
print("Jaccard Index:", jaccard_index_val)
