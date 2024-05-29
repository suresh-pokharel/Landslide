# coding: utf-8

from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, jaccard_score, f1_score
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

from keras_unet_collection import models,losses
from tensorflow.keras import backend as K
from keras.losses import Loss
from tensorflow.keras.losses import BinaryCrossentropy
import datetime, os, sys, glob, h5py, math
import segmentation_models as sm

# import custom functions
from utils.functions import *
from utils.loss_functions import *

# import configurations
import config

# read configurations
DATASET_FOLDER = config.DATASET_FOLDER
DATASET_TYPE = config.DATASET_TYPE # LANDSLIDE4SENSE or KERELA or ITALY or SKIN_LESION
NUM_EPOCHS = config.NUM_EPOCHS
BATCH_SIZE = config.BATCH_SIZE
LEARNING_RATE = config.LEARNING_RATE

# create output folder
full_path = create_output_folder()
print("Output Path: " + full_path)
print('Dataset: ' + DATASET_TYPE)

# process and get dataset ready
X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(DATASET_TYPE, DATASET_FOLDER)

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

# Scale X-train, X_val, X_test with respect to means/stds from X_train
# X_train = z_score_normalization(X_train, means, stds)
# X_val = z_score_normalization(X_val, means, stds)
# X_test = z_score_normalization(X_test, means, stds)

# Normalize
X_train = normalize(X_train)
X_val = normalize(X_val)
X_test = normalize(X_test)

# Scale
X_train = min_max_scaling(X_train)
X_val = min_max_scaling(X_val)
X_test = min_max_scaling(X_test)

# define model
model = models.att_unet_2d((X_train.shape[-3], X_train.shape[-2], X_train.shape[-1]), [64, 128, 256, 512, 1024], n_labels=1, stack_num_down=2, stack_num_up=2, activation='ReLU', atten_activation='ReLU', attention='add', output_activation='Sigmoid', batch_norm=True, pool='max', unpool='nearest', name='attunet')

# Define call backs
# best model path
filepath = (full_path+"/"+model.name+"_"+DATASET_TYPE+"_best-model.keras")

#early stopping
es = EarlyStopping(monitor='val_dice_score', patience=9, restore_best_weights=True, mode='max')

#checkpoint
checkpoint = ModelCheckpoint(filepath, monitor='val_dice_score', verbose=1, save_best_only=True, mode='max')

# lr_scheduler
lr_shceduler = LearningRateScheduler(lambda _, lr: lr * np.exp(-0.1), verbose=1)
    
# Define combined loss
loss_1 = sm.losses.DiceLoss()
loss_2 = sm.losses.JaccardLoss()
loss_3 = sm.losses.BinaryFocalLoss()
loss_4 = sm.losses.BinaryCELoss()
loss_5 = GeneralizedDiceLoss()
loss_6 = TverskyLoss
loss_7 = IoULoss
loss_8 = k_lovasz_hinge(per_image=True)

# Combined loss functions
loss_A = loss_1 + loss_2

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
    loss = loss_1,
    metrics=['accuracy',
         sm.metrics.Recall(),
         sm.metrics.Precision(),
         sm.metrics.FScore(),
         sm.metrics.IOUScore(),
         sm.metrics.DICEScore()
        ]
)

# Train the Model with Early Stopping
history = model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), shuffle=False, callbacks=[checkpoint])

# Save model
model.save(f"{full_path}/{DATASET_TYPE}_{model.name}_{DATASET_TYPE}.keras")
np.save(f"{full_path}/{DATASET_TYPE}_{model.name}_{DATASET_TYPE}_history.npy", history.history)

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

print(model.name)
