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

# import configurations
import config

# read configurations
DATASET_FOLDER = config.DATASET_FOLDER
DATASET_TYPE = config.DATASET_TYPE # LANDSLIDE4SENSE or KERELA or ITALY

# create output folder
full_path = create_output_folder()
print("Output Path: " + full_path)
print('Dataset: ' + DATASET_TYPE)


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

# Scale X-train, X_val, X_test with respect to means/stds from X_train
X_train = z_scale(X_train, means, stds)
X_val = z_scale(X_val, means, stds)
X_test = z_scale(X_test, means, stds)

model = models.att_unet_2d((X_train.shape[-3], X_train.shape[-2], X_train.shape[-1]), [64, 128, 256, 512, 1024], n_labels=1, stack_num_down=2, stack_num_up=2, activation='ReLU', atten_activation='ReLU', attention='add', output_activation='Sigmoid', batch_norm=True, pool='max', unpool='nearest', name='attunet')

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Define call backs
filepath = (full_path+"/"+model.name+"_"+DATASET_TYPE+"_best-model.keras")
checkpoint = ModelCheckpoint(filepath, monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max')
callback = [tf.keras.callbacks.LearningRateScheduler(scheduler), checkpoint]

# Compile
model.compile(
optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
loss=jaccard_coef_loss,
metrics=['accuracy',
         tf.keras.metrics.Recall(),
         tf.keras.metrics.Precision(),
         f1_score
        ]
)

# Train the Model with Early Stopping
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[callback])

# Save model
model.save(f"{full_path}/{DATASET_TYPE}_{model.name}_{DATASET_TYPE}.keras")
np.save(f"{full_path}/{DATASET_TYPE}_{model.name}_{DATASET_TYPE}_history.npy", history.history)

# Predict Masks on the Test Set
y_pred = model.predict(X_test)
predictions=(y_pred>0.5).astype(np.int8)

print("iou_score: " + str(compute_iou(predictions, y_test)))
print("f1_score: " + str(compute_f1_score(predictions, y_test)))
print("precision: " + str(compute_precision(predictions, y_test)))
print("recall:" + str(compute_recall(predictions, y_test)))
print("ATTN_UNET_2D")
