from __future__ import absolute_import
import math
import glob
import numpy as np
import h5py
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, jaccard_score, f1_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError, BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import Activation,Conv2D,MaxPooling2D,BatchNormalization,Input,DepthwiseConv2D,add,Dropout,AveragePooling2D,Concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer,InputSpec
from tensorflow.python.keras.utils import conv_utils
import tensorflow.keras.backend as K
import tensorflow as tf
import logging, datetime, os, sys
import segmentation_models as sm

# import custom functions
from utils.functions import *
from utils.loss_functions import *

# import configurations
import config

# read configurations
DATASET_FOLDER = config.DATASET_FOLDER
DATASET_TYPE = config.DATASET_TYPE 		# LANDSLIDE4SENSE or KERELA or ITALY or SKIN_LESION
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

# MODEL: Deeplab v3
class BilinearUpsampling(Layer):
    def __init__(self, upsampling=(2, 2), data_format=None, **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)
        self.data_format = data_format
        self.upsampling = upsampling

    def compute_output_shape(self, input_shape):
        height = self.upsampling[0] * input_shape[1] if input_shape[1] is not None else None
        width = self.upsampling[1] * input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0], height, width, input_shape[3])

    def call(self, inputs):
        new_height = inputs.shape[1] * self.upsampling[0]
        new_width = inputs.shape[2] * self.upsampling[1]
        return tf.image.resize(inputs, (new_height, new_width), method=tf.image.ResizeMethod.BILINEAR)

    def get_config(self):
        config = {'upsampling': self.upsampling, 'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def xception_downsample_block(x,channels,top_relu=False):
	##separable conv1
	if top_relu:
		x=Activation("relu")(x)
	x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Conv2D(channels,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Activation("relu")(x)
	
	##separable conv2
	x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Conv2D(channels,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Activation("relu")(x)
	
	##separable conv3
	x=DepthwiseConv2D((3,3),strides=(2,2),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Conv2D(channels,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	return x
def res_xception_downsample_block(x,channels):
	res=Conv2D(channels,(1,1),strides=(2,2),padding="same",use_bias=False)(x)
	res=BatchNormalization()(res)
	x=xception_downsample_block(x,channels)
	x=add([x,res])
	return x
def xception_block(x,channels):
	##separable conv1
	x=Activation("relu")(x)
	x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Conv2D(channels,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	
	##separable conv2
	x=Activation("relu")(x)
	x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Conv2D(channels,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	
	##separable conv3
	x=Activation("relu")(x)
	x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Conv2D(channels,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	return x	
def res_xception_block(x,channels):
	res=x
	x=xception_block(x,channels)
	x=add([x,res])
	return x
def aspp(x,input_shape,out_stride):
	b0=Conv2D(256,(1,1),padding="same",use_bias=False)(x)
	b0=BatchNormalization()(b0)
	b0=Activation("relu")(b0)
	
	b1=DepthwiseConv2D((3,3),dilation_rate=(6,6),padding="same",use_bias=False)(x)
	b1=BatchNormalization()(b1)
	b1=Activation("relu")(b1)
	b1=Conv2D(256,(1,1),padding="same",use_bias=False)(b1)
	b1=BatchNormalization()(b1)
	b1=Activation("relu")(b1)
	
	b2=DepthwiseConv2D((3,3),dilation_rate=(12,12),padding="same",use_bias=False)(x)
	b2=BatchNormalization()(b2)
	b2=Activation("relu")(b2)
	b2=Conv2D(256,(1,1),padding="same",use_bias=False)(b2)
	b2=BatchNormalization()(b2)
	b2=Activation("relu")(b2)	

	b3=DepthwiseConv2D((3,3),dilation_rate=(12,12),padding="same",use_bias=False)(x)
	b3=BatchNormalization()(b3)
	b3=Activation("relu")(b3)
	b3=Conv2D(256,(1,1),padding="same",use_bias=False)(b3)
	b3=BatchNormalization()(b3)
	b3=Activation("relu")(b3)
	
	out_shape=int(input_shape[0]/out_stride)
	b4=AveragePooling2D(pool_size=(out_shape,out_shape))(x)
	b4=Conv2D(256,(1,1),padding="same",use_bias=False)(b4)
	b4=BatchNormalization()(b4)
	b4=Activation("relu")(b4)
	b4=BilinearUpsampling((out_shape,out_shape))(b4)
	
	x=Concatenate()([b4,b0,b1,b2,b3])
	return x

def deeplabv3_plus(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]),out_stride=16,num_classes=21):
	img_input=Input(shape=input_shape)
	x=Conv2D(32,(3,3),strides=(2,2),padding="same",use_bias=False)(img_input)
	x=BatchNormalization()(x)
	x=Activation("relu")(x)
	x=Conv2D(64,(3,3),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Activation("relu")(x)
	
	x=res_xception_downsample_block(x,128)

	res=Conv2D(256,(1,1),strides=(2,2),padding="same",use_bias=False)(x)
	res=BatchNormalization()(res)	
	x=Activation("relu")(x)
	x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Activation("relu")(x)
	x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)
	skip=BatchNormalization()(x)
	x=Activation("relu")(skip)
	x=DepthwiseConv2D((3,3),strides=(2,2),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)	
	x=add([x,res])
	
	x=xception_downsample_block(x,728,top_relu=True)
	
	for i in range(16):
		x=res_xception_block(x,728)

	res=Conv2D(1024,(1,1),padding="same",use_bias=False)(x)
	res=BatchNormalization()(res)	
	x=Activation("relu")(x)
	x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Conv2D(728,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Activation("relu")(x)
	x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Conv2D(1024,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Activation("relu")(x)
	x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Conv2D(1024,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)	
	x=add([x,res])
	
	x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Conv2D(1536,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Activation("relu")(x)
	x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Conv2D(1536,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Activation("relu")(x)
	x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Conv2D(2048,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)	
	x=Activation("relu")(x)
	
	#aspp
	x=aspp(x,input_shape,out_stride)
	x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Activation("relu")(x)
	x=Dropout(0.9)(x)
	
	##decoder 
	x=BilinearUpsampling((4,4))(x)
	dec_skip=Conv2D(48,(1,1),padding="same",use_bias=False)(skip)
	dec_skip=BatchNormalization()(dec_skip)
	dec_skip=Activation("relu")(dec_skip)
	x=Concatenate()([x,dec_skip])
	
	x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Activation("relu")(x)
	x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Activation("relu")(x)
	
	x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Activation("relu")(x)
	x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)
	x=BatchNormalization()(x)
	x=Activation("relu")(x)
	
	x=Conv2D(num_classes,(1,1),padding="same")(x)
	x=BilinearUpsampling((4,4))(x)
	model=Model(img_input,x)
	return model

# Define Model
model=deeplabv3_plus(num_classes=1)
#model.name="DeepLabV3PLUS"
print(model.summary())

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

print(model.name)
