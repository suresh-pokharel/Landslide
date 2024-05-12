from __future__ import absolute_import
import math
import glob
import numpy as np
import h5py
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError, BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import Activation,Conv2D,MaxPooling2D,BatchNormalization,Input,DepthwiseConv2D,add,Dropout,AveragePooling2D,Concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer,InputSpec
from tensorflow.python.keras.utils import conv_utils
import tensorflow.keras.backend as K
import tensorflow as tf
import logging, datetime, os, sys

# import custom functions
from utils import eval_image, jaccard_coef, jaccard_coef_loss, \
			compute_iou, compute_mean_iou, f1_score, load_h5_data, \
			z_scale, calculate_means_stds, normalize, create_output_folder, scheduler

# import configurations
import config

# read configurations
DATASET_FOLDER = config.DATASET_FOLDER
DATASET_TYPE = config.DATASET_TYPE # LANDSLIDE4SENSE or KERELA or ITALY
NUM_EPOCHS = config.NUM_EPOCHS
BATCH_SIZE = config.BATCH_SIZE

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

def deeplabv3_plus(input_shape=(128,128,14),out_stride=16,num_classes=21):
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
print(model.summary())
print("DeepLabV3PLUS")


# Define call backs
filepath = (full_path+"/"+"deeplabv3"+"_best-model.keras")
checkpoint = ModelCheckpoint(filepath, monitor='val_f1_score', verbose=1, save_best_only=True, mode='max')
callback = [tf.keras.callbacks.LearningRateScheduler(scheduler), checkpoint]

# Compile
model.compile(
optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
loss=jaccard_coef_loss,
metrics=['accuracy',
         tf.keras.metrics.Recall(),
         tf.keras.metrics.Precision(),
         f1_score
        ]
)

# Train the Model with Early Stopping
history = model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), callbacks=[checkpoint])

# Save model
model.save(f"{full_path}/{DATASET_TYPE}_deeplabv3_model.keras")
np.save(f"{full_path}/{DATASET_TYPE}_{model.name}_history.npy", history.history)

# Predict Masks on the Test Set
y_pred = model.predict(X_test)
predictions=(y_pred>0.5).astype(np.int8)

print("iou_score: " + str(compute_iou(predictions, y_test)))
print("f1_score: " + str(compute_f1_score(predictions, y_test)))
print("precision: " + str(compute_precision(predictions, y_test)))
print("recall:" + str(compute_recall(predictions, y_test)))
print("deeplabv3")
