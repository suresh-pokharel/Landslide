from __future__ import absolute_import

#!/usr/bin/env python
# coding: utf-8

# In[61]:
import math
import glob
from PIL import Image
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError, BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras_unet_collection import models,losses
from tensorflow.keras import backend as K
from keras.losses import Loss
import logging, datetime, os, sys

# import custom functions
from utils import eval_image, jaccard_coef, jaccard_coef_loss, \
            compute_iou, compute_mean_iou, f1_score, load_h5_data, \
            z_scale, calculate_means_stds, normalize, create_output_folder, scheduler

full_path = create_output_folder()
print("Output Path: " + full_path)

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


def min_max_scaling(images, means, stds):
    # Normalize the entire dataset
    normalized_images = (images - means) / stds
    
    # Clip values to ensure they are in the range [0, 1]
    normalized_images = np.clip(normalized_images, 0, 1)
    return normalized_images

def calculate_means_stds(image_data):
    # Assuming image_data has shape (n, 128, 128, 14)
    
    # Calculate mean and standard deviation across all images and channels
    means = np.mean(image_data, axis=(0, 1, 2))
    stds = np.std(image_data, axis=(0, 1, 2))

    return means, stds


import tensorflow.keras.backend as K

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

# ### Prepare Datasets

# In[67]:


# dataset folder path
DATASET_FOLDER = "/home/sureshp/Landslide/Suresh/datasets/"
DATASET_TYPE = 'KERELA' # LANDSLIDE4SENSE or KERELA or ITALY

logging.info('Dataset: ' + DATASET_TYPE)

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
X_train = min_max_scaling(X_train, means, stds)
X_val = min_max_scaling(X_val, means, stds)
X_test = min_max_scaling(X_test, means, stds)

# MODEL
from keras_unet_collection.layer_utils import *
from keras_unet_collection.transformer_layers import patch_extract, patch_embedding, SwinTransformerBlock, patch_merging, patch_expanding

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def swin_transformer_stack(X, stack_num, embed_dim, num_patch, num_heads, window_size, num_mlp, shift_window=True, name=''):
    '''
    Stacked Swin Transformers that share the same token size.
    
    Alternated Window-MSA and Swin-MSA will be configured if `shift_window=True`, Window-MSA only otherwise.
    *Dropout is turned off.
    '''
    # Turn-off dropouts
    mlp_drop_rate = 0 # Droupout after each MLP layer
    attn_drop_rate = 0 # Dropout after Swin-Attention
    proj_drop_rate = 0 # Dropout at the end of each Swin-Attention block, i.e., after linear projections
    drop_path_rate = 0 # Drop-path within skip-connections
    
    qkv_bias = True # Convert embedded patches to query, key, and values with a learnable additive value
    qk_scale = None # None: Re-scale query based on embed dimensions per attention head # Float for user specified scaling factor
    
    if shift_window:
        shift_size = window_size // 2
    else:
        shift_size = 0
    
    for i in range(stack_num):
    
        if i % 2 == 0:
            shift_size_temp = 0
        else:
            shift_size_temp = shift_size

        X = SwinTransformerBlock(dim=embed_dim, num_patch=num_patch, num_heads=num_heads,
                                 window_size=window_size, shift_size=shift_size_temp, num_mlp=num_mlp, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 mlp_drop=mlp_drop_rate, attn_drop=attn_drop_rate, proj_drop=proj_drop_rate, drop_path_prob=drop_path_rate,
                                 name='name{}'.format(i))(X)
    return X


def swin_unet_2d_base(input_tensor, filter_num_begin, depth, stack_num_down, stack_num_up,
                      patch_size, num_heads, window_size, num_mlp, shift_window=True, name='swin_unet'):
    '''
    The base of SwinUNET.
    
    ----------
    Cao, H., Wang, Y., Chen, J., Jiang, D., Zhang, X., Tian, Q. and Wang, M., 2021.
    Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation. arXiv preprint arXiv:2105.05537.
    
    Input
    ----------
        input_tensor: the input tensor of the base, e.g., `keras.layers.Inpyt((None, None, 3))`.
        filter_num_begin: number of channels in the first downsampling block;
                          it is also the number of embedded dimensions.
        depth: the depth of Swin-UNET, e.g., depth=4 means three down/upsampling levels and a bottom level.
        stack_num_down: number of convolutional layers per downsampling level/block.
        stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
        name: prefix of the created keras model and its layers.
        
        ---------- (keywords of Swin-Transformers) ----------
        
        patch_size: The size of extracted patches,
                    e.g., patch_size=(2, 2) means 2-by-2 patches
                    *Height and width of the patch must be equal.
                    
        num_heads: number of attention heads per down/upsampling level,
                     e.g., num_heads=[4, 8, 16, 16] means increased attention heads with increasing depth.
                     *The length of num_heads must equal to `depth`.
                     
        window_size: the size of attention window per down/upsampling level,
                     e.g., window_size=[4, 2, 2, 2] means decreased window size with increasing depth.
                     
        num_mlp: number of MLP nodes.
        
        shift_window: The indicator of window shifting;
                      shift_window=True means applying Swin-MSA for every two Swin-Transformer blocks.
                      shift_window=False means MSA with fixed window locations for all blocks.

    Output
    ----------
        output tensor.
        
    Note: This function is experimental.
          The activation functions of all Swin-Transformers are fixed to GELU.
    
    '''
    # Compute number be patches to be embeded
    input_size = input_tensor.shape.as_list()[1:]
    num_patch_x = input_size[0]//patch_size[0]
    num_patch_y = input_size[1]//patch_size[1]
    
    # Number of Embedded dimensions
    embed_dim = filter_num_begin
    
    depth_ = depth
    X_skip = []
    X = input_tensor
    
    # Patch extraction
    X = patch_extract(patch_size)(X)

    # Embed patches to tokens
    X = patch_embedding(num_patch_x*num_patch_y, embed_dim)(X)
    
    # The first Swin Transformer stack
    X = swin_transformer_stack(X, stack_num=stack_num_down,
                               embed_dim=embed_dim, num_patch=(num_patch_x, num_patch_y),
                               num_heads=num_heads[0], window_size=window_size[0], num_mlp=num_mlp,
                               shift_window=shift_window, name='{}_swin_down0'.format(name))
    X_skip.append(X)
    
    # Downsampling blocks
    for i in range(depth_-1):
        
        # Patch merging
        X = patch_merging((num_patch_x, num_patch_y), embed_dim=embed_dim, name='down{}'.format(i))(X)
        
        # update token shape info
        embed_dim = embed_dim*2
        num_patch_x = num_patch_x//2
        num_patch_y = num_patch_y//2
        
        # Swin Transformer stacks
        X = swin_transformer_stack(X, stack_num=stack_num_down,
                                   embed_dim=embed_dim, num_patch=(num_patch_x, num_patch_y),
                                   num_heads=num_heads[i+1], window_size=window_size[i+1], num_mlp=num_mlp,
                                   shift_window=shift_window, name='{}_swin_down{}'.format(name, i+1))
        
        # Store tensors for concat
        X_skip.append(X)
        
    # reverse indexing encoded tensors and hyperparams
    X_skip = X_skip[::-1]
    num_heads = num_heads[::-1]
    window_size = window_size[::-1]
    
    # upsampling begins at the deepest available tensor
    X = X_skip[0]
    
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]
    
    depth_decode = len(X_decode)
    
    for i in range(depth_decode):
        
        # Patch expanding
        X = patch_expanding(num_patch=(num_patch_x, num_patch_y),
                            embed_dim=embed_dim, upsample_rate=2, return_vector=True, name='{}_swin_up{}'.format(name, i))(X)
        

        # update token shape info
        embed_dim = embed_dim//2
        num_patch_x = num_patch_x*2
        num_patch_y = num_patch_y*2
        
        # Concatenation and linear projection
        X = concatenate([X, X_decode[i]], axis=-1, name='{}_concat_{}'.format(name, i))
        X = Dense(embed_dim, use_bias=False, name='{}_concat_linear_proj_{}'.format(name, i))(X)
        
        # Swin Transformer stacks
        X = swin_transformer_stack(X, stack_num=stack_num_up,
                           embed_dim=embed_dim, num_patch=(num_patch_x, num_patch_y),
                           num_heads=num_heads[i], window_size=window_size[i], num_mlp=num_mlp,
                           shift_window=shift_window, name='{}_swin_up{}'.format(name, i))
        
    # The last expanding layer; it produces full-size feature maps based on the patch size
    # !!! <--- "patch_size[0]" is used; it assumes patch_size = (size, size)
    X = patch_expanding(num_patch=(num_patch_x, num_patch_y),
                        embed_dim=embed_dim, upsample_rate=patch_size[0], return_vector=False)(X)
    
    return X


def swin_unet_2d(input_size, filter_num_begin, n_labels, depth, stack_num_down, stack_num_up,
                      patch_size, num_heads, window_size, num_mlp, output_activation='Softmax', shift_window=True, name='swin_unet'):
    '''
    The base of SwinUNET.
    
    ----------
    Cao, H., Wang, Y., Chen, J., Jiang, D., Zhang, X., Tian, Q. and Wang, M., 2021.
    Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation. arXiv preprint arXiv:2105.05537.
    
    Input
    ----------
        input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
        filter_num_begin: number of channels in the first downsampling block;
                          it is also the number of embedded dimensions.
        n_labels: number of output labels.
        depth: the depth of Swin-UNET, e.g., depth=4 means three down/upsampling levels and a bottom level.
        stack_num_down: number of convolutional layers per downsampling level/block.
        stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
        name: prefix of the created keras model and its layers.
        
        ---------- (keywords of Swin-Transformers) ----------
        
        patch_size: The size of extracted patches,
                    e.g., patch_size=(2, 2) means 2-by-2 patches
                    *Height and width of the patch must be equal.
                    
        num_heads: number of attention heads per down/upsampling level,
                     e.g., num_heads=[4, 8, 16, 16] means increased attention heads with increasing depth.
                     *The length of num_heads must equal to `depth`.
                     
        window_size: the size of attention window per down/upsampling level,
                     e.g., window_size=[4, 2, 2, 2] means decreased window size with increasing depth.
                     
        num_mlp: number of MLP nodes.
        
        shift_window: The indicator of window shifting;
                      shift_window=True means applying Swin-MSA for every two Swin-Transformer blocks.
                      shift_window=False means MSA with fixed window locations for all blocks.
        
    Output
    ----------
        model: a keras model.
    
    Note: This function is experimental.
          The activation functions of all Swin-Transformers are fixed to GELU.
    '''
    IN = Input(input_size)
    
    # base
    X = swin_unet_2d_base(IN, filter_num_begin=filter_num_begin, depth=depth, stack_num_down=stack_num_down, stack_num_up=stack_num_up,
                          patch_size=patch_size, num_heads=num_heads, window_size=window_size, num_mlp=num_mlp, shift_window=shift_window, name=name)
    
    # output layer
    OUT = CONV_output(X, n_labels, kernel_size=1, activation=output_activation, name='{}_output'.format(name))
    
    # functional API model
    model = Model(inputs=[IN,], outputs=[OUT,], name='{}_model'.format(name))
    
    return model

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

model = swin_unet_2d((128, 128, 14), filter_num_begin=64, n_labels=1, depth=4, stack_num_down=2, stack_num_up=2, patch_size=(2, 2), num_heads=[4, 8, 8, 8], window_size=[4, 2, 2, 2], num_mlp=512, output_activation='Sigmoid', shift_window=True, name='swin_unet')

# Define call backs
filepath = (full_path+"/"+model.name+"_"+DATASET_TYPE+"_best-model.keras")
checkpoint = ModelCheckpoint(filepath, monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max')

callback = [tf.keras.callbacks.LearningRateScheduler(scheduler), checkpoint]

# Compile
model.compile(
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
loss=jaccard_coef_loss,
metrics=[
	 f1_score,
         losses.dice_coef,
         tf.keras.metrics.Recall(), 
         tf.keras.metrics.Precision(),
         tf.keras.metrics.MeanIoU(num_classes=2)
        ]
)

# Train the Model with Early Stopping
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val), callbacks=[checkpoint])

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
print("SWIN_UNET")
