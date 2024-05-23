from __future__ import print_function, division
import numpy as np
from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K
import tensorflow as tf

def TverskyLoss(targets, inputs, alpha=0.5, beta=0.5, smooth=1e-6):
        
        #flatten label and prediction tensors
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)
        
        #True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets))
        FP = K.sum(((1-targets) * inputs))
        FN = K.sum((targets * (1-inputs)))
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky
        
        
class GeneralizedDiceLoss(Loss):
    r"""Creates a criterion to measure Generalized Dice loss:
    
    .. math:: L(GDL) = 1 - \frac{2 \sum_{c=1}^{C} w_c \sum_{i=1}^{N} p_{ci} g_{ci}}{\sum_{c=1}^{C} w_c \left( \sum_{i=1}^{N} p_{ci} + \sum_{i=1}^{N} g_{ci} \right)}
    
    Where:
        - \( p_{ci} \) is the predicted probability for class \( c \) at pixel \( i \).
        - \( g_{ci} \) is the ground truth for class \( c \) at pixel \( i \).
        - \( w_c \) is the weight for class \( c \), typically chosen as \( w_c = \frac{1}{(\sum_{i=1}^{N} g_{ci})^2} \).
    
    Args:
        class_weights: Array (``tf.Tensor``) of class weights (``len(weights) = num_classes``). If None, weights are computed as described above.
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        per_image: If ``True`` loss is calculated for each image in batch and then averaged, else loss is calculated for the whole batch.
        smooth: Value to avoid division by zero.

    Returns:
        A callable ``generalized_dice_loss`` instance. Can be used in ``model.compile(...)`` function or combined with other losses.

    Example:

    .. code:: python

        loss = GeneralizedDiceLoss()
        model.compile('SGD', loss=loss)
    """
    
    def __init__(self, class_weights=None, class_indexes=None, per_image=False, smooth=1e-6):
        super().__init__(name='generalized_dice_loss')
        self.class_weights = class_weights
        self.class_indexes = class_indexes
        self.per_image = per_image
        self.smooth = smooth

    def __call__(self, y_true, y_pred, sample_weight=None):
        num_classes = tf.shape(y_true)[-1]

        if self.class_weights is None:
            # Compute class weights as the inverse of the squared sum of each class
            class_sums = tf.reduce_sum(y_true, axis=[0, 1, 2])
            class_weights = 1.0 / (tf.square(class_sums) + self.smooth)
        else:
            class_weights = tf.convert_to_tensor(self.class_weights, dtype=tf.float32)
        
        if self.class_indexes is not None:
            class_weights = tf.gather(class_weights, self.class_indexes)
            y_true = tf.gather(y_true, self.class_indexes, axis=-1)
            y_pred = tf.gather(y_pred, self.class_indexes, axis=-1)

        intersection = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2])
        y_true_sum = tf.reduce_sum(y_true, axis=[0, 1, 2])
        y_pred_sum = tf.reduce_sum(y_pred, axis=[0, 1, 2])

        numerator = 2.0 * tf.reduce_sum(class_weights * intersection)
        denominator = tf.reduce_sum(class_weights * (y_true_sum + y_pred_sum))

        dice_score = numerator / (denominator + self.smooth)
        loss = 1 - dice_score
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, dtype=loss.dtype)
            loss = loss * sample_weight

        return loss
        
        
def IoULoss(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    intersection = K.sum(K.dot(targets, inputs))
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU
    
    
"""
Lovasz-Softmax and Jaccard hinge loss in Tensorflow
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)

Source: https://github.com/bermanmaxim/LovaszSoftmax/blob/master/tensorflow/lovasz_losses_tf.py
https://solaris.readthedocs.io/en/latest/_modules/solaris/nets/_keras_losses.html

"""

def tf_lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper by Maxim Berman et al.
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard

def k_lovasz_hinge(per_image=False):
    """
    Wrapper for the Lovasz Hinge Loss Function, for use in Keras.
    """

    def lovasz_hinge_flat(y_true, y_pred):
        eps = 1e-12  # for stability
        y_pred = K.clip(y_pred, eps, 1 - eps)
        logits = K.log(y_pred / (1 - y_pred))
        logits = tf.reshape(logits, (-1,))
        y_true = tf.reshape(y_true, (-1,))
        y_true = tf.cast(y_true, logits.dtype)
        signs = 2. * y_true - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(y_true, perm)
        grad = tf_lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return tf.convert_to_tensor(loss, dtype=logits.dtype)  # Ensure the return value is a tensor

    def lovasz_hinge_per_image(y_true, y_pred):
        losses = tf.map_fn(_treat_image, (y_true, y_pred), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
        return tf.convert_to_tensor(loss, dtype=y_pred.dtype)  # Ensure the return value is a tensor

    def _treat_image(ytrue_ypred):
        y_true, y_pred = ytrue_ypred
        y_true, y_pred = tf.expand_dims(y_true, 0), tf.expand_dims(y_pred, 0)
        return lovasz_hinge_flat(y_true, y_pred)

    if per_image:
        return lovasz_hinge_per_image
    else:
        return lovasz_hinge_flat
