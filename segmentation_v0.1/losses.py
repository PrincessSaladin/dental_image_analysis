import keras.backend as K
import tensorflow as tf

def custom_mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def dice_loss(gt, pred, mask):
    mask = tf.cast(mask, tf.bool)
    
    gt = tf.cast(gt, tf.float32)
    gt = tf.boolean_mask(gt, mask)
    
    pred = tf.boolean_mask(pred, mask)
    
    sum_gt = tf.reduce_sum(gt) #tf.reduce_sum(gt, axis=(1, 2, 3, 4))
    sum_pred = tf.reduce_sum(pred) #tf.reduce_sum(pred, axis=(1, 2, 3, 4))

    # sum_dot = tf.reduce_sum(gt * pred, axis=(1, 2, 3, 4))
    sum_dot = tf.reduce_sum(gt * pred)
    epsilon = 1e-6
    dice = (2. * sum_dot + epsilon) / (sum_gt + sum_pred + epsilon)
    dice_loss = 1 - tf.reduce_mean(dice, name='dice_loss')
    return dice_loss

def custom_categorical_crossentropy(gt, pred, mask):
    mask = tf.cast(mask, tf.bool)
    
    gt = tf.cast(gt, tf.float32)
    gt = tf.boolean_mask(gt, mask)
    
    pred = tf.boolean_mask(pred, mask)
    # manual computation of crossentropy
    epsilon = 1e-6
    pred = tf.clip_by_value(pred, epsilon, 1. - epsilon)
    return - tf.reduce_mean(gt * tf.math.log(pred) + (1. - gt) * tf.math.log(1. - pred), name='crossentropy')

def hybrid_loss(gt, pred, mask):
    return dice_loss(gt, pred, mask) + custom_categorical_crossentropy(gt, pred, mask)
