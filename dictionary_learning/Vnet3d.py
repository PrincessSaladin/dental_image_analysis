from GroupNorm3D import GroupNormalization
from Dictionary4 import DictionaryLayer

import warnings
import numpy as np
from keras.layers import Layer, Input, Conv3D, Conv3DTranspose, Activation, Add, Concatenate, Lambda, Dense, Reshape
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.models import Model
import keras.backend as K
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops

#def custom_distance_loss(gt, dist, numClasses=3):
#    
#    loss_list = []
#    for idx in range(numClasses):
#        gt_idx = gt[:, idx:idx+1, :, :, :]
#        dist_idx = dist[:, idx:idx+1, :, :, :]
#        
#        #dist_del_idx = tf.concat([dist[:, :idx, :, :, :], dist[:, idx+1:, :, :, :]], axis=1)
#                
#        dist_idx_lab_idx = tf.gather_nd(dist_idx, tf.where(tf.equal(gt_idx, 1)))
#        loss_list.append(K.switch(tf.size(dist_idx_lab_idx)>0, tf.reduce_mean(dist_idx_lab_idx), tf.constant(-1, dtype=tf.float32)))
#    
#    raw_losses = tf.stack(loss_list)
#    final_loss = tf.gather(raw_losses, tf.where(tf.greater_equal(raw_losses, 0)))
#    return tf.reduce_mean(final_loss)

#def custom_distance_loss(gt, dist, numClasses=3):
#    
#    loss_list = []
#    for idx in range(numClasses):
#        gt_idx = gt[:, idx:idx+1, :, :, :]
#        dist_idx = dist[:, idx:idx+1, :, :, :]
#        
#        dist_del_idx = tf.reduce_min(tf.concat([dist[:, :idx, :, :, :], dist[:, idx+1:, :, :, :]], axis=1), axis=1, keepdims=True)
#        
#        dist_idx_lab_idx = tf.gather_nd(dist_idx, tf.where(tf.equal(gt_idx, 1)))
#        dist_del_idx_lab_idx = tf.gather_nd(dist_del_idx, tf.where(tf.equal(gt_idx, 1)))
#        
#        loss_idx_inter = tf.maximum(dist_idx_lab_idx-dist_del_idx_lab_idx + 6, 0)
#        loss_idx_intra = dist_idx_lab_idx
#        
#        loss_list.append(K.switch(tf.size(dist_idx_lab_idx)>0, tf.reduce_mean(loss_idx_inter)+tf.reduce_mean(loss_idx_intra), tf.constant(-1, dtype=tf.float32)))
#    
#    raw_losses = tf.stack(loss_list)
#    final_loss = tf.gather(raw_losses, tf.where(tf.greater_equal(raw_losses, 0)))
#    return K.switch(tf.size(final_loss), tf.reduce_mean(final_loss), tf.constant(0, dtype=tf.float32))

def custom_distance_loss(gt, dist, numClasses=3):

    loss_list = []
    
    margin = 3
    for idx in range(numClasses):
        gt_idx = gt[:, idx:idx+1, :, :, :]
        dist_idx = dist[:, idx:idx+1, :, :, :]
        dist_rest_idx = tf.concat([dist[:, :idx, :, :, :], dist[:, idx+1:, :, :, :]], axis=1)

        dist_idx_lab_idx = tf.gather_nd(dist_idx, tf.where(tf.equal(gt_idx, 1)))
        
        for idx2 in range(numClasses-1):
            
            dist_rest_one_idx = dist_rest_idx[:, idx2:idx2+1, :, :, :]
            dist_rest_one_idx_lab_idx = tf.gather_nd(dist_rest_one_idx, tf.where(tf.equal(gt_idx, 1)))
            loss_idx_inter = tf.maximum(dist_idx_lab_idx-dist_rest_one_idx_lab_idx + margin, 0)
            #loss_list.append(K.switch(tf.size(dist_idx_lab_idx)>0, tf.reduce_mean(loss_idx_inter), tf.constant(-1, dtype=tf.float32)))
            loss_list.append(K.switch(tf.size(dist_idx_lab_idx)>0, loss_idx_inter, -1.0*tf.ones_like(loss_idx_inter, dtype=tf.float32)))

    raw_losses = tf.concat(loss_list, axis=0)
    final_loss = tf.gather(raw_losses, tf.where(tf.greater_equal(raw_losses, 0)))
    return K.switch(tf.size(final_loss), tf.reduce_mean(final_loss), tf.constant(0, dtype=tf.float32))

def custom_categorical_crossentropy(gt, pred):
    gt = tf.cast(gt, tf.float32)
    # manual computation of crossentropy
    epsilon = 1e-3
    pred = tf.clip_by_value(pred, epsilon, 1. - epsilon)
    return - tf.reduce_mean(tf.reduce_sum(gt * tf.log(pred), axis=1), name='crossentropy')

def semantic_loss(gt, semantic):
    loss = K.mean(K.binary_crossentropy(gt, semantic, from_logits=False))
    return loss

def kl_divergence(pred1, pred2):
    epsilon = 1e-3
    pred1 = tf.clip_by_value(pred1, epsilon, 1. - epsilon)
    pred2 = tf.clip_by_value(pred2, epsilon, 1. - epsilon)
    
    loss = tf.reduce_sum(pred1 * tf.log(pred1 / pred2), axis=[-1, -2, -3])
    
    return tf.reduce_sum(loss)

class Head_block:
    def __init__(self, numofbranch, outchannel_per_branch, name='head'):
        self.numofbranch = numofbranch
        self.outchannel_per_branch = outchannel_per_branch
        self.name = name
        assert self.numofbranch == 2

    def __call__(self, input1, input2):

        conv1 = Conv3D(self.outchannel_per_branch, (3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_first', kernel_initializer='he_normal', name=self.name + '_conv1')(input1)
        gn1 = GroupNormalization(groups=self.outchannel_per_branch, axis=1, name=self.name + '_gn1')(conv1)

        conv2 = Conv3D(self.outchannel_per_branch, (5, 5, 5), strides=(2, 2, 2), padding='same', data_format='channels_first', kernel_initializer='he_normal', name=self.name + '_conv2_1')(input2)
        gn2 = GroupNormalization(groups=self.outchannel_per_branch, axis=1, name=self.name + '_gn2_1')(conv2)
        activ2 = Activation('relu', name=self.name + '_activ2')(gn2)
        conv2 = Conv3D(self.outchannel_per_branch, (3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_first', kernel_initializer='he_normal', name=self.name + '_conv2_2')(activ2)
        gn2 = GroupNormalization(groups=self.outchannel_per_branch, axis=1, name=self.name + '_gn2_2')(conv2)

        out = Concatenate(axis=1, name=self.name + '_concat')([gn1, gn2])
        out = Activation('relu', name=self.name + '_activ_all')(out)
        
        return out

def vnet(num_input_channel, base_size, numofclasses, batch_size=1, lambda_dis=0.05, lambda_ce=1., data_format='channels_first'):

    input1 = Input((1, base_size, base_size, base_size))
    input2 = Input((1, 2*base_size, 2*base_size, 2*base_size))
    
    inputs = Head_block(2, 8)(input1, input2)
    
    conv1 = Conv3D(16, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(inputs)
    conv1 = GroupNormalization(groups=16, axis=1)(conv1)
    conv1 = Activation('relu')(conv1)

    identity1 = Conv3D(16, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(inputs)
    identity1 = GroupNormalization(groups=16, axis=1)(identity1)
    identity1 = Activation('relu')(identity1)

    conv1 = Add()([conv1, identity1])
    
    down1 = Conv3D(32, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv1)
    down1 = Activation('relu')(down1)

    conv2 = Conv3D(32, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down1)
    conv2 = GroupNormalization(groups=32, axis=1)(conv2)
    conv2 = Activation('relu')(conv2)

    conv2 = Conv3D(32, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv2)
    conv2 = GroupNormalization(groups=32, axis=1)(conv2)
    conv2 = Activation('relu')(conv2)

    identity2 = Conv3D(32, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down1)
    identity2 = GroupNormalization(groups=32, axis=1)(identity2)
    identity2 = Activation('relu')(identity2)

    conv2 = Add()([conv2, identity2])

    down2 = Conv3D(64, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv2)
    down2 = Activation('relu')(down2)

    conv3 = Conv3D(64, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down2)
    conv3 = GroupNormalization(groups=64, axis=1)(conv3)
    conv3 = Activation('relu')(conv3)

#        conv3 = Conv3D(64, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv3)
#        conv3 = GroupNormalization(groups=64, axis=1)(conv3)
#        conv3 = Activation('relu')(conv3)

    conv3 = Conv3D(64, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv3)
    conv3 = GroupNormalization(groups=64, axis=1)(conv3)
    conv3 = Activation('relu')(conv3)
    
    identity3 = Conv3D(64, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down2)
    identity3 = GroupNormalization(groups=64, axis=1)(identity3)
    identity3 = Activation('relu')(identity3)

    conv3 = Add()([conv3, identity3])

    down3 = Conv3D(128, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv3)
    down3 = Activation('relu')(down3)

    conv4 = Conv3D(128, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down3)
    conv4 = GroupNormalization(groups=128, axis=1)(conv4)
    conv4 = Activation('relu')(conv4)

#        conv4 = Conv3D(128, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv4)
#        conv4 = GroupNormalization(groups=128, axis=1)(conv4)
#        conv4 = Activation('relu')(conv4)

    conv4 = Conv3D(128, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv4)
    conv4 = GroupNormalization(groups=128, axis=1)(conv4)
    conv4 = Activation('relu')(conv4)
    
    identity4 = Conv3D(128, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down3)
    identity4 = GroupNormalization(groups=128, axis=1)(identity4)
    identity4 = Activation('relu')(identity4)

    conv4 = Add()([conv4, identity4])

    down4 = Conv3D(256, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv4)
    down4 = Activation('relu')(down4)

    conv5 = Conv3D(256, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down4)
    conv5 = GroupNormalization(groups=256, axis=1)(conv5)
    conv5 = Activation('relu')(conv5)

#        conv5 = Conv3D(256, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv5)
#        conv5 = GroupNormalization(groups=256, axis=1)(conv5)
#        conv5 = Activation('relu')(conv5)

    conv5 = Conv3D(256, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv5)
    conv5 = GroupNormalization(groups=256, axis=1)(conv5)
    conv5 = Activation('relu')(conv5)
    
    identity5 = Conv3D(256, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(down4)
    identity5 = GroupNormalization(groups=256, axis=1)(identity5)
    identity5 = Activation('relu')(identity5)

    conv5 = Add()([conv5, identity5])

    up1 = Conv3DTranspose(128, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv5)
    concat1 = Concatenate(axis=1)([up1, conv4])

    conv6 = Conv3D(256, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(concat1)
    conv6 = GroupNormalization(groups=256, axis=1)(conv6)
    conv6 = Activation('relu')(conv6)

#        conv6 = Conv3D(256, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv6)
#        conv6 = GroupNormalization(groups=256, axis=1)(conv6)
#        conv6 = Activation('relu')(conv6)

    conv6 = Conv3D(256, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv6)
    conv6 = GroupNormalization(groups=256, axis=1)(conv6)
    conv6 = Activation('relu')(conv6)
    
    identity6 = Conv3D(256, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(up1)
    identity6 = GroupNormalization(groups=256, axis=1)(identity6)
    identity6 = Activation('relu')(identity6)
    
    conv6 = Add()([conv6, identity6])

    up2 = Conv3DTranspose(64, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv6)
    concat2 = Concatenate(axis=1)([up2, conv3])

    conv7 = Conv3D(128, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(concat2)
    conv7 = GroupNormalization(groups=128, axis=1)(conv7)
    conv7 = Activation('relu')(conv7)

#        conv7 = Conv3D(128, kernel_size=5, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv7)
#        conv7 = GroupNormalization(groups=128, axis=1)(conv7)
#        conv7 = Activation('relu')(conv7)

    conv7 = Conv3D(128, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv7)
    conv7 = GroupNormalization(groups=128, axis=1)(conv7)
    conv7 = Activation('relu')(conv7)
    
    identity7 = Conv3D(128, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(up2)
    identity7 = GroupNormalization(groups=128, axis=1)(identity7)
    identity7 = Activation('relu')(identity7)
    
    conv7 = Add()([conv7, identity7])

    up3 = Conv3DTranspose(32, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv7)
    concat3 = Concatenate(axis=1)([up3, conv2])

    conv8 = Conv3D(64, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(concat3)
    conv8 = GroupNormalization(groups=64, axis=1)(conv8)
    conv8 = Activation('relu')(conv8)

    conv8 = Conv3D(64, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv8)
    conv8 = GroupNormalization(groups=64, axis=1)(conv8)
    conv8 = Activation('relu')(conv8)

    identity8 = Conv3D(64, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(up3)
    identity8 = GroupNormalization(groups=64, axis=1)(identity8)
    identity8 = Activation('relu')(identity8)
    
    conv8 = Add()([conv8, identity8])

    up4 = Conv3DTranspose(16, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv8)
    concat4 = Concatenate(axis=1)([up4, conv1])

    conv9 = Conv3D(32, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(concat4)
    conv9 = GroupNormalization(groups=32, axis=1)(conv9)
    conv9 = Activation('relu')(conv9)
    
    identity9 = Conv3D(32, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')(up4)
    identity9 = GroupNormalization(groups=32, axis=1)(identity9)
    identity9 = Activation('relu')(identity9)
    
    conv9 = Add()([conv9, identity9])
    
    # weight #
    weighted_distance = DictionaryLayer(32, numofclasses, axis=1)(conv9)
    
    # output #
    logits = Conv3D(numofclasses, kernel_size=(1, 1, 1), padding='same', data_format='channels_first', kernel_initializer='he_normal')(conv9)
    output1 = Lambda(lambda x: K.softmax(x, axis=1))(logits)

    gt = Input((numofclasses, base_size, base_size, base_size))

    #seg_loss = Lambda(lambda x: custom_categorical_crossentropy(*x), name="ce_loss")([gt, output1])
    dis_loss = Lambda(lambda x: custom_distance_loss(*x, numClasses=numofclasses), name="dis_loss")([gt, weighted_distance])
    seg_loss = Lambda(lambda x: custom_categorical_crossentropy(*x), name="ce_loss")([gt, output1])
    model = Model(inputs=[input1, input2, gt], outputs=[output1, dis_loss, seg_loss])
    model.add_loss(lambda_dis * dis_loss)
    model.add_loss(lambda_ce * seg_loss)
    
    model.compile(optimizer=Adam(lr=0.0001), loss=[None] * len(model.outputs))

    metrics_names = ["dis_loss", "ce_loss"]
    loss_weights = {
        "dis_loss": lambda_dis,
        "ce_loss": lambda_ce,
    }
    
    for name in metrics_names:
        layer = model.get_layer(name)
        loss = (layer.output * loss_weights.get(name, 1.))
        model.metrics_tensors.append(loss)
    
    return model

if __name__ == '__main__':
    model = vnet(1, 64, 3)
    model.summary()














