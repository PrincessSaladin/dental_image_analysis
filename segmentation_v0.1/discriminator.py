from segmentation.GroupNorm3D import GroupNormalization

from keras.layers import Input, Concatenate, Add
from keras.layers.core import Activation
from keras.layers.convolutional import Conv3D, UpSampling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling3D

from keras.models import Model
from keras.optimizers import Adam

class conv_block:
    def __init__(self, in_channels, out_channels, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def __call__(self, input_):
        x = Conv3D(self.out_channels, (3, 3, 3), padding='same', data_format='channels_first', kernel_initializer='he_normal')(input_)
        #x = BatchNormalization(axis=1)(x)
        x = GroupNormalization(groups=16, axis=1)(x)
        x = Activation('relu')(x)
        return x

class deconv_block:
    def __init__(self, in_channels, out_channels, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def __call__(self, input_1, input_2):
        up_input_1 = UpSampling3D(size=(2, 2, 2), data_format='channels_first')(input_1)
        x = Add()([up_input_1, input_2])
        x = Conv3D(self.out_channels, (3, 3, 3), padding='same', data_format='channels_first', kernel_initializer='he_normal')(x)
        #x = BatchNormalization(axis=1)(x)
        x = GroupNormalization(groups=16, axis=1)(x)
        x = Activation('relu')(x)
        return x

class Head_block:
    def __init__(self, numofbranch, outchannel_per_branch, name='head'):
        self.numofbranch = numofbranch
        self.outchannel_per_branch = outchannel_per_branch
        self.name = name
        assert self.numofbranch == 2
    
    def __call__(self, input1, input2):

        conv1 = Conv3D(self.outchannel_per_branch, (3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_first', kernel_initializer='he_normal', name=self.name + '_conv1')(input1)

        conv2 = Conv3D(self.outchannel_per_branch, (5, 5, 5), strides=(2, 2, 2), padding='same', data_format='channels_first', kernel_initializer='he_normal', name=self.name + '_conv2_1')(input2) # (7, 7, 7)
        conv2 = Conv3D(self.outchannel_per_branch, (3, 3, 3), padding='same', data_format='channels_first', kernel_initializer='he_normal', name=self.name + '_conv2_2')(conv2)

        out = Concatenate(axis=1, name=self.name + '_concat')([conv1, conv2])
        
        return out

class FCDiscriminator:
    def __init__(self, numofclasses=3):
        self.numofclasses = numofclasses
    
    def __call__(self, predict, input1, input2):

        images = Head_block(2, 1, name='d_head')(input1, input2)
        inputs = Concatenate(axis=1)([predict, images])
        
        conv1 = conv_block(5, 32)(inputs)
        conv1 = conv_block(32, 32)(conv1)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format='channels_first')(conv1)
        
        conv2 = conv_block(32, 64)(pool1)
        conv2 = conv_block(64, 64)(conv2)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format='channels_first')(conv2)
        
        conv3 = conv_block(64, 128)(pool2)
        conv3 = conv_block(128, 128)(conv3)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format='channels_first')(conv3)
        
        conv4 = conv_block(128, 128)(pool3)
        
        deconv1 = deconv_block(128, 64)(conv4, conv3)
        deconv2 = deconv_block(64, 32)(deconv1, conv2)
        deconv3 = deconv_block(32, 16)(deconv2, conv1)
        
        output = Conv3D(self.numofclasses, (1, 1, 1), padding='same', data_format='channels_first', kernel_initializer='he_normal')(deconv3)
        sigmoid = Activation('sigmoid')(output)

        #model = Model(inputs=[predict, input1, input2], outputs=softmax)
        #model.compile(optimizer='adam', loss='binary_crossentropy')
        return sigmoid
