from segmentation.GroupNorm3D import GroupNormalization

from keras.layers import Input, Lambda, Concatenate, Add, Multiply, Reshape, Dense
from keras.layers.core import Activation
from keras.layers.convolutional import Conv3D, UpSampling3D
from keras.layers.pooling import MaxPooling3D, GlobalAveragePooling3D
from keras import backend as K  
from keras.models import Model

from segmentation.losses import hybrid_loss

class conv_block:
    def __init__(self, in_channels, inter_channels, out_channels, name='cb', **kwargs):
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.name = name
    
    def __call__(self, input_):
        x = Conv3D(self.inter_channels, (3, 3, 3), padding='same', data_format='channels_first', dilation_rate=(2, 2, 2), kernel_initializer='he_normal', name=self.name + '_conv1')(input_)
        x = GroupNormalization(groups=16, axis=1, name=self.name + '_gn1')(x)
        x = Activation('relu', name=self.name + '_acti1')(x)
        # x = PReLU(alpha_initializer='zeros', name=self.name + '_acti1')(x)
        x = Conv3D(self.out_channels, (3, 3, 3), padding='same', data_format='channels_first', dilation_rate=(2, 2, 2), kernel_initializer='he_normal', name=self.name + '_conv2')(x)
        x = GroupNormalization(groups=16, axis=1, name=self.name + '_gn2')(x)
        x = Activation('relu', name=self.name + '_acti2')(x)
        # x = PReLU(alpha_initializer='zeros', name=self.name + '_acti2')(x)
        return x

class PCAModule:
    def __init__(self, in_channels, inter_channels, out_channels, name='cb', **kwargs):
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.name = name
    
    def __call__(self, input_):
        x1 = Conv3D(self.inter_channels, (3, 3, 3), padding='same', data_format='channels_first', dilation_rate=(1, 1, 1), kernel_initializer='he_normal', name=self.name + '_conv1')(input_)
        x1 = GroupNormalization(groups=16, axis=1, name=self.name + '_gn1')(x1)
        x1 = Activation('relu', name=self.name + '_acti1')(x1)
        x2 = Conv3D(self.inter_channels, (3, 3, 3), padding='same', data_format='channels_first', dilation_rate=(2, 2, 2), kernel_initializer='he_normal', name=self.name + '_conv2')(input_)
        x2 = GroupNormalization(groups=16, axis=1, name=self.name + '_gn2')(x2)
        x2 = Activation('relu', name=self.name + '_acti2')(x2)
        x3 = Conv3D(self.inter_channels, (3, 3, 3), padding='same', data_format='channels_first', dilation_rate=(3, 3, 3), kernel_initializer='he_normal', name=self.name + '_conv3')(input_)
        x3 = GroupNormalization(groups=16, axis=1, name=self.name + '_gn3')(x3)
        x3 = Activation('relu', name=self.name + '_acti3')(x3)
        x4 = Conv3D(self.inter_channels, (3, 3, 3), padding='same', data_format='channels_first', dilation_rate=(4, 4, 4), kernel_initializer='he_normal', name=self.name + '_conv4')(input_)
        x4 = GroupNormalization(groups=16, axis=1, name=self.name + '_gn4')(x4)
        x4 = Activation('relu', name=self.name + '_acti4')(x4)
        x5 = Conv3D(self.inter_channels, (3, 3, 3), padding='same', data_format='channels_first', dilation_rate=(5, 5, 5), kernel_initializer='he_normal', name=self.name + '_conv5')(input_)
        x5 = GroupNormalization(groups=16, axis=1, name=self.name + '_gn5')(x5)
        x5 = Activation('relu', name=self.name + '_acti5')(x5)
        
        concat = Concatenate(axis=1, name=self.name + '_concat')([x1, x2, x3, x4, x5])

        x1 = Conv3D(self.inter_channels, (3, 3, 3), padding='same', data_format='channels_first', dilation_rate=(1, 1, 1), kernel_initializer='he_normal', name=self.name + '_conv1_2')(concat)
        x1 = GroupNormalization(groups=16, axis=1, name=self.name + '_gn1_2')(x1)
        x1 = Activation('relu', name=self.name + '_acti1_2')(x1)
        x2 = Conv3D(self.inter_channels, (3, 3, 3), padding='same', data_format='channels_first', dilation_rate=(2, 2, 2), kernel_initializer='he_normal', name=self.name + '_conv2_2')(concat)
        x2 = GroupNormalization(groups=16, axis=1, name=self.name + '_gn2_2')(x2)
        x2 = Activation('relu', name=self.name + '_acti2_2')(x2)
        x3 = Conv3D(self.inter_channels, (3, 3, 3), padding='same', data_format='channels_first', dilation_rate=(3, 3, 3), kernel_initializer='he_normal', name=self.name + '_conv3_2')(concat)
        x3 = GroupNormalization(groups=16, axis=1, name=self.name + '_gn3_2')(x3)
        x3 = Activation('relu', name=self.name + '_acti3_2')(x3)
        x4 = Conv3D(self.inter_channels, (3, 3, 3), padding='same', data_format='channels_first', dilation_rate=(4, 4, 4), kernel_initializer='he_normal', name=self.name + '_conv4_2')(concat)
        x4 = GroupNormalization(groups=16, axis=1, name=self.name + '_gn4_2')(x4)
        x4 = Activation('relu', name=self.name + '_acti4_2')(x4)
        x5 = Conv3D(self.inter_channels, (3, 3, 3), padding='same', data_format='channels_first', dilation_rate=(5, 5, 5), kernel_initializer='he_normal', name=self.name + '_conv5_2')(concat)
        x5 = GroupNormalization(groups=16, axis=1, name=self.name + '_gn5_2')(x5)
        x5 = Activation('relu', name=self.name + '_acti5_2')(x5)
        
        x = Concatenate(axis=1, name=self.name + '_concat_2')([x1, x2, x3, x4, x5])
        
        return x

class res_block:
    def __init__(self, in_channels, inter_channels, out_channels, **kwargs):
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels

    def __call__(self, input_):

        x = Conv3D(self.inter_channels, (3, 3, 3), padding='same', data_format='channels_first', kernel_initializer='he_normal')(input_)
        x = GroupNormalization(groups=16, axis=1)(x)
        x = Activation('relu')(x)

        x = Conv3D(self.out_channels, (3, 3, 3), padding='same', data_format='channels_first', kernel_initializer='he_normal')(x)
        x = GroupNormalization(groups=16, axis=1)(x)

        if self.in_channels == self.out_channels:
            shortcut = input_
            x = Add()([shortcut, x])

        else:
            y = Conv3D(self.out_channels, (1, 1, 1), padding='same', data_format='channels_first', kernel_initializer='he_normal')(input_)
            y = GroupNormalization(groups=16, axis=1)(y)
            x = Add()([y, x])
        x = Activation('relu')(x)
        return x

class deconv_block:
    def __init__(self, in_channels, inter_channels, out_channels, name='db', **kwargs):
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.name = name
    
    def __call__(self, input_, input_2):
        up_input_ = UpSampling3D(size=(2, 2, 2), data_format='channels_first', name=self.name + '_upsample')(input_)
        x = Concatenate(axis=1, name=self.name + '_concat1')([up_input_, input_2])
        x = Conv3D(self.inter_channels, (3, 3, 3), padding='same', data_format='channels_first', dilation_rate=(2, 2, 2), kernel_initializer='he_normal', name=self.name + '_conv1')(x)
        x = GroupNormalization(groups=16, axis=1, name=self.name + '_gn1')(x)
        x = Activation('relu', name=self.name + '_acti1')(x)
        # x = PReLU(alpha_initializer='zeros', name=self.name + '_acti1')(x)
        x = Conv3D(self.out_channels, (3, 3, 3), padding='same', data_format='channels_first', dilation_rate=(2, 2, 2), kernel_initializer='he_normal', name=self.name + '_conv2')(x)
        x = GroupNormalization(groups=16, axis=1, name=self.name + '_gn2')(x)
        x = Activation('relu', name=self.name + '_acti2')(x)
        # x = PReLU(alpha_initializer='zeros', name=self.name + '_acti2')(x)
        return x

class deconv_block_add:
    def __init__(self, in_channels, inter_channels, out_channels, name='db', **kwargs):
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.name = name
    
    def __call__(self, input1, input2):
        up_input1 = UpSampling3D(size=(2, 2, 2), data_format='channels_first', name=self.name + '_upsample')(input1)
        x = Conv3D(self.inter_channels, (3, 3, 3), padding='same', data_format='channels_first', dilation_rate=(2, 2, 2), kernel_initializer='he_normal', name=self.name + '_conv1')(input2)
        x = GroupNormalization(groups=16, axis=1, name=self.name + '_gn1')(x)
        x = Activation('relu', name=self.name + '_acti1')(x)
        x = Add(name=self.name + '_add')([up_input1, x])
        # x = PReLU(alpha_initializer='zeros', name=self.name + '_acti1')(x)
        x = Conv3D(self.out_channels, (3, 3, 3), padding='same', data_format='channels_first', dilation_rate=(2, 2, 2), kernel_initializer='he_normal', name=self.name + '_conv2')(x)
        x = GroupNormalization(groups=16, axis=1, name=self.name + '_gn2')(x)
        x = Activation('relu', name=self.name + '_acti2')(x)
        # x = PReLU(alpha_initializer='zeros', name=self.name + '_acti2')(x)
        return x

class deconv_res_block:
    def __init__(self, in_channels, inter_channels, out_channels):
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels

    def __call__(self, input_1, input_2):
        up_input_ = UpSampling3D(size=(2, 2, 2), data_format='channels_first')(input_1)
        x = Concatenate(axis=1)([up_input_, input_2])
        
        y = res_block(self.in_channels, self.inter_channels, self.out_channels)(x)

        return y

class Head_block:
    def __init__(self, numofbranch, outchannel_per_branch, name='head'):
        self.numofbranch = numofbranch
        self.outchannel_per_branch = outchannel_per_branch
        self.name = name
        assert self.numofbranch == 3
    
    def __call__(self, input1, input2, input3):

        conv1 = Conv3D(self.outchannel_per_branch, (3, 3, 3), strides=(1, 1, 1), padding='same', data_format='channels_first', kernel_initializer='he_normal', name=self.name + '_conv1')(input1)
        gn1 = GroupNormalization(groups=16, axis=1, name=self.name + '_gn1')(conv1)

        conv2 = Conv3D(self.outchannel_per_branch, (5, 5, 5), strides=(2, 2, 2), padding='same', data_format='channels_first', kernel_initializer='he_normal', name=self.name + '_conv2_1')(input2)
        gn2 = GroupNormalization(groups=16, axis=1, name=self.name + '_gn2_1')(conv2)
        activ2 = Activation('relu', name=self.name + '_activ2')(gn2)
        conv2 = Conv3D(self.outchannel_per_branch, (3, 3, 3), padding='same', data_format='channels_first', dilation_rate=(2, 2, 2), kernel_initializer='he_normal', name=self.name + '_conv2_2')(activ2)
        gn2 = GroupNormalization(groups=16, axis=1, name=self.name + '_gn2_2')(conv2)
        
        conv3 = Conv3D(self.outchannel_per_branch, (7, 7, 7), strides=(4, 4, 4), padding='same', data_format='channels_first', kernel_initializer='he_normal', name=self.name + '_conv3_1')(input3)
        gn3 = GroupNormalization(groups=16, axis=1, name=self.name + '_gn3_1')(conv3)
        activ3 = Activation('relu', name=self.name + '_activ3_1')(gn3)
        conv3 = Conv3D(self.outchannel_per_branch, (3, 3, 3), padding='same', data_format='channels_first', dilation_rate=(2, 2, 2), kernel_initializer='he_normal', name=self.name + '_conv3_2')(activ3)
        gn3 = GroupNormalization(groups=16, axis=1, name=self.name + '_gn3_2')(conv3)
        activ3 = Activation('relu', name=self.name + '_activ3_2')(gn3)
        conv3 = Conv3D(self.outchannel_per_branch, (3, 3, 3), padding='same', data_format='channels_first', dilation_rate=(2, 2, 2), kernel_initializer='he_normal', name=self.name + '_conv3_3')(activ3)
        gn3 = GroupNormalization(groups=16, axis=1, name=self.name + '_gn3_3')(conv3)

        out = Concatenate(axis=1, name=self.name + '_concat')([gn1, gn2, gn3])
        out = Activation('relu', name=self.name + '_activ_all')(out)
        
        return out

class PCANet:
    def __init__(self, numofclasses=3, name='seg'):
        self.numofclasses = numofclasses
        self.name = name
    
    def __call__(self, input1, input2, input3):

        inputs = Head_block(3, 16, name=self.name + '_head')(input1, input2, input3)
        
        conv1 = PCAModule(48, 16, 80, name=self.name + 'en_cb1')(inputs)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format='channels_first', name=self.name + 'en_maxpool1')(conv1)
        
        conv2 = PCAModule(80, 16, 80, name=self.name + 'en_cb2')(pool1)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format='channels_first', name=self.name + 'en_maxpool2')(conv2)
        
        conv3 = PCAModule(80, 16, 80, name=self.name + 'en_cb3')(pool2)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format='channels_first', name=self.name + 'en_maxpool3')(conv3)
        
        conv4 = PCAModule(80, 16, 80, name=self.name + 'en_cb4')(pool3)
        
        deconv1_1 = deconv_block_add(160, 80, 80, name=self.name + 'dbseg_1_1')(conv4, conv3) # 256 + 256 = 512
        deconv1_2 = deconv_block_add(160, 80, 80, name=self.name + 'dbseg_1_2')(deconv1_1, conv2) # 256 ?
        deconv1_3 = deconv_block_add(160, 80, 80, name=self.name + 'dbseg_1_3')(deconv1_2, conv1) # 256 ?

        ## Output ##
        deconv1_3 = Conv3D(80, (3, 3, 3), padding='same', data_format='channels_first', kernel_initializer='he_normal', name=self.name + 'seg_conv1')(deconv1_3)
        output1 = Conv3D(self.numofclasses, (1, 1, 1), padding='same', data_format='channels_first', kernel_initializer='he_normal', name=self.name + 'seg_conv2')(deconv1_3)
        lambda1 = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 3, 4, 1)), name=self.name + 'seg_lambda1')(output1)
        softmax = Activation('softmax', name=self.name + 'seg_acti')(lambda1)
        output1 = Lambda(lambda x: K.permute_dimensions(x, (0, 4, 1, 2, 3)), name=self.name + 'seg')(softmax)

        return output1
