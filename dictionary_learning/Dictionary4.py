from keras.engine import Layer, InputSpec
from keras import initializers
from keras import backend as K
import tensorflow as tf

class DictionaryLayer(Layer):

    def __init__(self,
                 numChannels,
                 numCodewords,
                 axis=1,
                 **kwargs):
        super(DictionaryLayer, self).__init__(**kwargs)
        self.axis = axis
        self.numChannels = numChannels
        self.numCodewords = numCodewords
        self.codewords_initializer = initializers.TruncatedNormal(mean=0., stddev=1.)
        self.scale_dim_initializer = initializers.RandomUniform(minval=-0.1, maxval=0.1)
        self.scale_center_initializer = initializers.RandomUniform(minval=-0.1, maxval=0.1)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})

        self.codewords = self.add_weight(shape=(1, self.numCodewords, self.numChannels),
                                     name='codewords',
                                     initializer=self.codewords_initializer)

        self.scale_dim = self.add_weight(shape=(1, 1, 1, self.numChannels,),
                                    name='scale_dim',
                                    initializer=self.scale_dim_initializer)

        self.scale_center = self.add_weight(shape=(1, 1, self.numCodewords,),
                                            name='scale_center',
                                            initializer=self.scale_center_initializer)

        self.built = True

    def call(self, inputs, training=None):    
        input_shape = K.int_shape(inputs)
        #print('\ninputs: ', inputs)

        # Prepare broadcasting shape.
        ndim = len(input_shape)
        assert ndim == 5, 'Only 5D inputs are supported'
        
        if self.axis==1:
            inputs  = tf.transpose(inputs, [0, 2, 3, 4, 1])
            shape = K.int_shape(inputs)
            inputs  = tf.reshape(inputs, [tf.shape(inputs)[0], shape[1]*shape[2]*shape[3], shape[4], 1])
            inputs  = tf.transpose(inputs, [0, 1, 3, 2])
        else:
            shape = K.int_shape(inputs)
            inputs  = tf.reshape(inputs, [tf.shape(inputs)[0], shape[1]*shape[2]*shape[3], shape[4], 1])
            inputs  = tf.transpose(inputs, [0, 1, 3, 2])

        #print('\ninputs: ', inputs)
        #print('\ncodewords_0: ', self.codewords)
        codewords = tf.tile(self.codewords, [tf.shape(inputs)[0], 1, 1])
        #print('\ncodewords_1: ', codewords)
        #assert 0

        # Residual vectors
        R = inputs - tf.expand_dims(codewords, axis=1)
        #print('\nR: ', R)
        R_square = tf.square(R)
        #print('\nR_square: ', R_square)
        
        #print('\ntf.math.exp(self.scale_dim): ', tf.math.exp(self.scale_dim))
        
        epsilon = 1e-6
        weighted_dis = tf.sqrt(tf.reduce_sum(tf.multiply(R_square, tf.math.exp(self.scale_dim)), axis=-1)+epsilon)
        #print('\nweighted_dis: ', weighted_dis)
        #assert 0        
        scale_center = tf.tile(self.scale_center, [tf.shape(inputs)[0], 1, 1])
        weighted_dis = tf.multiply(weighted_dis, tf.math.exp(scale_center))
        #print('\nweighted_dis: ', weighted_dis)
        #assert 0         
        #weight = tf.nn.softmax(-1.0*weighted_dis, axis=-1)

        if self.axis == 1:
            weighted_dis = tf.transpose(weighted_dis, [0, 2, 1])
            weighted_dis = tf.reshape(weighted_dis, [tf.shape(weighted_dis)[0], self.numCodewords, input_shape[2], input_shape[3], input_shape[4]])
        else:
            weighted_dis = tf.reshape(weighted_dis, [tf.shape(weighted_dis)[0], input_shape[1], input_shape[2], input_shape[3], self.numCodewords])
        
        return weighted_dis
    
    def get_config(self):
        config = {
            'axis': self.axis,
            'numChannels': self.numChannels,
            'numCodewords': self.numCodewords,
            'codewords_initializer': initializers.serialize(self.codewords_initializer),
            'scale_dim_initializer': initializers.serialize(self.scale_dim_initializer),
            'scale_center_initializer': initializers.serialize(self.scale_center_initializer),
        }
        base_config = super(DictionaryLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if self.axis == 1:
            return (None, self.numCodewords, input_shape[2], input_shape[3], input_shape[4]) # [input_shape, (None, self.numCodewords, input_shape[2], input_shape[3], input_shape[4])]
        else:
            return (None, input_shape[1], input_shape[2], input_shape[3], self.numCodewords) # [input_shape, (None, input_shape[1], input_shape[2], input_shape[3], self.numCodewords)]

if __name__ == '__main__':
    from keras.layers import Input

    inputs = Input([32, 8, 8, 8])
    output = DictionaryLayer(32, 16, axis=1)(inputs)
