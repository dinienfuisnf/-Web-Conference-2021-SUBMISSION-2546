import tensorflow as tf

import keras
from keras import backend as K
from tensorflow.keras.layers import Layer, Dense, BatchNormalization
from keras import activations, initializers, regularizers, constraints
tf.compat.v1.disable_eager_execution()
class GNNLayer(Layer):
    r"""
    A GraphSAGE layer modified with softmax aggregator.
    This layer computes:
    $$
        \Z = \big[ \textrm{AGGREGATE}(\X) \| \X \big] \W + \b; \\
        \Z = \frac{\Z}{\|\Z\|}
    $$
    where \( \textrm{AGGREGATE} \) is a function to aggregate a node's
    neighbourhood. The supported aggregation methods are: sum, mean,
    max, min, and product.
    **Input**
    - Node features of shape `(N, F)`;
    - Binary adjacency matrix of shape `(N, N)`.
    **Output**
    - Node features with the same shape as the input, but with the last
    dimension changed to `channels`.
    **Arguments**
    - `channels`: number of output channels;
    - `activation`: activation function to use;
    - `use_bias`: whether to add a bias to the linear transformation;
    - `kernel_initializer`: initializer for the kernel matrix;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the kernel matrix;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    - `bias_constraint`: constraint applied to the bias vector.
    """

    def __init__(self,
                 channels,
                 BNs,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        # super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.channels = channels # 32*32
        self.BNs = BNs
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.delta = tf.constant(1e-7, tf.float32)

        super().__init__(**kwargs)

    def build(self, input_shape): # None*10000*32,None*10000*12
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1] #32
        self.kernel_1 = self.add_weight(shape=(input_dim, self.channels[0]),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.kernel_2 = self.add_weight(shape=(self.channels[0], self.channels[1]),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)                         
        if self.use_bias:
            self.bias_1 = self.add_weight(shape=(self.channels[0],),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

            self.bias_2 = self.add_weight(shape=(self.channels[1],),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs):
        # inputs=[feature,visit_]
        features_people = inputs[0] # None * pop * f
        fltr = inputs[1]    # None * pop * region

        if not K.is_sparse(fltr):
            fltr = tf.sparse.from_dense(tf.transpose(fltr, [0,2,1]))
        
        # To regions
        weight_softmax = tf.sparse.to_dense(tf.sparse.softmax(fltr)) # None * region * pop
        features_region = tf.matmul(weight_softmax, features_people) # None*region*f
        features_region = K.dot(features_region, self.kernel_1)  # None * region * d1

        if self.use_bias:
            features_region = K.bias_add(features_region, self.bias_1)
        if self.activation is not None:
            features_region = self.activation(features_region)
        
        features_region = self.BNs[0](features_region)

        # To people
        weight_softmax = tf.transpose(weight_softmax, [0,2,1]) # None * pop * region
        features_people = tf.matmul(weight_softmax, features_region) # None*pop*d1

        features_people = K.dot(features_people, self.kernel_2)  # None * pop * d2
        if self.use_bias:
            features_people = K.bias_add(features_people, self.bias_2)
        if self.activation is not None:
            features_people = self.activation(features_people)
        
        features_people = self.BNs[1](features_people)

        return features_people
    
    def compute_output_shape(self, input_shape):
        features_shape = input_shape[0]
        output_shape = features_shape[:-1] + (self.channels[-1],)
        return output_shape
    
    def get_config(self):
        config = {
            'channels': self.channels,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    