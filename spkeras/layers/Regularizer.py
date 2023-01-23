from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Layer

from tensorflow.python.ops import state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes

class Regularizer(Layer):
    def __init__(self, scale=0, **kwargs):
        super(Regularizer, self).__init__(**kwargs)
        scale = math_ops.cast(scale, dtypes.float32)   
        self.scale = scale
        
    def call(self, inputs):
        import numpy as np
        rank = len(inputs.shape)
        rank = np.clip(rank,3,rank)
        axis = np.arange(rank-1)+1    
        xmax = tf.math.reduce_max(inputs)
        sigma = tf.math.reduce_mean(tf.math.maximum(inputs,0)**2,axis=axis)**0.5
        sigma = tf.math.reduce_min(sigma)
        r = xmax/sigma
        r = tf.math.maximum(r,1)
        outputs = inputs
        loss = self.scale*tf.math.log(r)
        self.add_loss(loss)
        return outputs

    def get_config(self):
        config = {'scale': self.scale}
        base_config = super(Regularizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))