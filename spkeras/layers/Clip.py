from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Layer

from tensorflow.python.ops import state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import numpy as np
class Clip(Layer):
    def __init__(self, L=16, **kwargs):
        super(Clip, self).__init__(**kwargs)
        L = math_ops.cast(L, dtypes.float32)   
        self.L = L
        
    def build(self, input_shape):
        self._lambda = self.add_weight(
            shape=(1,),
            name='lambda',
            initializer=tf.keras.initializers.Constant(value=6),
            trainable=True,
        )
        
    def call(self, inputs):      
        outputs = inputs/self._lambda 
        outputs = tf.clip_by_value(outputs,0,1)*self._lambda 
        return outputs

    def get_config(self):
        config = {'L': int(self.L)}
        base_config = super(Clip, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))