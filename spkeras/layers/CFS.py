from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Layer

from tensorflow.python.ops import state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import numpy as np
class CFS(Layer):
    def __init__(self, L=16, **kwargs):
        super(CFS, self).__init__(**kwargs)
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
        @tf.custom_gradient
        def activation_floor(x):
            def grad_fn(dy):
                grad_g = tf.math.greater(x,0)
                grad_l = tf.math.less(x,self.L)
                grad = tf.math.logical_and(grad_g,grad_l)
                grad = tf.cast(grad, dtypes.float32)
                return dy*grad
            return tf.floor(x), grad_fn   
        
        outputs = self._lambda*tf.clip_by_value(activation_floor(inputs*self.L/self._lambda+0.5)/self.L,0,1)
        return outputs

    def get_config(self):
        config = {'L': int(self.L)}
        base_config = super(CFS, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))