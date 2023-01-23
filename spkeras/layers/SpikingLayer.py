from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

from tensorflow.python.ops import state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.framework import ops

class SpikingLayer(Layer):
    _USE_V2_BEHAVIOR = True
    def __init__(self,T,layer='hidden',vthr=1,alpha=0, static=False, trainable=True, initial_value=1, method=None, neuron='LIF', **kwargs):
        super(SpikingLayer, self).__init__(**kwargs)
        self.vthr = vthr
        self.T = T
        self.layer = layer
        self.static = static
        self.trainable = trainable if neuron == 'LIF' else False
        self.initial_value = initial_value
        self.neuron = neuron
        self.alpha = alpha
        
    def build(self,input_shape): 
        if self.layer == 'hidden':
            self.fleak = self.add_weight(
                shape=(1,),
                initializer=tf.keras.initializers.Constant(value=self.initial_value),
                constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0, axis=0),
                trainable=self.trainable,
            )          

    def _get_training_value(self, training=None):
        if training is None:
            training = K.learning_phase()
        if self._USE_V2_BEHAVIOR:
            if isinstance(training, int):
                training = bool(training)
            if not self.trainable:
                # When the layer is not trainable, it overrides the value passed from
                # model.
                training = False
        return training
    
    #@tf.custom_gradient
    #def spikeforward(self,x):
    #    spike = tf.math.maximum(tf.math.sign(x),0)
    #    def grad_fn(dy):
    #        _x = tf.math.less(tf.math.abs(x),1/2)
    #        _x = tf.cast(_x,"float32")
    #        grad = tf.math.sign(_x)/2
    #        return dy*grad
    #    return spike, grad_fn   
    @tf.custom_gradient
    def spikeforward(self,x):
        spike = tf.math.maximum(tf.math.sign(x),0)
        def grad_fn(dy):
            grad = tf.math.maximum(1-tf.math.abs(x),0)
            return dy*grad
        return spike, grad_fn 
    
    #@tf.custom_gradient
    #def spikeforward(self,x):
    #    import numpy as np
    #    spike = tf.math.atan(np.pi*x)/np.pi+1/2
    #    def grad_fn(dy):
    #        grad = 1/(1+(np.pi*x)**2)
    #        return dy*grad
    #    return spike, grad_fn 
    def call(self, inputs, training=None):
        if self.layer == 'input':
            if self.static:
                inputs = tf.expand_dims(inputs, axis=1, name=None)
                shape = inputs.shape
                m = np.ones(len(shape)) 
                m[1] = self.T
                shape = shape*m
                inputs = tf.broadcast_to(inputs,shape)

            shape = [inputs.shape[0]*inputs.shape[1]]
            shape.extend(inputs.shape[2:])
            return tf.reshape(inputs,shape)
        elif self.layer == 'output_by_pass':
            shape = [int(inputs.shape[0]/self.T),int(self.T)]
            shape.extend(inputs.shape[1:])
            return  tf.reshape(inputs,shape)
        elif self.layer == 'output':
            shape = [int(inputs.shape[0]/self.T),int(self.T)]
            shape.extend(inputs.shape[1:])
            inputs = tf.reshape(inputs,shape)
            return  tf.math.reduce_mean(inputs,axis=[1])
        else:
            import numpy as np
            rank = len(inputs.shape)-1
            rank = np.clip(rank,3,rank)
            axis = np.arange(rank-1)+2     
            xmax = tf.math.reduce_min(inputs)
            sigma = tf.math.reduce_mean(tf.math.maximum(inputs,0)**2+0.00001,axis=axis)**0.5
            sigma = tf.math.reduce_mean(sigma,axis=1)
            sigma = tf.math.reduce_min(sigma)
            r = xmax/sigma
            r = tf.math.maximum(r,1)
            
            shape = [int(inputs.shape[0]/self.T),int(self.T)]
            shape.extend(inputs.shape[1:])
            inputs = tf.reshape(inputs,shape)
            spike_out = []
            vmem = 0
            #_loss=[]
            fleak = 1 if self.neuron == 'IF' else tf.math.sigmoid(self.fleak)
            for t in range(inputs.shape[1]):
                #fleak = self.fleak
                vmem = vmem*(fleak)+inputs[:,t]
                spike = self.spikeforward(vmem-self.vthr)
                vmem = vmem*(1-spike)
                #if self.layer == 'output':
                #    spike_out.append(inputs[:,t]) 
                #else:
                spike_out.append(spike) 
            outputs = tf.stack(spike_out,axis=1)
            if self.layer != 'output':
                shape = [inputs.shape[0]*inputs.shape[1]]
                shape.extend(inputs.shape[2:])
                outputs = tf.reshape(outputs,shape)

                self.add_loss(self.alpha*tf.math.log(r))
        return outputs  
    
    def _assign_new_value(self, variable, value):
        with K.name_scope('AssignNewValue') as scope:
            with ops.colocate_with(variable):
                return state_ops.assign(variable, value, name=scope)
            
    def get_config(self):
        config = {'T':self.T,
                  'vthr':self.vthr,
                  'neuron':self.neuron,
                  'initial_value':self.initial_value,
                  'trainable':self.trainable,
                  'layer':self.layer}
        base_config = super(SpikingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))