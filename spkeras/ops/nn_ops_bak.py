# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrappers for primitive Neural Net (NN) Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numbers
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables as variables_lib
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_nn_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.platform import device_context
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup

from tensorflow.python.util.tf_export import tf_export

# Aliases for some automatically-generated names.
local_response_normalization = gen_nn_ops.lrn

# pylint: disable=protected-access

# Acceptable channels last formats (robust to H, W, D order).
_CHANNELS_LAST_FORMATS = frozenset({
    "NWC", "NHC", "NHWC", "NWHC", "NDHWC", "NDWHC", "NHDWC", "NHWDC", "NWDHC",
    "NWHDC"
})



def _get_noise_shape(x, noise_shape):
  # If noise_shape is none return immediately.
  if noise_shape is None:
    return array_ops.shape(x)

  try:
    # Best effort to figure out the intended shape.
    # If not possible, let the op to handle it.
    # In eager mode exception will show up.
    noise_shape_ = tensor_shape.as_shape(noise_shape)
  except (TypeError, ValueError):
    return noise_shape

  if x.shape.dims is not None and len(x.shape.dims) == len(noise_shape_.dims):
    new_dims = []
    for i, dim in enumerate(x.shape.dims):
      if noise_shape_.dims[i].value is None and dim.value is not None:
        new_dims.append(dim.value)
      else:
        new_dims.append(noise_shape_.dims[i].value)
    return tensor_shape.TensorShape(new_dims)

  return noise_shape

def normdropout(x, n=0, c_range=None, drop_center=None, keep_prob=None, noise_shape=None, seed=None, name=None,
            rate=None):
  """Computes dropout.
  For each element of `x`, with probability `rate`, outputs `0`, and otherwise
  scales up the input by `1 / (1-rate)`. The scaling is such that the expected
  sum is unchanged.
  By default, each element is kept or dropped independently.  If `noise_shape`
  is specified, it must be
  [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
  will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
  and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
  kept independently and each row and column will be kept or not kept together.
  Args:
    x: A floating point tensor.
    keep_prob: (deprecated) A deprecated alias for `(1-rate)`.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.
    name: A name for this operation (optional).
    rate: A scalar `Tensor` with the same type as `x`. The probability that each
      element of `x` is discarded.
  Returns:
    A Tensor of the same shape of `x`.
  Raises:
    ValueError: If `rate` is not in `[0, 1)` or if `x` is not a floating
      point tensor.
  """
  try:
    keep = 1. - keep_prob if keep_prob is not None else None
  except TypeError:
    raise ValueError("keep_prob must be a floating point number or Tensor "
                     "(got %r)" % keep_prob)

  rate = deprecation.deprecated_argument_lookup(
      "rate", rate,
      "keep_prob", keep)

  if rate is None:
    raise ValueError("You must provide a rate to dropout.")

  return normdropout_v2(x, rate, c_range=c_range, n=n, drop_center=drop_center, noise_shape=noise_shape, seed=seed, name=name)


@tf_export("nn.dropout", v1=[])
@dispatch.add_dispatch_support
def normdropout_v2(x, rate, c_range=0.5, n=0, drop_center=True, noise_shape=None, seed=None, name=None):
  """Computes dropout: randomly sets elements to zero to prevent overfitting.
  Note: The behavior of dropout has changed between TensorFlow 1.x and 2.x.
  When converting 1.x code, please use named arguments to ensure behavior stays
  consistent.
  See also: `tf.keras.layers.Dropout` for a dropout layer.
  [Dropout](https://arxiv.org/abs/1207.0580) is useful for regularizing DNN
  models. Inputs elements are randomly set to zero (and the other elements are
  rescaled). This encourages each node to be independently useful, as it cannot
  rely on the output of other nodes.
  More precisely: With probability `rate` elements of `x` are set to `0`.
  The remaining elements are scaled up by `1.0 / (1 - rate)`, so that the
  expected value is preserved.
  >>> tf.random.set_seed(0)
  >>> x = tf.ones([3,5])
  >>> tf.nn.dropout(x, rate = 0.5, seed = 1).numpy()
  array([[2., 0., 0., 2., 2.],
       [2., 2., 2., 2., 2.],
       [2., 0., 2., 0., 2.]], dtype=float32)
  >>> tf.random.set_seed(0)
  >>> x = tf.ones([3,5])
  >>> tf.nn.dropout(x, rate = 0.8, seed = 1).numpy()
  array([[0., 0., 0., 5., 5.],
       [0., 5., 0., 5., 0.],
       [5., 0., 5., 0., 5.]], dtype=float32)
  >>> tf.nn.dropout(x, rate = 0.0) == x
  <tf.Tensor: shape=(3, 5), dtype=bool, numpy=
    array([[ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True]])>
  By default, each element is kept or dropped independently.  If `noise_shape`
  is specified, it must be
  [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
  will make independent decisions. This is useful for dropping whole
  channels from an image or sequence. For example:
  >>> tf.random.set_seed(0)
  >>> x = tf.ones([3,10])
  >>> tf.nn.dropout(x, rate = 2/3, noise_shape=[1,10], seed=1).numpy()
  array([[0., 0., 0., 3., 3., 0., 3., 3., 3., 0.],
       [0., 0., 0., 3., 3., 0., 3., 3., 3., 0.],
       [0., 0., 0., 3., 3., 0., 3., 3., 3., 0.]], dtype=float32)
  Args:
    x: A floating point tensor.
    rate: A scalar `Tensor` with the same type as x. The probability
      that each element is dropped. For example, setting rate=0.1 would drop
      10% of input elements.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    seed: A Python integer. Used to create random seeds. See
      `tf.random.set_seed` for behavior.
    name: A name for this operation (optional).
  Returns:
    A Tensor of the same shape of `x`.
  Raises:
    ValueError: If `rate` is not in `[0, 1)` or if `x` is not a floating point
      tensor. `rate=1` is disallowed, because the output would be all zeros,
      which is likely not what was intended.
  """
  with ops.name_scope(name, "dropout", [x]) as name:
    is_rate_number = isinstance(rate, numbers.Real)
    _rate = rate
    if is_rate_number and (rate < 0 or rate >= 1):
      raise ValueError("rate must be a scalar tensor or a float in the "
                       "range [0, 1), got %g" % rate)
    x = ops.convert_to_tensor(x, name="x")
    x_dtype = x.dtype
    if not x_dtype.is_floating:
      raise ValueError("x has to be a floating point tensor since it's going "
                       "to be scaled. Got a %s tensor instead." % x_dtype)
    is_executing_eagerly = context.executing_eagerly()
    noise_shape = _get_noise_shape(x, noise_shape)
    contraints = tf.zeros(noise_shape)
    mean = tf.math.reduce_mean(x)
    xmax = tf.math.reduce_max(x)
    _xmax = xmax-mean
    if n == 0:
      random_tensor = random_ops.random_uniform(
          noise_shape, seed=seed, dtype=x_dtype) 
      drop_mask = random_tensor <= rate
      contraints = gen_math_ops.cast(drop_mask, x_dtype) 
    else:
      for _n in range(n):
        random_tensor = random_ops.random_uniform(
          noise_shape, seed=seed, dtype=x_dtype)
        drop_mask = random_tensor <= rate/(n-_n)
        drop_mask = gen_math_ops.cast(drop_mask, x_dtype)
        bound_low = x < _n*mean/n
        bound_low = gen_math_ops.cast(bound_low, x_dtype)  
        bound_up = x <= (_n+1)*mean/n
        bound_up = gen_math_ops.cast(bound_up, x_dtype)   
        _contraints = gen_math_ops.mul(bound_up - bound_low,drop_mask)
        contraints +=  _contraints
      for _n in range(n):
        random_tensor = random_ops.random_uniform(
          noise_shape, seed=seed, dtype=x_dtype)
        drop_mask = random_tensor <= rate/(n-_n)
        drop_mask = gen_math_ops.cast(drop_mask, x_dtype)
        bound_low = x <= xmax - (_n+1)*_xmax/n 
        bound_low = gen_math_ops.cast(bound_low, x_dtype)  
        bound_up = x <= xmax - _n*_xmax/n 
        bound_up = gen_math_ops.cast(bound_up, x_dtype)   
        _contraints = gen_math_ops.mul(bound_up - bound_low,drop_mask)
        contraints +=  _contraints
    #calculate
    one_tensor = constant_op.constant(1, dtype=x_dtype)
    contraints = contraints if drop_center else gen_math_ops.sub(one_tensor,contraints)
    contraints_sum = tf.math.reduce_sum(gen_math_ops.cast(contraints, x_dtype))
    _noise_shape = gen_math_ops.cast(noise_shape, x_dtype)
    contraints_rate = gen_math_ops.real_div(contraints_sum,tf.math.reduce_prod(_noise_shape))
    rate = contraints_rate
    variance = tf.math.reduce_variance(x)  
    tf.print('max:',xmax)
    tf.print('mean:',mean)
    tf.print('variance:',variance)
    if not tensor_util.is_tensor(rate):
      if is_rate_number:
        keep_prob = 1 - rate
        scale = 1 / keep_prob
        scale = ops.convert_to_tensor(scale, dtype=x_dtype)
        ret = gen_math_ops.mul(x, scale)
      else:
        raise ValueError("rate is neither scalar nor scalar tensor %r" % rate)
    else:
      rate.get_shape().assert_has_rank(0)
      rate_dtype = rate.dtype
      if rate_dtype != x_dtype:
        if not rate_dtype.is_compatible_with(x_dtype):
          raise ValueError(
              "Tensor dtype %s is incomptaible with Tensor dtype %s: %r" %
              (x_dtype.name, rate_dtype.name, rate))
        rate = gen_math_ops.cast(rate, x_dtype, name="rate")
      one_tensor = constant_op.constant(1, dtype=x_dtype)
      ret = gen_math_ops.real_div(x, gen_math_ops.sub(one_tensor, rate))

    
    # Sample a uniform distribution on [0.0, 1.0) and select values larger
    # than rate.
    #
    # NOTE: Random uniform can only generate 2^23 floats on [1.0, 2.0)
    # and subtract 1.0.
    #random_tensor = random_ops.random_uniform(
    #    noise_shape, seed=seed, dtype=x_dtype)
    # NOTE: if (1.0 + rate) - 1 is equal to rate, then that float is selected,
    # hence a >= comparison is used.
    drop_mask = contraints
    keep_mask = gen_math_ops.sub(one_tensor,drop_mask)
    ret = gen_math_ops.mul(ret, keep_mask)

    if not is_executing_eagerly:
      ret.set_shape(x.get_shape())
    return ret
