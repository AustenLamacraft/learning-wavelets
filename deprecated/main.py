from __future__ import absolute_import, division, print_function, unicode_literals

# !pip install -q tensorflow-gpu==2.0.0-beta1
# !pip install -q tensorflow-probability
import tensorflow as tf
import tensorflow_probability as tfp
import math
import numpy as np

def convolve_circular(a, b):
    '''
    a: vector
    b: kernel (must have odd length!)
    Naive implementation of circular convolution. The middle entry of b corresponds to the coefficient of z^0:
    b[0] b[1] b[2] b[3] b[4]
    z^-2 z^-1 z^0  z^1  z^2
    '''
    len_a = int(tf.size(a))
    len_b = int(tf.size(b))
    result = np.zeros(len_a)
    for i in range(0, len_a):
        for j in range(0, len_b):
            result[i] += b[-j-1] * a[(i + (j - len_b//2)) % len_a]
    return tf.constant(result, dtype='float32')

# CURRENTLY UNUSED
#
# def convolve_circular(a, b):
#   '''
#   a: vector
#   b: kernel
#   Requires that 2*tf.size(b) <= tf.size(a). If this is not satisfied, overlap
#   will occur in the convolution.
#   '''
#   b_padding = tf.constant([[0, int(tf.size(a) - tf.size(b))]])
#   b_padded = tf.pad(b, b_padding, "CONSTANT")
#   a_fft = tf.signal.fft(tf.complex(a, 0.0))
#   b_fft = tf.signal.fft(tf.complex(b_padded, 0.0))
#   ifft = tf.signal.ifft(a_fft * b_fft)
#   return tf.cast(tf.math.real(ifft), 'float32')

class LazyWavelet(tfp.bijectors.Bijector):
  '''
  This layer corresponds to the downsampling step of a single-step wavelet transform.
  Input to _forward: 1D tensor of even length.
  Output of _forward: Two stacked 1D tensors of the form [[even x componenents], [odd x components]]
  See https://uk.mathworks.com/help/wavelet/ug/lifting-method-for-constructing-wavelets.html
  for notation.
  '''
  def __init__(self,
               validate_args=False,
               name="lazy_wavelet"):
        super().__init__(
        validate_args=validate_args,
        forward_min_event_ndims=1,
        name=name)

  def _forward(self, x):
    x_evens = x[0::2]
    x_odds = x[1::2]
    return tf.stack([x_evens, x_odds])

  def _inverse(self, y):
    x_evens = y[0,:]
    x_odds = y[1,:]
    x = tf.reshape(tf.stack([x_evens, x_odds], axis=1), shape=[-1])  # interleave evens and odds
    return x

  def _inverse_log_det_jacobian(self, y):
    return 0  # QUESTION: Are these log determinants correct?

  def _forward_log_det_jacobian(self, x):
    return 0  # QUESTION: Are these log determinants correct?

class Lifting(tfp.bijectors.Bijector):
  '''
  This layer corresponds to two elementary matrices of the polyphase matrix of a single-step wavelet transform.
  Input to _forward: Two stacked 1D tensors of the form [[lowpass wavelet coefficients], [highpass wavelet coefficients]],
      i.e. the output of LazyWavelet or another Lifting layer.
  Output of _forward: Two stacked 1D tensors of the form [[lowpass wavelet coefficients], [highpass wavelet coefficients]].
  See https://uk.mathworks.com/help/wavelet/ug/lifting-method-for-constructing-wavelets.html
  for notation.
  '''
  def __init__(self,
               validate_args=False,
               name="lifting",
               n_lifting_coeffs=3,
               P_coeff=tf.random.uniform(shape=(3,)),
               U_coeff=tf.random.uniform(shape=(3,))):
    super().__init__(
        validate_args=validate_args,
        forward_min_event_ndims=1,
        name=name)
    self.n_lifting_coeffs = n_lifting_coeffs
    self.P_coeff = tf.Variable(initial_value=P_coeff)  # P: predict (primal lifting)
    self.U_coeff = tf.Variable(initial_value=U_coeff)  # U: update (dual lifting)

  def _forward(self, x):
    x_evens = x[0,:]
    x_odds = x[1,:]
    evens_conv_P = convolve_circular(x_evens, self.P_coeff)
    detail = x_odds - evens_conv_P
    detail_conv_U = convolve_circular(detail, self.U_coeff)
    average = x_evens + detail_conv_U
    return tf.stack([average, detail])

  def _inverse(self, y):
    average = y[0,:]
    detail = y[1,:]
    detail_conv_U = convolve_circular(detail, self.U_coeff)
    x_evens = average - detail_conv_U
    evens_conv_P = convolve_circular(x_evens, self.P_coeff)
    x_odds = evens_conv_P + detail
    x = tf.stack([x_evens, x_odds])
    return x

  def _inverse_log_det_jacobian(self, y):
    return 0  # QUESTION: Are these log determinants correct?

  def _forward_log_det_jacobian(self, x):
    return 0  # QUESTION: Are these log determinants correct?