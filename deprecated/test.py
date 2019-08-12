# See https://github.com/tensorflow/tensorflow/issues/31249
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import tensorflow_probability as tfp

from main import LazyWavelet, Lifting

class TestLazyWaveletLayer(tf.test.TestCase):
    def test_forward(self):
        x = tf.constant([1,2,3,4,5,6,7,8], dtype='float32')
        y_expected = tf.constant([[1,3,5,7], [2,4,6,8]], dtype='float32')
        lazy_layer = LazyWavelet()
        y_result = lazy_layer._forward(x)
        self.assertAllEqual(y_expected, y_result)

    def test_inverse(self):
        lazy_layer = LazyWavelet()
        x = tf.constant([1,2,3,4,5,6,7,8], dtype='float32')
        y = lazy_layer._forward(x)
        y_inv = lazy_layer._inverse(y)
        self.assertAllClose(x, y_inv)

class TestLiftingLayer(tf.test.TestCase):
    def test_forward(self):
        def test_x_transforms_to_y_expected(x, P_coeff, U_coeff, y_expected):
            lifting = Lifting(P_coeff=P_coeff, U_coeff=U_coeff)
            y_result = lifting.forward(x)
            self.assertAllEqual(y_expected, y_result)

        # Haar wavelet, verified with MATLAB:
        # y_expected = dwt(x, [1/2, 1/2], [1, -1]))
        test_x_transforms_to_y_expected(x=tf.constant([[1,3,5,7], [2,4,6,8]], dtype='float32'),
                        P_coeff=tf.constant([1.]),
                        U_coeff=tf.constant([.5]),
                        y_expected=tf.constant([[1.5, 3.5, 5.5, 7.5], [1.,  1.,  1.,  1.]]))

        # Wavelet with lowpass filter h(z) = (1/8) * (2z^3 - z^2 + 2z + 6 - z^-2) and highpass
        # filter g(z) = (-1/2)z^-2 - 1/2 + z.
        # The output can be verified by doing circular convolution manually. MATLAB's dwt is not
        # usable for verification, since it seems to only take causal filters as arguments.
        test_x_transforms_to_y_expected(x=tf.constant([[1,3,5,7], [2,4,6,8]], dtype='float32'),
                        P_coeff=tf.constant([0, .5, .5]),
                        U_coeff=tf.constant([.25, .25, 0]),
                        y_expected=tf.constant([[1., 4., 6., 7.], [-2., 2., 2., 2.]]))

    def test_inverse(self):
        lifting = Lifting()
        x = tf.constant([[1,3,5,7], [2,4,6,8]], dtype='float32')
        y = lifting._forward(x)
        y_inv = lifting._inverse(y)
        self.assertAllClose(x, y_inv)

class TestChaining(tf.test.TestCase):
    def test_chain_of_lifting_and_lazy_wavelet_forward(self):
        x = tf.constant([1,2,3,4,5,6,7,8], dtype='float32')
        y_expected = tf.constant([[1., 4., 6., 7.], [-2., 2., 2., 2.]])
        chain = tfp.bijectors.Chain([Lifting(P_coeff=tf.constant([0, .5, .5]),U_coeff=tf.constant([.25, .25, 0])), LazyWavelet()])
        y_result = chain.forward(x)
        self.assertAllEqual(y_expected, y_result)

    def test_chain_of_lifting_and_lazy_wavelet_inverse(self):
        x = tf.constant([1,2,3,4,5,6,7,8], dtype='float32')
        chain = tfp.bijectors.Chain([Lifting(P_coeff=tf.constant([0, .5, .5]),U_coeff=tf.constant([.25, .25, 0])), LazyWavelet()])
        y = chain.forward(x)
        y_inv = chain._inverse(y)
        self.assertAllClose(x, y_inv)

if __name__ == '__main__':
    unittest.main()