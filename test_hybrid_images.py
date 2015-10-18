import numpy as np
from numpy.testing import (assert_array_equal, assert_equal, assert_raises,
                           assert_array_almost_equal)
from hybrid_image import convolve
import scipy.ndimage.filters as filt


test_image = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                       [ 0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
                       [ 0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
                       [ 0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
                       [ 0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
                       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

even_kernel = np.array([[0, 0, 0, 0],
                        [1, 1, 1, 0],
                        [1, 1, 1, 1]], bool)

kernel = np.array([[ 0.1111,  0.1111,  0.1111],
                   [ 0.1111,  0.1111,  0.1111],
                   [ 0.1111,  0.1111,  0.1111]])

def test_convolve_even_kernel_error():
    """ Tests whether errors are thrown when given an even kernel"""

    assert_raises(TypeError, convolve, test_image, even_kernel)

def test_convolve_vs_scipy():
    """Tests the output of my convolve v.s. scipy.ndimage.filters.convolve"""

    scipy_conv = filt.convolve(test_image, kernel, mode = 'constant')

    conv_arr = convolve(test_image, kernel)

    # Checks that the array is equal within 6 decimal places due to small rounding errors
    assert_array_almost_equal(scipy_conv, conv_arr)

def test_out_size_equal():
    """Tests that the size of the output array is equal to the size of the input array"""

    expected = test_image.shape

    conv_arr = convolve(test_image, kernel)

    assert_equal(actual = conv_arr.shape, desired = expected)



# if __name__ == "__main__":
#     np.testing.run_module_suite()
