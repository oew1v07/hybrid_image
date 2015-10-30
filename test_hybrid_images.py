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

    # Checks that the array is equal within 6 decimal places due to small 
    # rounding errors
    assert_array_almost_equal(scipy_conv, conv_arr)

def test_out_size_equal():
    """Tests that the size of the output array is equal to the size of the 
    input array"""

    expected = test_image.shape

    conv_arr = convolve(test_image, kernel)

    assert_equal(actual = conv_arr.shape, desired = expected)

def test_convolve_ndim():
    """Tests whether convolve accepts 3 dimensional arrays and returns an
    array of the same size as was input"""

    test_image_3d = np.array([test_image,test_image])
    test_image_3d = np.swapaxes(test_image_3d,0,2)
    conv_arr_3d = convolve(test_image_3d, kernel)
    assert conv_arr_3d.ndim == test_image_3d.ndim

def test_convolve_ndim():
    """Tests whether convolve accepts 3 dimensional arrays and returns the same
    output twice as it should"""

    test_image_3d = np.array([test_image.copy(),test_image.copy()])
    test_image_3d = np.swapaxes(test_image_3d,0,2)
    conv_arr_3d = convolve(test_image_3d, kernel)
    conv_arr = convolve(test_image, kernel)

    expected = np.array([conv_arr.copy(),conv_arr.copy()])
    expected = np.swapaxes(expected,0,2)

    assert_array_almost_equal(conv_arr_3d,expected)

def test_convolve_4_dims():
    """ Tests whether errors are thrown when given greater than 3 dimensions"""
    test_image_4d = np.zeros((2,2,2,2))
    assert_raises(ValueError, convolve, test_image_4d, kernel)

def test_gaussian():
    """ Tests the my gaussian function is almost equal to the book's gaussian.

    The Gaussian given is taken from Feature Extraction and Image Processing"""
    expected = np.array([[ 0.002,  0.013,  0.220,  0.013,  0.002],
                         [ 0.013,  0.060,  0.098,  0.060,  0.013],
                         [ 0.220,  0.098,  0.162,  0.098,  0.220],
                         [ 0.013,  0.060,  0.098,  0.060,  0.013],
                         [ 0.002,  0.013,  0.220,  0.013,  0.002]])

    actual = create_gaussian_kernel(sigma, n = 5)
    assert_array_almost_equal(actual, expected, decimal = 3)
