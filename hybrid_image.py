import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import skimage as sk
from scipy import fftpack
import matplotlib.pyplot as plt
from numpy.testing import assert_array_equal

# A module to convolve over an image and carry out a hybrid image
# Seperate the high pass and low pass and then put into the hybrid function
# This creates an easy way of showing each step of the process.

def _odd(number):
    """Raises an error if a number is odd"""
    if number % 2 == 0:
        raise TypeError("Only odd sizes are supported. "
                        "Got {number}.".format(number = number))

def _check_type_supported(n): 
    if not (np.issubdtype(type(n), np.integer)):
        raise TypeError("Only integer types are supported. "
                        "Got %s." % type(n))

def convolve(image, kernel):
    """Convolves a kernel over an image using the fast fourier transform.

    Parameters
    ----------
    image: ndarray (arbitrary shape,float or int type)
        Image to be convolved over.
    kernel: ndarray
        Kernel to convolve with of odd shape not necessarily square.

    Raises
    ------
    TypeError
        If the input kernel is of even shape in one or both dimensions.

    Returns
    -------
    out: ndarray, same shape as input `image`
        The convolved image
    """

    # Check kernel is odd
    _odd(kernel.size)

    # Creates variable for later use to return original sized image
    image_size = image.shape

    # Creates variable of how many rows with which the image needs to be padded
    pad_shape = np.array(image.shape) + int(np.floor(len(kernel)/2))*2
    pad_shape = list(pad_shape)

    # Calculate FFT for image: using the option to pad with zeros to the 
    # the necessary size 
    fft_image = fftpack.fftn(image, shape = pad_shape)

    # Calculate FFT for kernel using the option to pad the kernel with zeros to
    # make it the shape of the padded image.
    fft_kernel = fftpack.fftn(kernel, shape = pad_shape)

    # Multiply the fourier tranforms by element
    fft_convolved = fft_image * fft_kernel

    # Calculate the inverse fourier of the convolved image, cropping the image
    # to the original image size.
    convolved = np.fft.ifftn(fft_convolved)

    # Calculate how much padding has been added by the fft
    expanded = np.array(convolved.shape) - np.array(image_size)

    # Padding by the fft is always equal on each side so taking half of the 
    # difference provides padding on each side.
    pad = int((expanded/2)[0])

    # Index the new array to get the central array without the padding
    out = convolved[pad:-pad, pad:-pad]

    # Take the real part of the convolved array
    out = np.real(out)

    return out


def create_gaussian_kernel(n):
    """Creates Gaussian kernel of size n and standard deviation sigma.

    Parameters
    ----------
    n: int
        The size of kernel to be created it will return an n x n array.
    sigma: int or float
        The standard deviation of the resulting gaussian kernel

    Raises
    ------
    TypeError
        If the input n is odd.

    Returns
    -------
    kernel: ndarray, of size n x n
        A kernel 
    """

    # Check that n is of type int
    _check_type_supported(n)

    # Check that n is an odd size
    _odd(n)

    # Find centre of window
    centre = floor(n/2) + 1

    # Normalise by the total sum
    total_sum = 0

    #

    return kernel


def low_pass():
    pass


def high_pass():
    # Calls low pass to substract from the image to create high pass
    pass


def hybrid(image_1, image_2):
    """Creates a hybrid image from two different images

    Parameters
    ----------
    image_1: numpy.array

    image_2: numpy.array"""
    pass


def check_kernel_odd(kernel):
    """Checks a kernel is odd in both dimensions."""
    pass
