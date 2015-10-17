import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import skimage as sk
from scipy import fftpack
import matplotlib.pyplot as plt
from numpy.testing import assert_array_equal

# A module to convolve over an image and carry out a hybrid image
# Seperate the high pass and low pass and then put into the hybrid function
# This creates an easy way of showing each step of the process.

def convolve(image, kernel):
    # Check that the kernal is an odd size
    if kernel.size%2 == 0:
        raise TypeError("Only kernels of odd size are supported. "
                        "Got {shape}.".format(shape = kernel.shape))

    #Creates variable for later use to return original sized image
    image_size = image.shape

    # Creates variable of how many rows with which the image needs to be padded
    pad_shape = np.array(image.shape) + int(np.floor(len(kernel)/2))*2
    pad_shape = list(pad_shape)

    # Calls numpy.pad to create border of zeros half the size of the kernel
    # padded = np.pad(image, pad_width, mode = 'constant')

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

    # Write something to take it back to origin shape using image_size

    return np.real(convolved)


def create_gaussian_kernel(n):
    """Creates Gaussian kernel of size n to do smoothing.

    Parameters
    ----------
    n: string
        The type of kernel to be created. Choices include:
        "gaussian",
        "average",
    """
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
