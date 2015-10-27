import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import skimage as sk
from scipy import fftpack
import matplotlib.pyplot as plt
from numpy.testing import assert_array_equal

# A module to convolve over an image and carry out a hybrid image
# Seperate the high pass and low pass and then put into the hybrid function
# This creates an easy way of showing each step of the process.
# The term cut-off frequency is to do with the sigma of the gaussian kernel.


def _odd(number):
    """Raises an error if a number is odd"""
    if number % 2 == 0:
        raise TypeError("Only odd sizes are supported. "
                        "Got {number}.".format(number = number))

def _dim(number):
    """Raises an error if a number is greater than 2"""
    if number > 2:
        raise TypeError("Only dimensions of 2 or fewer are allowed. "
                        "Got {number}.".format(number = number))

def _check_type_supported(n): 
    if not (np.issubdtype(type(n), np.integer)):
        raise TypeError("Only integer types are supported. "
                        "Got %s." % type(n))

def image_to_array(image):
    """Reads in image and turns values into a numpy array

    Parameters
    ----------
    image: string (filepath to image)
        Image to be read in and put into array. It can be of shape (x,y,z) 
        where x and y are any size and z is the number of bands (1 or 3)
    """
    pass

def convolve(image, kernel):
    """Convolves a kernel over an image using the fast fourier transform.

    Parameters
    ----------
    image: ndarray (arbitrary shape,float or int type)
        Image to be convolved over. It is expected that images will be of the
        shape (x,y,z) where z is the number of bands (1 or 3)
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

    # Check if size is odd then both rows and columns must be odd
    _odd(kernel.size)

    # Check dimensions for kernel are two or fewer
    _dim(kernel.ndim)

    # Check that any dimensions with a "single" column are squeezed for kernels
    # or image
    image = np.squeeze(image)
    kernel = np.squeeze(kernel)

    # Checks dimensionality of image for if we need to do colour convolving
    if image.ndim > 2:
        if image.ndim > 3:
            raise ValueError("Image should have no more than 3 bands "
                             "Got {number}.".format(number = image.ndim))
        else:
            # Create output array for multiple 
            output = np.zeros(image.shape)

            for i in range(image.shape[2]): # Gets the number of 3d bands
                image_it = image[:,:,i]
                output[:, :, i] = convolve(image_it,kernel)

            return output

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

    # Do I need to remove padding at end - ask Jonathan Hare
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


def create_gaussian_kernel(n, sigma):
    """Creates Gaussian kernel of size n and standard deviation sigma.
    Mean of the gaussian is automatically set to 0

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

    # Define array kernel
    kernel = np.zeros((n, n))

    # Find centre of window
    centre = np.floor(n/2) + 1

    # To normalise by the total sum
    total_sum = 0

    # Calculating the gaussian for each entry of the template
    for i in range(1, n+1):
        for j in range(1, n+1):
            superscript = -((j - centre)**2 + (i - centre)**2)/(2*sigma**2)
            kernel[j-1, i-1] = np.exp(superscript)
            total_sum += np.exp(superscript)

    kernel = kernel/total_sum

    return kernel


def low_pass(image, n = 3, sigma = 0.5):
    """Creates low pass image

    Uses a Gaussian kernel of size n and standard deviation sigma.

    Parameters
    ----------
    image: ndarray (arbitrary shape,float or int type)
        Image to have a low pass created.
    n: int, optional (default: 3)
        The size of kernel to be created, an n x n array will be used.
    sigma: int or float, optional (default: 0.5)
        The standard deviation of the gaussian kernel

    Raises
    ------
    TypeError
        If the input n is odd.

    Returns
    -------
    out: ndarray
        A low pass image 
    """
    # Create Gaussian kernel
    kernel = create_gaussian_kernel(n, sigma)

    # Convolve over image using Gaussian kernel
    out = convolve(image, kernel)
    return out


def high_pass(image, n = 3, sigma = 0.5):
    """Creates high pass image

    Uses a Gaussian kernel of size n and standard deviation sigma. A low pass
    image is created which we subtract from the original image to get the high
    pass image.

    Parameters
    ----------
    image: ndarray (arbitrary shape,float or int type)
        Image to have a high pass created.
    n: int, optional (default: 3)
        The size of kernel to be created, an n x n array will be used.
    sigma: int or float, optional (default: 0.5)
        The standard deviation of the gaussian kernel

    Raises
    ------
    TypeError
        If the input n is odd.

    Returns
    -------
    out: ndarray
        A low pass image 
    """
    # Calls low pass to substract from the image to create high pass
    low = low_pass(image, n = 3, sigma = 0.5)

    out = image - low

    return out


def hybrid(image_1, image_2):
    """Creates a hybrid image from two different images

    Parameters
    ----------
    image_1: ndarray

    image_2: ndarray

    Returns
    -------
    out: ndarray
        Hybrid image"""

    out = 0.5*image_1 + 0.5*image_2
    
    return out

def show_image(image):
    """A custom imshow, for outputs of float type need to be shown as uint8.

    Parameters
    ----------
    image_1: ndarray
        Image to be viewed
    
    Notes
    -----
        Outputs created from low_pass and high pass are of type float64.
        However imshow only allows type of float for values between 0 and 1 or 
        type uint8. To view the different images the type of the output is 
        changed for viewing. If an image is only 2 dimensions then it also 
        makes the arguments for viewing sensible with cmap = 'gray' and 
        interpolation = 'nearest'"""
    
    #Check whether image is 2 or 3 dimensions
    if image.ndim == 2:
        cmap = 'gray'
        inter = 'nearest'
        plt.imshow(image.astype(np.uint8),cmap = cmap, interpolation = inter)
    else:
        plt.imshow(image.astype(np.uint8))
