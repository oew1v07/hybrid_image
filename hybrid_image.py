"""Hybrid Images Algorithm written for COMP6223 at University of Southampton

A hybrid image is an image which can be viewed in two ways as a function of the
distance from it. This module takes two images and creates a hybrid from them
using high and low pass filtering to gain the desired effect."""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from skimage.io import imread, imsave
from scipy import fftpack, misc
import matplotlib.pyplot as plt
from numpy.testing import assert_array_equal
from os.path import split, splitext, join, exists
from os import mkdir
from scipy.ndimage.interpolation import zoom
import sys

def _odd(number):
    """Raises an error if a number isn't odd"""
    if number % 2 == 0:
        raise TypeError("Only odd sizes are supported. "
                        "Got {number}.".format(number = number))

def _dim(number):
    """Raises an error if a number is greater than 2"""
    if number > 2:
        raise TypeError("Only dimensions of 2 or fewer are allowed. "
                        "Got {number}.".format(number = number))

def _check_type(ar, data_type):
    """Raises an error if ar is not of the type given"""
    if not type(ar) == data_type:
        raise TypeError("Only {} types are supported. Got {}.".format(data_type,
                                                                      type(ar)))

def image_to_array(image):
    """Reads in image and turns values into a numpy array

    Parameters
    ----------
    image: string (filepath to image)
        Image to be read in and put into array. It can be of shape (x,y,z)
        where x and y are any size and z is the number of bands (1 or 3)

    Raises
    ------
    OSError: cannot identify image file
        If the file given is not a readable image type

    Returns
    -------
    out: ndarray
        Image as an (M x N) or (M x N x 3) array
    """
    # Uses PIL plugin to read image in
    out = imread(image)
    return out

def convolve(image, kernel):
    """Convolves a kernel over an image using the fast fourier transform.

    Parameters
    ----------
    image: ndarray (arbitrary shape,float or int type)
        Image to be convolved over. It is expected that images will be of
        the shape (x,y,z) where z is the number of bands (1 or 3)
    kernel: ndarray
        Kernel to convolve with (odd shape not necessarily square).

    Raises
    ------
    ValueError
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

    # Check that any dimensions with a "single" column are squeezed for
    # kernels or image
    image = np.squeeze(image)
    kernel = np.squeeze(kernel)

    # Checks dimensionality of image for if we need to do colour
    # convolving
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

    # Creates variable of how many rows with which the image needs to be
    # padded
    pad_shape = np.array(image.shape) + int(np.floor(len(kernel)/2))*2
    pad_shape = list(pad_shape)

    # Calculate FFT for image: using the option to pad with zeros to the
    # the necessary size
    fft_image = fftpack.fftn(image, shape = pad_shape)

    # Calculate FFT for kernel using the option to pad the kernel with
    # zeros to make it the shape of the padded image.
    fft_kernel = fftpack.fftn(kernel, shape = pad_shape)

    # Multiply the fourier tranforms elementwise
    fft_convolved = fft_image * fft_kernel

    # Calculate the inverse fourier of the convolved image, cropping the
    # image to the original image size.
    convolved = np.fft.ifftn(fft_convolved)

    # Calculate how much padding has been added by the fft
    expanded = np.array(convolved.shape) - np.array(image_size)

    # Padding by the fft is always equal on each side so taking half of
    # the difference provides padding on each side.
    pad = int((expanded/2)[0])

    # Index the new array to get the central array without the padding
    out = convolved[pad:-pad, pad:-pad]

    # Take the real part of the convolved array
    out = np.real(out)

    return out

def create_gaussian_kernel(sigma, n = None):
    """Creates Gaussian kernel of size n and standard deviation sigma.
    Mean of the gaussian is automatically set to 0

    Parameters
    ----------
    sigma: int or float
        The standard deviation of the resulting gaussian kernel
    n: int, optional (default: None)
        The size of kernel to be created it will return an n x n array. This is
        for setting the size manually otherwise it will assume that n is a
        function of sigma.

    Raises
    ------
    TypeError
        If the input n is odd.

    Returns
    -------
    kernel: ndarray, of size n x n
        A kernel
    """

    # Standard practice to make the size of the kernel a function of sigma
    # value. It's also possible to set the size of the kernel if necessary but
    # by default it's none, in which case the following code is run.
    if n is None:
        n = int(8*sigma + 1)

        # If n is even then add 1
        if n % 2 == 0:
            n += 1

    # Check that n is of type int
    _check_type(n, int)

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

def low_pass(image, sigma = 0.5, n = None):
    """Creates low pass image

    Uses a Gaussian kernel of size n and standard deviation sigma.

    Parameters
    ----------
    image: ndarray (arbitrary shape,float or int type)
        Image to have a low pass created.
    sigma: int or float, optional (default: 0.5)
        The standard deviation of the gaussian kernel
    n: int, optional (default: None)
        n is the size of gaussian kernel to set the size manually if wanted.

    Raises
    ------
    TypeError
        If the input n is even.

    Returns
    -------
    out: ndarray
        A low pass image
    """
    # Create Gaussian kernel
    kernel = create_gaussian_kernel(sigma, n)

    # Convolve over image using Gaussian kernel
    out = convolve(image, kernel)
    return out

def high_pass(image, sigma = 0.5, n = None):
    """Creates high pass image

    Uses a Gaussian kernel of size n and standard deviation sigma. A low pass
    image is created which we subtract from the original image to get the high
    pass image.

    Parameters
    ----------
    image: ndarray (arbitrary shape,float or int type)
        Image to have a high pass created.
    sigma: int or float, optional (default: 0.5)
        The standard deviation of the gaussian kernel
    n: int, optional (default: None)
        n is the size of gaussian kernel to set the size manually if wanted.

    Raises
    ------
    TypeError
        If the input n is odd.

    Returns
    -------
    out: ndarray
        A high pass image
    """
    # Calls low pass to substract from the image to create high pass
    low = low_pass(image, sigma = sigma, n = n)

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
    """A custom imshow, takes outputs of float type shows them as uint8.

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

def run_hybrid(image1, image2, sigmas = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
               n = None,save_intermediate = True):
    """Creates hybrid image from two image filepaths.

    Parameters
    ----------
    image1: string
        File path to image 1, image to have low pass filter
    image2: string
        File path to image 2, image to have high pass filter
    sigmas: list, optional (default: [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5])
        List of different sigmas to try. Trial and error found that sigma values
        of less than 2.5 did not have enough effect on the images.
    n: int, optional (default: None)
        n is the size of gaussian kernel to set the size manually if wanted.
    save_intermediate: bool, optional (default: True)
        A flag to save all high and low pass images as well as the resulting
        final hybrid image.

    Raises
    ------
    RuntimeError
        If the filepaths do not exist.
    ValueError
        If the input images are not the same shape.

    Returns
    -------
    out: ndarray, same shape as input `image`
        The hybrid image
    """

    # Check that image paths exist
    if exists(image1) == False:
        raise RuntimeError("That file does not exist")
    elif exists(image2) == False:
        raise RuntimeError("That file does not exist")

    # Check that n is None or int
    if n is not None:
        _check_type(n, int)

    # Check save_intermediate is boolean
    _check_type(save_intermediate, bool)

    # Getting names from the filepaths
    x, y = split(image1)
    name1 = splitext(y)[0]

    x, y = split(image2)
    name2 = splitext(y)[0]

    # Read image into array
    image1 = image_to_array(image1)
    image2 = image_to_array(image2)

    # Need to check that images are the same shape
    if image1.shape != image2.shape:
        raise ValueError("Only images of the same shape are supported. "
                        "{shape1} not equal {shape2}.".format(shape1 = image1.shape,
                                                              shape2 = image2.shape))

    # Creating a directory to put all the possible images in
    name = name1 + '_' + name2
    if exists(name) == False:
        mkdir(name)

    # Creating low and high pass images, and hybrid images for each sigma pair
    # so that we can have different sigmas for each image
    for low_sigma in sigmas:
        # Create low pass image
        low = low_pass(image1, sigma = low_sigma, n = n)

        if save_intermediate:
            low_name = name1 + '_low_pass_sigma_' + str(low_sigma*10) + '.png'
            low_name = join(name, low_name)
            # Save low pass image
            misc.imsave(low_name, low)

        for high_sigma in sigmas:
            # Create high pass image
            high = high_pass(image2, sigma = high_sigma, n = n)

            if save_intermediate:
                high_name = name2 + '_high_pass_sigma_' + str(high_sigma*10) + '.png'
                high_name = join(name, high_name)
                # Visualise the high pass better by adding 0.5
                new_high = high.copy()
                new_high = new_high - np.min(new_high)
                # Save high pass image
                misc.imsave(high_name, new_high)

            hybrid_image = hybrid(low, high)

            hybrid_name = name1 + '_' + name2 + '_sigma_' + str(low_sigma*10) + '_' + str(high_sigma*10) + '.png'
            hybrid_name = join(name, hybrid_name)

            misc.imsave(hybrid_name, hybrid_image)

def scale_n_images(image, n = 4, spacing = 10, name = None):
    """Creates a series of scaled versions of image each decreased by a half

    Parameters
    ----------
    image: ndarray or string
        Image to be scaled either as an ndarray or a filepath leading to the File
        to be scaled.
    n: int, optional (default: 4)
        Number of copies of image
    spacing: int, optional (default: 10)
        Number of pixels spacing required between each image
    name: string, optional (default: None)
        Name of file if an exported image is required.

    Raises
    ------
    RuntimeError
        If the filepath does not exist.
    ValueError


    Returns
    -------
    arr: ndarray
        An image of n versions of the original each half the size of the former
        image, with spacing in between.
    """

    if type(image) == str:
        # Check that image paths exist
        if exists(image) == False:
            raise RuntimeError("That file does not exist")
        else:
            image = imread(image)

    # Check that n is int
    _check_type(n, int)

    # Check spacing is int
    _check_type(spacing, int)

    # Make variables for original size to be used later
    orig_height = image.shape[0]
    orig_width = image.shape[1]
    orig_depth = image.shape[2]

    # Work out dimensions for the resulting array
    arr_height = orig_height + 2*spacing

    sizes = []

    for size in range(n):
        sizes.append(np.power(2., -size))

    # These will be used later to calculate positioning of images
    widths = np.round(np.array(sizes)*orig_width)
    heights = np.round(np.array(sizes)*orig_height)

    arr_width = sum(widths) + (n+1)*spacing

    if image.ndim == 3:
        arr = np.zeros((arr_height, arr_width, 3)) + 255
    else:
        arr = np.zeros((arr_height, arr_width)) + 255

    # Initialising the positions for each side.
    if spacing == 0:
        bottom = arr_height
    else:
        bottom = -spacing # this never changes (we want them all to be level)

    right = 0
    left = spacing

    for i in range(n):
        # We set right and top in here to make the equations work well
        right += spacing + widths[i]
        top = spacing + orig_height - heights[i]

        if image.ndim == 3:
            # Zooming into the image
            zoomed = zoom(image, [sizes[i],sizes[i],1])
            canvas = arr[top:bottom, left:right, :]

            if canvas.shape == zoomed.shape:
                arr[top:bottom, left:right, :] = zoomed
            else:
                raise ValueError("The zoomed image is not the same size as the "
                                "array slice. Got {}, expected "
                                "{}.".format(zoomed.shape, canvas.shape))
        else:
            zoomed = zoom(image, [sizes[i],sizes[i]])
            arr[top:bottom, left:right] = zoomed

            if canvas.shape == zoomed.shape:
                arr[top:bottom, left:right, :] = zoomed
            else:
                raise ValueError("The zoomed image is not the same size as the "
                                "array slice. Got {}, expected "
                                "{}.".format(zoomed.shape, canvas.shape))

        left = right + spacing

    if name is not None:
        name = name + '.png'
        misc.imsave(name, arr)

    return arr

if __name__ == '__main__':
    # if the commange line has three arguments then only images have been
    # provided
    if len(sys.argv) == 3:
        pass
    else:
        print("Usage: python hybrid_image.py image1 image2")
        sys.exit()
