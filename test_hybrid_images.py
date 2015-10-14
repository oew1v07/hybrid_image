import numpy as np
from numpy.testing import (assert_array_equal, assert_equal, assert_raises,
                           assert_warns)
from skimage.morphology import remove_small_objects, remove_small_holes
from ..._shared._warnings import expected_warnings

# Create test image here!
test_image = 

even_kernel = np.array([[0, 0, 0, 0],
                        [1, 1, 1, 0],
                        [1, 1, 1, 1]], bool)

def test_convolve_even_kernel_error():
    assert_raises(TypeError,convolve,even_kernel,test_image)

# if __name__ == "__main__":
#     np.testing.run_module_suite()
