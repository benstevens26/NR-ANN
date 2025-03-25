"""
Module that contains standalone functions for image preprocessing.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from convert_sim_ims import convert_im, get_dark_sample


def gaussian_smoothing(image, smoothing_sigma=3.5):
    """
    Apply Gaussian smoothing to an event image.

    Parameters
    ----------
    image : np.ndarray
        The image to smooth.
    smoothing_sigma : float, optional
        The standard deviation for Gaussian kernel, controlling the smoothing level (default is 3.5).

    Returns
    -------
    np.ndarray
        The smoothed image.

    """
    return gaussian_filter(image, sigma=smoothing_sigma)


def noise_adder(image, m_dark, example_dark_list, noise_index=None):
    """
    Add noise to an event image based on a master dark image and random sample from example darks.

    Parameters
    ----------
    image : np.ndarray
        The image to which noise will be added.
    m_dark : np.ndarray
        2D array (image) containing the master dark.
    example_dark_list : list of np.ndarray
        List of example dark images from which a random sample is selected for noise addition.
    noise_index : int, optional
        The index of the example dark to use for noise addition (if you want to specify it).


    Returns
    -------
    np.ndarray
        The image with noise added.

    """
    if noise_index is None:
        noise_index = np.random.randint(0, len(example_dark_list) - 1)

    im_dims = [len(image[0]), len(image)]

    noised_image = convert_im(
        image,
        get_dark_sample(
            m_dark,
            im_dims,
            example_dark_list[noise_index],
        ),
    )
    return noised_image