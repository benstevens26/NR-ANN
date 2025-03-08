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


def uncropped_check(
    image: np.ndarray, search_fraction: float, method: str = "max_comparison"
) -> bool:
    """
    Check if an image is uncropped using different methods based on a search fraction.

    Parameters:
    image (np.ndarray): The input image as a 2D numpy array.
    search_fraction (float): Depends on the method used. For 'max_comparison', it is the fraction of the image dimensions.
    method (str): The method to use for the check. Options are 'max_comparison' or 'area_comparison'.

    Returns:
    bool: True if the image is uncropped based on the selected method, otherwise False.
    """
    # Get the dimensions of the image
    max_y, max_x = image.shape
    from feature_extraction import extract_bounding_box

    if method == "max_comparison":
        # Determine a threshold intensity to avoid floating point errors
        non_zero_values = image[image > 0]
        if non_zero_values.size == 0:
            return False  # No non-zero pixels, so cannot determine uncropped status

        threshold = np.percentile(
            non_zero_values, 1
        )  # Smallest 1% of non-zero intensities

        # Find the largest x and y coordinates with intensity greater than the threshold
        nonzero_indices = np.argwhere(image > threshold)
        if nonzero_indices.size == 0:
            return False  # No valid indices after applying threshold

        max_nonzero_y = nonzero_indices[:, 0].max() + 1  # Add 1 to account for indexing
        max_nonzero_x = nonzero_indices[:, 1].max() + 1

        # Compare max_nonzero_x and max_nonzero_y with thresholds
        uncropped_x = max_nonzero_x < search_fraction * max_x
        uncropped_y = max_nonzero_y < search_fraction * max_y

        return uncropped_x or uncropped_y

    elif method == "area_comparison":
        # extract the bounding box
        bounding_box = extract_bounding_box(image)

        # calculate the area of the bounding box
        box_area = (bounding_box[2] - bounding_box[0]) * (
            bounding_box[3] - bounding_box[1]
        )

        # calculate the area of the image
        image_area = max_y * max_x

        # return True if the box area is less than the search fraction of the image area
        return box_area / image_area < search_fraction

    else:
        raise ValueError(
            "Invalid method. Choose 'max_comparison' or 'area_comparison'."
        )


def dim_check(image: np.ndarray, min_dim: int) -> bool:
    """
    Check if an image has dimensions larger than a minimum value.

    Parameters:
    image (np.ndarray): The input image as a 2D numpy array.
    min_dim (int): Minimum dimension in either x or y direction.

    Returns:
    bool: True if the either x or y dimension is less than the minimum value, otherwise False.
    """
    y_dim, x_dim = image.shape
    return y_dim < min_dim or x_dim < min_dim


def zero_edges(image: np.ndarray, edge_width: int = 1) -> np.ndarray:
    """
    Zero the intensities in the specified number of pixels from each edge of the image.

    Parameters:
    image (np.ndarray): The input image as a 2D numpy array.
    edge_width (int): The width of the edges to zero out.

    Returns:
    np.ndarray: The image with edges zeroed.
    """
    # Copy the image to avoid modifying the original
    image_zeroed = image.copy()

    # Zero out the edges
    image_zeroed[:edge_width, :] = 0  # Top edge
    image_zeroed[-edge_width:, :] = 0  # Bottom edge
    image_zeroed[:, :edge_width] = 0  # Left edge
    image_zeroed[:, -edge_width:] = 0  # Right edge

    return image_zeroed
