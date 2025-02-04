"""
Module that contains standalone functions for feature extraction from Event images.
"""

import numpy as np
from scipy.ndimage import center_of_mass
from scipy.linalg import svd
from scipy.interpolate import griddata
from scipy.interpolate import splprep, splev


def extract_energy_deposition(image):
    """
    Extract the total energy deposition from image

    Parameters
    ----------
    image : np.ndarray
        The image to extract the energy deposition from.

    Returns
    -------
    float
        The total energy deposition in the image.
    """
    return np.sum(image)


def extract_max_pixel_intensity(image):
    """
    Extract the maximum pixel intensity from image

    Parameters
    ----------
    image : np.ndarray
        The image to extract the maximum pixel intensity from.

    Returns
    -------
    float
        The maximum pixel intensity in the image.
    """
    return np.max(image)


def extract_axis(image, method="eigen"):
    """
    Extracts the principal axis of a recoil image.

    Parameters:
        image (numpy.ndarray): 2D array representing the image.
        method (str): Method to use for principal axis extraction. Options are 'eigen' (default) and 'svd'.

    Returns:
        principal_axis (numpy.ndarray): Unit vector indicating the direction of the principal axis.
        centroid (tuple): The coordinates of the image centroid.
    """

    if method == "eigen":

        y_coords, x_coords = np.nonzero(image)
        intensities = image[y_coords, x_coords]

        total_intensity = intensities.sum()
        mean_x = (x_coords * intensities).sum() / total_intensity
        mean_y = (y_coords * intensities).sum() / total_intensity

        centered_x = x_coords - mean_x
        centered_y = y_coords - mean_y

        weighted_coords = np.stack(
            [centered_x * intensities, centered_y * intensities], axis=0
        )
        cov_matrix = np.cov(weighted_coords)

        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

        if principal_axis[0] < 0:
            principal_axis = -principal_axis

        return principal_axis, (mean_x, mean_y)

    elif method == "svd":

        height, width = image.shape

        # Create a grid of coordinates for each pixel
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

        # Flatten the arrays to get a list of (x, y) points
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        intensities_flat = image.flatten()

        # Filter out zero intensities
        non_zero = intensities_flat > 0
        x_flat = x_flat[non_zero]
        y_flat = y_flat[non_zero]

        # Create a matrix where rows represent pixel positions, and columns are the coordinates weighted by intensity
        data_matrix = np.vstack((x_flat, y_flat)).T

        # Perform SVD on the data matrix
        U, S, Vt = svd(data_matrix - np.mean(data_matrix, axis=0), full_matrices=False)

        # The principal axis is given by the first row of Vt (right singular vectors)
        principal_axis = Vt[0]

        if principal_axis[0] < 0:
            principal_axis[0] = -principal_axis[0]

        mean_x = np.mean(x_flat)
        mean_y = np.mean(y_flat)

        return principal_axis, (mean_x, mean_y)


def extract_intensity_contour(image, resolution=500):
    """
    Performs grid interpolation on a nuclear recoil image to produce intensity contours.

    Parameters:
        image (numpy.ndarray): 2D array representing the image.
        resolution (int): Number of interpolation points along each axis.

    Returns:
        tuple: grid_x, grid_y, grid_z - Interpolated grid coordinates and intensity values.
    """

    # Extract non-zero intensity points
    y_coords, x_coords = np.nonzero(image)
    intensities = image[y_coords, x_coords]

    if len(x_coords) == 0 or len(y_coords) == 0:
        raise ValueError("The input image contains no non-zero intensities.")

    # Define a high-resolution grid
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, image.shape[1] - 1, resolution),
        np.linspace(0, image.shape[0] - 1, resolution),
    )

    # Perform cubic interpolation
    grid_z = griddata(
        points=(x_coords, y_coords),
        values=intensities,
        xi=(grid_x, grid_y),
        method="cubic",
        fill_value=0,
    )

    return grid_x, grid_y, grid_z


def extract_spline(image, smoothing=0.5, resolution=500):
    """
    Extracts the recoil principal axis as a 1D cubic spline.

    Parameters:
        image (numpy.ndarray): 2D array representing the image.
        smoothing (float): Smoothing factor for the spline fitting. Default is 0.5.
        resolution (int): Number of points to interpolate along the spline.

    Returns:
        tuple: x_spline, y_spline - Coordinates of the spline points.
    """
    # Extract non-zero intensity points
    y_coords, x_coords = np.nonzero(image)
    intensities = image[y_coords, x_coords]

    if len(x_coords) == 0 or len(y_coords) == 0:
        raise ValueError("The input image contains no non-zero intensities.")

    # Weight points by intensity
    weights = intensities / intensities.max()

    # Debugging: Ensure inputs are valid
    if len(x_coords) < 4 or len(y_coords) < 4:
        raise ValueError(
            "Insufficient points for spline fitting. At least 4 points are required."
        )
    if len(x_coords) != len(y_coords) or len(x_coords) != len(weights):
        raise ValueError(
            f"Input dimensions mismatch: x_coords({len(x_coords)}), y_coords({len(y_coords)}), weights({len(weights)})"
        )

    # Fit a spline through the weighted points
    tck, _ = splprep([x_coords, y_coords], w=weights, s=smoothing)

    # Interpolate the spline
    u_fine = np.linspace(0, 1, resolution)
    x_spline, y_spline = splev(u_fine, tck)

    return x_spline, y_spline


def extract_bounding_box(image: np.ndarray) -> tuple:
    """
    Extract the bounding box coordinates and dimensions from an image using the 50th percentile threshold.

    Parameters:
    image (np.ndarray): The input image as a 2D numpy array.

    Returns:
    tuple: (min_y, min_x, max_y, max_x) representing the bounding box.
    """
    # Determine the 50th percentile intensity threshold
    threshold = np.percentile(image, 50)

    # Find the indices of pixels above the threshold
    nonzero_indices = np.argwhere(image > threshold)
    if nonzero_indices.size == 0:
        return None  # No pixels above the threshold, so no bounding box

    # Calculate the bounding box
    min_y, min_x = nonzero_indices.min(axis=0)
    max_y, max_x = nonzero_indices.max(axis=0)

    return min_y, min_x, max_y, max_x
