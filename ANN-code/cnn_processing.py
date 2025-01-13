""" Module for CNN preprocessing

contains methods for changing the event.image to be correct for use in the CNN models


"""

import cv2
import tensorflow as tf
import random
import numpy as np
from bb_event import Event
import os
import random
import scipy.ndimage as nd
from convert_sim_ims import convert_im, get_dark_sample


def resize_pad_image(image, target_size=(224, 224)):
    """
    Uses tensorflow method to resize the image with padding to target size.

    Parameters
    ----------
    image: numpy.ndarray
        Input image
    target_size: tuple of int
        Target dimensions of image

    Returns
    -------
    resized_image: numpy.ndarray
        Resized image
    """
    resized_image = tf.image.resize_with_pad(image, target_size[0], target_size[1])

    return resized_image


def pad_image_2(image, target_size=(572, 768)):
    """
    Pad an image to a target size by embedding it in a larger frame with random offsets.

    Parameters:
    ----------
    image : np.ndarray
        The input image to be padded.
    target_size : tuple of int, optional
        The target size for the padded image, specified as (height, width). Default is (572, 768).

    Returns:
    -------
    np.ndarray
        The padded image with the specified target size, where the original image is randomly offset within the frame.

    Raises:
    -------
    Exception
        If the image cannot fit inside the target frame.
    """
    small_image = image

    try:
        small_height, small_width = small_image.shape[:2]
        target_height, target_width = target_size

        # Create an empty frame filled with zeros (black) of size (572, 768)
        target_frame = np.zeros((target_height, target_width), dtype=small_image.dtype)

        # Calculate maximum offsets so the small image fits inside the target frame
        max_y_offset = target_height - small_height
        max_x_offset = target_width - small_width

        # Generate random offsets within the allowable range
        y_offset = random.randint(0, max_y_offset)
        x_offset = random.randint(0, max_x_offset)

        # Insert the small image into the target frame at the random offset
        target_frame[
            y_offset : y_offset + small_height, x_offset : x_offset + small_width
        ] = small_image

        return target_frame

    except:
        "Image could not fit inside target frame"


def load_events_bb(file_path):
    """
    Load events from a folder, creating barebones Event objects for each .npy file.

    Parameters:
    ----------
    file_path : str
        Path to the folder containing .npy files.

    Returns:
    -------
    list of Event
        A list of Event objects created from the .npy files in the specified folder.
    """
    events = []

    # Iterate over all .npy files in the directory
    for filename in os.listdir(file_path):
        if filename.endswith(".npy"):
            # Construct full file path
            full_path = os.path.join(file_path, filename)

            # Load image data from .npy file
            image = np.load(full_path)

            # Instantiate Event object
            event = Event(filename, image)

            # Append to events list
            events.append(event)

    return events


def bin_image(image, n):
    """
    Bin an image by reducing its resolution using a block summation method.

    Parameters:
    ----------
    image : np.ndarray
        The input image to be binned.
    n : int
        The binning factor. Each NxN block in the original image is summed to form one pixel in the binned image.

    Returns:
    -------
    np.ndarray
        The binned image with reduced resolution.
    """
    height, width = image.shape

    new_height = (height // n) * n
    new_width = (width // n) * n

    trimmed_image = image[:new_height, :new_width]

    binned_image = trimmed_image.reshape(new_height // n, n, new_width // n, n).sum(
        axis=(1, 3)
    )

    return binned_image


def smooth_operator(image, smoothing_sigma=5):
    """
    Apply Gaussian smoothing to an image.

    Parameters:
    ----------
    image : np.ndarray
        The input image to be smoothed.
    smoothing_sigma : int or float, optional
        The standard deviation (sigma) for the Gaussian kernel. Default is 5.

    Returns:
    -------
    np.ndarray
        The smoothed image.
    """
    image = nd.gaussian_filter(image, sigma=smoothing_sigma)

    return image


def noise_adder(image, m_dark=None, example_dark_list=None):
    """
    Add noise to an image using the master dark frame and an example dark frame.

    Parameters:
    ----------
    image : np.ndarray
        The input image to which noise will be added.
    m_dark : np.ndarray, optional
        The master dark frame used for noise generation. Default is None.
    example_dark_list : list of np.ndarray, optional
        A list of example dark frames for sampling noise. Default is None.

    Returns:
    -------
    np.ndarray
        The image with added noise. If `m_dark` or `example_dark_list` is None, the original image is returned with a warning.
    """
    if m_dark is None or example_dark_list is None:
        print("WARNING: Noise isn't being added.")
        return image

    image = convert_im(
        image,
        get_dark_sample(
            m_dark,
            [len(image[0]), len(image)],
            example_dark_list[np.random.randint(0, len(example_dark_list) - 1)],
        ),
    )
    return image


def pad_image(image, target_size=(572, 768)):
    """
    Pad an image to a target size by embedding it in a larger frame with random offsets.

    Parameters:
    ----------
    image : np.ndarray
        The input image to be padded.
    target_size : tuple of int, optional
        The target size for the padded image, specified as (height, width). Default is (572, 768).

    Returns:
    -------
    np.ndarray
        The padded image with the specified target size, where the original image is randomly offset within the frame.
    """
    small_height, small_width = image.shape[:2]
    target_height, target_width = target_size

    # Create an empty frame filled with zeros (black) of size (572, 768)
    target_frame = np.zeros((target_height, target_width), dtype=image.dtype)

    # Calculate maximum offsets so the small image fits inside the target frame
    max_y_offset = target_height - small_height
    max_x_offset = target_width - small_width

    # Generate random offsets within the allowable range
    y_offset = random.randint(0, max_y_offset)
    x_offset = random.randint(0, max_x_offset)

    # Insert the small image into the target frame at the random offset
    target_frame[
        y_offset : y_offset + small_height, x_offset : x_offset + small_width
    ] = image

    return target_frame


def parse_function(file_path, m_dark, example_dark_list_unbinned, binning=1):
    """
    Parse a file into a preprocessed image tensor and corresponding label.

    Parameters:
    ----------
    file_path : tf.Tensor
        Tensor containing the file path to the .npy file.
    m_dark : np.ndarray
       master dark frame used for preprocessing.
    example_dark_list_unbinned : list of np.ndarray
        List of unbinned example dark frames for noise addition.
    binning : int, optional
        Factor by which the images are binned (default is 1, meaning no binning).

    Returns:
    -------
    image : np.ndarray
        A preprocessed 3D tensor representing the image, with shape ((572, 768), 1).
    label : int
        The label extracted from the file name: 0 for 'C' (carbon), 1 for 'F' (fluorine).
    """
    file_path_str = file_path.numpy().decode("utf-8")

    # Load the image data from the .npy file
    image = np.load(file_path_str)

    # Extract label from file name ('C' or 'F')
    label = (
        0 if "C" in os.path.basename(file_path_str) else 1
    )  # Assume 'C' maps to 0 and 'F' maps to 1

    if binning != 1:
        example_dark_list = [bin_image(i, binning) for i in example_dark_list_unbinned]

    else:
        example_dark_list = example_dark_list_unbinned

    # processing steps
    image = noise_adder(image, m_dark=m_dark, example_dark_list=example_dark_list)
    image = smooth_operator(image)
    image = pad_image(image)

    image = image.astype(np.float32)
    image = np.expand_dims(
        image, axis=-1
    )  # shape becomes ((572, 768), 1) for grayscale

    return image, label


def load_data(base_dirs, batch_size, example_dark_list, m_dark, data_frac=1.0):
    """
    Load and preprocess data from directories into a TensorFlow dataset.

    Parameters:
    ----------
    base_dirs : list of str
        List of base directories containing .npy files.
    batch_size : int
        The batch size for the TensorFlow dataset.
    example_dark_list : list
        A list of example dark frames used for image processing.
    m_dark : list or np.ndarray
        Master dark frame for image.
    data_frac : float, optional
        Fraction of data to use (default is 1.0, meaning all data is used).

    Returns:
    -------
    dataset : tf.data.Dataset
        A TensorFlow dataset containing preprocessed image-label pairs.
    """

    file_list = []
    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            files = [f for f in files if f.endswith(".npy")]
            file_list.extend([os.path.join(root, file) for file in files])

    file_list.sort()

    # perform a reproducible (seeded) shuffle to the directories to ensure randomness
    np.random.seed(77)
    np.random.shuffle(file_list)

    if data_frac < 1.0:
        fraction_to_remove = 1 - data_frac
        num_to_remove = int(len(file_list) * fraction_to_remove)
        indices_to_remove = set(random.sample(range(len(file_list)), num_to_remove))
        filtered_list = [
            x for i, x in enumerate(file_list) if i not in indices_to_remove
        ]

    # create a tensorflow dataset from the list of directories
    dataset = tf.data.Dataset.from_tensor_slices(file_list)

    m_dark_tensor = tf.convert_to_tensor(m_dark, dtype=tf.float32)
    example_dark_tensor = tf.convert_to_tensor(example_dark_list, dtype=tf.float32)
    # apply the parsing function to convert the raw file path, master dark and example dark into data
    dataset = dataset.map(
        lambda file_path: tf.py_function(
            func=parse_function,
            inp=[file_path, m_dark_tensor, example_dark_tensor],
            Tout=(tf.float32, tf.int32),
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # set output shapes to avoid rank issues
    dataset = dataset.map(
        lambda image, label: (
            tf.ensure_shape(image, (572, 768, 1)),
            tf.ensure_shape(label, ()),
        )
    )

    # batch the dataset into chosen size
    dataset = dataset.batch(batch_size)

    return dataset
