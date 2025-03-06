""" Module for CNN preprocessing

contains methods for changing the event.image to be correct for use in the CNN models


"""

import os
import random

import numpy as np
import scipy.ndimage as nd
import tensorflow as tf
from bb_event import BB_Event
from convert_sim_ims import convert_im, get_dark_sample
from tensorflow.keras.applications.vgg16 import preprocess_input  # type: ignore

# import tensorflow_addons as tfa


class PreprocessingLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        smoothing_sigma=3.5,
        m_dark=None,
        example_dark_list=None,
        target_size=(224, 224),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.smoothing_sigma = smoothing_sigma
        self.m_dark = m_dark
        self.example_dark_list = example_dark_list
        self.target_size = target_size

    def call(self, inputs):
        # inputs is a tuple: (images, original_shapes)
        images, original_shapes = (
            inputs  # images: [B, H_pad, W_pad, 3]; original_shapes: [B, 2]
        )

        @tf.function(
            jit_compile=True
        )  # Need this for it to run on the GPU - currently doesn't work though
        def process_single_image(args):
            image, orig_shape = args
            # orig_shape is a tensor with shape [2] (height, width)
            orig_shape = tf.cast(orig_shape, tf.int32)
            h, w = orig_shape[0], orig_shape[1]

            # Crop the image to its original size. (Assuming padding is at the bottom/right.)
            image = image[:h, :w, :]

            # Apply noise addition and smoothing.
            # Use tf.py_function to call the Python functions.
            image = tf.py_function(
                func=lambda x: noise_adder(x, self.m_dark, self.example_dark_list),
                inp=[image],
                Tout=image.dtype,
            )
            image = tf.py_function(
                func=lambda x: smooth_operator(x, self.smoothing_sigma),
                inp=[image],
                Tout=image.dtype,
            )

            # It is a good idea to set the shape if you know it is still [h, w, 3],
            # but note h and w are dynamic.
            image.set_shape([None, None, 3])

            # # Finally, resize to the target size using a built-in TensorFlow op (GPU accelerated).
            # image = tf.image.resize(image, self.target_size)
            # # I'm gonna stick to doing this as its own seperate layer I think. I prefer the control given by the other method
            return image

        # Process each image in the batch individually.
        processed_images = tf.map_fn(
            process_single_image,
            (images, original_shapes),
            fn_output_signature=images.dtype,
        )
        return processed_images

    def get_config(self):
        config = super(PreprocessingLayer, self).get_config()
        config.update(
            {
                "smoothing_sigma": self.smoothing_sigma,
                "m_dark": self.m_dark,
                "example_dark_list": self.example_dark_list,
                "target_size": self.target_size,
            }
        )
        return config


# Not checked either of these classes. They might just be chatgpt gobbledegook.
class SmoothOperator(tf.keras.layers.Layer):
    def __init__(self, smoothing_sigma=3.5, **kwargs):
        """
        A custom layer that applies a Gaussian smoothing filter to the input.
        """
        super(SmoothOperator, self).__init__(**kwargs)
        self.smoothing_sigma = smoothing_sigma

    def call(self, inputs):
        # This inner function will be run in Python.
        def _smooth(x):
            # x is a NumPy array.
            # Apply Gaussian filter from scipy.ndimage:
            return nd.gaussian_filter(x, sigma=self.smoothing_sigma)

        # Wrap the python function in tf.py_function.
        outputs = tf.py_function(func=_smooth, inp=[inputs], Tout=inputs.dtype)
        # Make sure the output tensor has the same shape as the input.
        outputs.set_shape(inputs.shape)
        return outputs


class NoiseAdder(tf.keras.layers.Layer):
    def __init__(self, m_dark=None, example_dark_list=None, **kwargs):
        """
        A custom layer that adds noise to the image.
        If m_dark or example_dark_list is not provided, a warning is printed and the image is returned unchanged.
        """
        super(NoiseAdder, self).__init__(**kwargs)
        self.m_dark = m_dark
        self.example_dark_list = example_dark_list

    def call(self, inputs):
        def _add_noise(x):
            # x is a NumPy array.
            if self.m_dark is None or self.example_dark_list is None:
                print("WARNING: Noise isn't being added.")
                return x
            # Shuffle the example_dark_list using TF and then convert to a NumPy value.
            shuffled = tf.random.shuffle(tf.convert_to_tensor(self.example_dark_list))
            sample_dark = shuffled[0].numpy()
            # Assume that get_dark_sample returns a dark sample given the parameters
            # and that convert_im applies that dark sample to the image.
            # We use x.shape to get dimensions (x.shape: (height, width, channels))
            dark_sample = get_dark_sample(
                self.m_dark, [x.shape[1], x.shape[0]], sample_dark
            )
            return convert_im(x, dark_sample)

        outputs = tf.py_function(func=_add_noise, inp=[inputs], Tout=inputs.dtype)
        outputs.set_shape(inputs.shape)
        return outputs


def resize_pad_image_tf(event, target_size=(224, 224)):
    """
    Rescales the image (with padding) using tensorflow for CNN input
    """

    event.image = tf.image.resize_with_pad(event.image, target_size[0], target_size[1])


def pad_image(event, target_size=(572, 562)):

    small_image = event.image

    try:
        small_height, small_width = small_image.shape[:2]
        target_height, target_width = target_size

        # Create an empty frame filled with zeros (black) of size (572, 562)
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

        event.image = target_frame

    except:
        "Image could not fit inside target frame"


def pad_image_2(image, target_size=(572, 562)):

    small_image = image

    try:
        small_height, small_width = small_image.shape[:2]
        target_height, target_width = target_size

        # Create an empty frame filled with zeros (black) of size (572, 562)
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


def pad_image_3(small_image, target_size=(559, 415)):

    # Convert the input numpy array to a TensorFlow tensor
    small_image = tf.convert_to_tensor(small_image, dtype=tf.float32)

    # Ensure the image has 3 dimensions (add a channel dimension if needed)
    if len(small_image.shape) == 2:
        small_image = tf.expand_dims(small_image, axis=-1)  # Add a channel dimension

    # Get the dimensions of the small image
    small_height, small_width = small_image.shape[:2]
    target_height, target_width = target_size

    # Calculate maximum offsets for random placement
    max_y_offset = target_height - small_height
    max_x_offset = target_width - small_width

    # Generate random offsets within the allowable range
    y_offset = random.randint(0, max_y_offset)
    x_offset = random.randint(0, max_x_offset)

    # Calculate the padding values
    top_padding = y_offset
    bottom_padding = max_y_offset - y_offset
    left_padding = x_offset
    right_padding = max_x_offset - x_offset

    # Use tf.image.pad_to_bounding_box for padding
    padded_image = tf.image.pad_to_bounding_box(
        small_image,
        offset_height=top_padding,
        offset_width=left_padding,
        target_height=target_height,
        target_width=target_width,
    )

    # Update the event image
    return padded_image.numpy()  # Convert back to numpy if needed


def load_events_bb(file_path):
    """
    load events from folder
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
            event = BB_Event(filename, image)

            # Append to events list
            events.append(event)

    return events


def bin_image(image, N: int):

    height, width = image.shape

    new_height = (height // N) * N
    new_width = (width // N) * N

    trimmed_image = image[:new_height, :new_width]

    if N > 1:
        binned_image = trimmed_image.reshape(new_height // N, N, new_width // N, N).sum(
            axis=(1, 3)
        )
        return binned_image
    else:
        return trimmed_image


def smooth_operator(image, smoothing_sigma=3.5):

    image = nd.gaussian_filter(image, sigma=smoothing_sigma)

    return image


def tf_smooth_operator(image, smoothing_sigma=3.5):
    """
    Applies a Gaussian smoothing filter using TensorFlow Addons.
    If the input image is 2D ([H, W]), a channel dimension is added before filtering.
    """
    # If image is 2D, add a channel dimension.
    squeeze_output = False
    if tf.rank(image) == 2:
        image = tf.expand_dims(image, axis=-1)
        squeeze_output = True

    # Compute a kernel size: we use 6*sigma rounded up and ensure it’s odd.
    kernel_size = tf.cast(tf.math.ceil(smoothing_sigma * 6), tf.int32)
    kernel_size = kernel_size + (1 - kernel_size % 2)  # add 1 if even

    # Apply Gaussian filtering.
    try:
        smoothed = tfa.image.gaussian_filter2d(  # type: ignore
            image, filter_shape=[kernel_size, kernel_size], sigma=smoothing_sigma
        )
    except ModuleNotFoundError as e:
        print(f"probably didn't import tfa: {e}")
    if squeeze_output:
        smoothed = tf.squeeze(smoothed, axis=-1)
    return smoothed


def noise_adder(image, m_dark=None, example_dark_list=None):

    if m_dark is None or example_dark_list is None:
        print("WARNING: Noise isn't being added.")
        return image

    image = convert_im(
        image,
        get_dark_sample(
            m_dark,
            [len(image[0]), len(image)],
            # example_dark_list[np.random.randint(0, len(example_dark_list) - 1)],
            tf.random.shuffle(example_dark_list)[
                0
            ],  # Apparently tensorflow didn't like using numpy's randomiser. Not sure how to properly seed this randomisation beacuse it's tf not np. May need to come back here later
        ),
    )
    return image


def pad_image(image, target_size=(572, 562), random=True):

    small_height, small_width = image.shape[:2]
    target_height, target_width = target_size

    # Create an empty frame filled with zeros (black) of size (572, 562)
    target_frame = np.zeros((target_height, target_width), dtype=image.dtype)

    # Calculate maximum offsets so the small image fits inside the target frame
    max_y_offset = target_height - small_height
    max_x_offset = target_width - small_width

    # Generate random offsets within the allowable range
    y_offset = random.randint(0, max_y_offset) if random else 0
    x_offset = random.randint(0, max_x_offset) if random else 0

    # Insert the small image into the target frame at the random offset
    target_frame[
        y_offset : y_offset + small_height, x_offset : x_offset + small_width
    ] = image

    return target_frame


# Function to load a single file and preprocess it
# def parse_function(file_path, binning=1, dark_dir="/vols/lz/MIGDAL/sim_ims/darks"):
def parse_function(
    file_path, m_dark, example_dark_list_unbinned, channels=1, binning=1
):
    if type(file_path) == str:
        file_path_str = file_path
    else:
        try:
            file_path_str = file_path.numpy().decode("utf-8")
        except Exception as e:
            print(f"HANDLED FILE PATH BADLY IN PARSE_FUNCTION {e}")

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

    image = noise_adder(image, m_dark=m_dark, example_dark_list=example_dark_list)
    image = smooth_operator(image)
    image = pad_image(image)
    # print(image.shape)
    # Set shape explicitly for TensorFlow to know
    image = np.float32(image)
    # label = np.int32(label)
    # if channels == 3:
    # # Ensure float type and normalize
    #     image = image.astype(np.float32)
    #     max_val = np.max(image)
    #     if max_val > 0:  # Avoid division by zero
    #         image /= max_val
    #     image = np.stack([image]*channels, axis=-1)
    #     # image = preprocess_input(image)

    # else:
    #     image = np.expand_dims(image, axis=-1)  # Shape becomes (572, 562, 1)

    if channels == 3:
        # Ensure the image is a 2D array
        if image.ndim == 3:
            if image.shape[-1] == 1:  # Handle (height, width, 1)
                image = np.squeeze(image, axis=-1)
            else:
                raise ValueError(
                    f"Unexpected shape {image.shape} for single-channel image."
                )
        elif image.ndim != 2:
            raise ValueError(
                f"Unexpected image shape: {image.shape}. Expected 2D array."
            )

        # Ensure the type is float32
        image = image.astype(np.float32)

        # Normalize to range [0, 255] if needed
        max_val = np.max(image)
        if max_val > 0:  # Avoid division by zero
            image = image / max_val
        else:
            print("POTENTIAL ERROR: max value in image is 0 or less")
        # Create 3 channels by stacking
        image = np.repeat(
            image[:, :, np.newaxis], 3, axis=-1
        )  # Shape becomes (height, width, 3)

        # Apply VGG16 preprocessing
        image = preprocess_input(image)
    else:
        # Expand dimensions for single-channel images
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)  # Shape becomes (height, width, 1)
        elif image.ndim != 3 or image.shape[-1] != 1:
            raise ValueError(f"Unexpected shape {image.shape} for grayscale image.")

    # print(image.shape)
    # print(type(label))

    # label = np.int32(label)
    # image = np.float32(image)
    return image, label


def parse_function_2(
    file_path, m_dark, example_dark_list_unbinned, channels=1, binning=1
):
    # Ensure `file_path` is a string (needed for `np.load`)
    if isinstance(file_path, tf.Tensor):
        file_path = file_path.numpy().decode("utf-8")  # Convert from Tensor to string

    # Load the image
    image = np.load(file_path)

    # Extract label (assumes filenames contain "C" or "F")
    label = 0 if "C" in os.path.basename(file_path) else 1  # 0: Carbon, 1: Fluorine

    # Apply binning if necessary
    example_dark_list = (
        example_dark_list_unbinned
        if binning == 1
        else [bin_image(i, binning) for i in example_dark_list_unbinned]
    )

    # Apply processing functions
    image = noise_adder(image, m_dark=m_dark, example_dark_list=example_dark_list)
    image = smooth_operator(image)
    # image = pad_image(image)

    # Convert to float32
    image = image.astype(np.float32)

    # Ensure correct number of channels
    if channels == 3:
        # Normalize and stack channels
        max_val = np.max(image)
        if max_val > 0:
            image = 255 * image / max_val
        else:
            print("Warning: Image max value is 0, potential issue in normalization.")

        # Create 3-channel image
        image = np.repeat(image[:, :, np.newaxis], 3, axis=-1)

    else:
        # Ensure grayscale format
        image = np.expand_dims(image, axis=-1)

    try:
        image = tf.image.resize_with_pad(image, 224, 224)
    except Exception as e:
        print("###################################################")
        print(f"PROBLEM FILE: {file_path}")
        print("###################################################")

    # Apply VGG16 preprocessing
    image = preprocess_input(image)
    # scale back to [-1, 1]
    image /= np.max(abs(image))

    # Convert to TensorFlow tensors
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    label = tf.convert_to_tensor(label, dtype=tf.int32)

    return image, label


def parse_function_bb(file_path, channels=3, binning=1, max_dims=(572, 562)):
    # Ensure `file_path` is a string (needed for `np.load`)
    if isinstance(file_path, tf.Tensor):
        file_path = file_path.numpy().decode("utf-8")  # Convert from Tensor to string

    # Load the image
    image = np.load(file_path)

    # Extract label (assumes filenames contain "C" or "F")
    label = 0 if "C" in os.path.basename(file_path) else 1  # 0: Carbon, 1: Fluorine

    # Convert to float32
    image = image.astype(np.float32)
    original_size = tf.shape(image)[:2]  # should be [height, width]

    image = pad_image(image, random=False)  # padding in preprocessing

    # print("ORIGINAL SIZE ASADASDNSGNSDFJGSJDFGNSDJFGND = " + str(original_size))
    # Ensure correct number of channels
    if channels == 3:
        # Normalize and stack channels
        max_val = np.max(image)
        if max_val > 0:
            image = image / max_val
        else:
            print("Warning: Image max value is 0, potential issue in normalization.")

        # Create 3-channel image
        image = np.repeat(image[:, :, np.newaxis], 3, axis=-1)

        # Apply VGG16 preprocessing
        image = preprocess_input(image)
    else:
        # Ensure grayscale format
        image = np.expand_dims(image, axis=-1)

    # Convert to TensorFlow tensors
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    label = tf.convert_to_tensor(label, dtype=tf.int32)

    return (image, original_size), label


# Dataset Preparation Function Using `tf.data`
def load_data(base_dirs, batch_size, example_dark_list, m_dark, channels=1):
    # Get all the .npy files from base_dirs
    file_list = []
    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            files = [f for f in files if f.endswith(".npy")]
            file_list.extend([os.path.join(root, file) for file in files])

    file_list.sort()
    np.random.seed(77)
    np.random.shuffle(file_list)

    # Create a TensorFlow dataset from the list of file paths
    dataset = tf.data.Dataset.from_tensor_slices(file_list)

    m_dark_tensor = tf.convert_to_tensor(m_dark, dtype=tf.float32)
    example_dark_tensor = tf.convert_to_tensor(example_dark_list, dtype=tf.float32)
    # Apply the parsing function
    dataset = dataset.map(
        lambda file_path: tf.py_function(
            func=parse_function,
            inp=[file_path, m_dark_tensor, example_dark_tensor, channels],
            Tout=(tf.float32, tf.int32),
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # Set output shapes explicitly to avoid unknown rank issues
    dataset = dataset.map(
        lambda image, label: (
            tf.ensure_shape(image, (572, 562, channels)),
            tf.ensure_shape(label, ()),
        )
    )

    # Shuffle, batch, and prefetch the data for training
    # dataset = dataset.shuffle(buffer_size=100)  # Shuffle the dataset to ensure randomness
    # dataset = dataset.batch(batch_size)
    # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def load_data_yield(base_dirs, example_dark_tensor, m_dark_tensor, channels=1):
    uncropped_error = np.loadtxt(
        "/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/uncropped_error.csv",
        delimiter=",",
        dtype=str,
    )
    min_dim_error = np.loadtxt(
        "/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/min_dim_error.csv",
        delimiter=",",
        dtype=str,
    )
    # Get all the .npy files from base_dirs
    errors = np.concatenate((uncropped_error, min_dim_error))

    # Get all the .npy files from base_dirs
    file_list = []
    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            files = [
                f
                for f in files
                if (f.endswith(".npy") and os.path.join(root, f) not in errors)
            ]
            file_list.extend([os.path.join(root, file) for file in files])

    file_list.sort()
    np.random.seed(77)
    np.random.shuffle(file_list)

    # Process the single image
    for file_path in file_list:
        image, label = parse_function_2(
            file_path, m_dark_tensor.numpy(), example_dark_tensor.numpy(), channels
        )
        yield image, label

    # # Set output shapes explicitly to avoid unknown rank issues
    # dataset = dataset.map(
    #     lambda image, label: (
    #         tf.ensure_shape(image, (572, 562, channels)),
    #         tf.ensure_shape(label, ()),
    #     )
    # )


def yield_preprocessed_data(base_dirs):
    # Get all the .npy files from base_dirs
    file_list = []
    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            files = [f for f in files if (f.endswith(".npy"))]
            file_list.extend([os.path.join(root, file) for file in files])

    file_list.sort()
    np.random.seed(77)
    np.random.shuffle(file_list)

    # Process the single image
    for file_path in file_list:
        image = np.load(file_path)
        label = 0 if "C" in os.path.basename(file_path) else 1
        yield image, label


def get_file_list(base_dirs):
    file_list = []
    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            files = [f for f in files if f.endswith(".npy")]
            file_list.extend([os.path.join(root, file) for file in files])
    file_list.sort()
    return file_list


def load_data_yield_bb(base_dirs, channels=3):
    # Removing bad eggs
    uncropped_error = np.loadtxt(
        "/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/uncropped_error.csv",
        delimiter=",",
        dtype=str,
    )
    min_dim_error = np.loadtxt(
        "/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/min_dim_error.csv",
        delimiter=",",
        dtype=str,
    )
    # Get all the .npy files from base_dirs
    errors = np.concatenate((uncropped_error, min_dim_error))

    file_list = []
    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            files = [f for f in files if (f.endswith(".npy") and f not in errors)]
            file_list.extend([os.path.join(root, file) for file in files])

    file_list.sort()
    np.random.seed(77)
    np.random.shuffle(file_list)

    # Process the single image
    for file_path in file_list:
        image, label = parse_function_bb(file_path, channels)
        yield image, label


# =========CHAT GPT HAIL MARY===============


# def chatgpt_hailmary():

#     import math
#     import tensorflow as tf
#     import numpy as np

#     ###############################################
#     # Section 1: Custom Gaussian Filtering
#     ###############################################

#     def tf_gauss_kernel(size, sigma):
#         """
#         Create a 2D Gaussian kernel of shape [size, size].
#         """
#         # Create a coordinate grid from -(size-1)/2 to (size-1)/2.
#         pos = tf.linspace(- (size - 1) / 2.0, (size - 1) / 2.0, size)
#         xx, yy = tf.meshgrid(pos, pos)
#         kernel = tf.exp(- (xx**2 + yy**2) / (2.0 * sigma**2))
#         kernel = kernel / tf.reduce_sum(kernel)
#         return kernel

#     def tf_gaussian_filter2d(image, sigma):
#         """
#         Applies a 2D Gaussian filter to an image using depthwise convolution.
#         Works for both 2D images (shape [H, W]) and multi-channel images (shape [H, W, C]).
#         """
#         # If image is 2D, add a channel dimension.
#         squeeze_output = False
#         if tf.rank(image) == 2:
#             image = tf.expand_dims(image, axis=-1)
#             squeeze_output = True

#         # Compute kernel size: typically 6*sigma rounded up, then ensure it's odd.
#         kernel_size = tf.cast(tf.math.ceil(sigma * 6), tf.int32)
#         kernel_size = kernel_size + (1 - kernel_size % 2)

#         kernel = tf_gauss_kernel(kernel_size, sigma)  # shape: [kernel_size, kernel_size]
#         # Reshape kernel to [kernel_size, kernel_size, 1, 1] for convolution.
#         kernel = tf.reshape(kernel, [kernel_size, kernel_size, 1, 1])

#         # Get number of channels and tile the kernel.
#         in_channels = tf.shape(image)[-1]
#         kernel = tf.tile(kernel, [1, 1, in_channels, 1])

#         # Expand image to include batch dimension: [1, H, W, C].
#         image = tf.expand_dims(image, axis=0)

#         # Use depthwise_conv2d.
#         filtered = tf.nn.depthwise_conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')

#         # Remove batch dimension.
#         filtered = tf.squeeze(filtered, axis=0)
#         if squeeze_output:
#             filtered = tf.squeeze(filtered, axis=-1)
#         return filtered

#     def tf_smooth_operator(image, smoothing_sigma=3.5):
#         """
#         Smooths an image by applying a Gaussian filter.
#         Expects the input image to be either 2D ([H, W]) or 3D ([H, W, C]).
#         """
#         return tf_gaussian_filter2d(image, smoothing_sigma)

#     ###############################################
#     # Section 2: Noise Addition Functions
#     ###############################################

#     # These functions re-implement your existing convert_im and get_dark_sample logic.

#     def tf_get_dark_sample(m_dark, im_dims, example_dark):
#         """
#         Mimics your get_dark_sample function.
#         m_dark: 2D tensor [H_m, W_m] (master dark).
#         im_dims: [width, height] (list or tensor).
#         example_dark: 2D tensor [H_e, W_e] (example dark).
#         Returns: dark_sample, a 2D tensor of shape [height, width].
#         """
#         m_dark_shape = tf.shape(m_dark)  # [H_m, W_m]
#         H_m = m_dark_shape[0]
#         W_m = m_dark_shape[1]

#         # im_dims is [width, height]
#         if not isinstance(im_dims, tf.Tensor):
#             im_dims = tf.convert_to_tensor(im_dims, dtype=tf.int32)
#         w_im = im_dims[0]
#         h_im = im_dims[1]

#         max_x = W_m - w_im + 1
#         max_y = H_m - h_im + 1
#         start_x = tf.random.uniform([], minval=0, maxval=max_x, dtype=tf.int32)
#         start_y = tf.random.uniform([], minval=0, maxval=max_y, dtype=tf.int32)

#         m_dark_sample = m_dark[start_y : start_y + h_im, start_x : start_x + w_im]
#         example_dark_sample = example_dark[start_y : start_y + h_im, start_x : start_x + w_im]

#         dark_sample = tf.cast(example_dark_sample, tf.float32) - tf.cast(m_dark_sample, tf.float32)
#         return dark_sample

#     def tf_convert_im(im, dark_sample, light_fraction, reflect_fraction=0, gain_corr=10, seed=None):
#         """
#         Mimics your convert_im function.
#         im: 2D float32 tensor.
#         dark_sample: 2D float32 tensor.
#         light_fraction: scalar float.
#         reflect_fraction: scalar float (if nonzero, applies convolution).
#         gain_corr: scalar.
#         seed: 2-element int32 tensor; if None, one is generated.
#         Returns a processed image as a 2D tensor.
#         """
#         im = tf.cast(im, tf.float32)
#         # Multiply im by gain_corr and cast to int32 for binomial sampling.
#         n_trials = tf.cast(gain_corr * im, tf.int32)
#         if seed is None:
#             seed = tf.random.uniform([2], maxval=10000, dtype=tf.int32)
#         binom_samples = tf.random.stateless_binomial(
#             shape=tf.shape(im),
#             seed=seed,
#             counts=n_trials,
#             probs=light_fraction,
#             dtype=tf.int32
#         )
#         im_out = tf.cast(binom_samples, tf.float32) / 0.11

#         if reflect_fraction:
#             # Mimic convolve2d(im, gauss_kernel(10,5), mode='same', boundary='symm')
#             # Here we use our custom Gaussian filter with fixed parameters.
#             kernel = tf_gauss_kernel(10, 5)  # [10, 10]
#             kernel = tf.reshape(kernel, [10, 10, 1, 1])
#             im_out_exp = tf.expand_dims(tf.expand_dims(im_out, axis=0), axis=-1)  # [1, H, W, 1]
#             im_conv = tf.nn.conv2d(im_out_exp, kernel, strides=[1, 1, 1, 1], padding='SAME')
#             im_conv = tf.squeeze(im_conv, axis=[0, -1])
#             im_out = im_out + reflect_fraction * im_conv

#         im_out = tf.round(im_out) + dark_sample
#         return im_out

#     def tf_noise_adder(image, m_dark, example_dark_list, light_fraction, reflect_fraction=0, gain_corr=10):
#         """
#         Re-implements noise_adder in TF.
#         image: 2D tensor [H, W] (float32).
#         m_dark: 2D tensor [H_m, W_m] (master dark, float32).
#         example_dark_list: 3D tensor [N, H_e, W_e] (example dark events, float32).
#         light_fraction: scalar float.
#         reflect_fraction: scalar float.
#         gain_corr: scalar.
#         Returns the processed image as a 2D tensor.
#         """
#         image = tf.cast(image, tf.float32)
#         shape = tf.shape(image)
#         H = shape[0]
#         W = shape[1]

#         num_examples = tf.shape(example_dark_list)[0]
#         rand_index = tf.random.uniform([], minval=0, maxval=num_examples, dtype=tf.int32)
#         selected_example_dark = example_dark_list[rand_index]

#         # im_dims is [width, height]
#         im_dims = [W, H]
#         dark_sample = tf_get_dark_sample(m_dark, im_dims, selected_example_dark)

#         seed = tf.random.uniform([2], maxval=10000, dtype=tf.int32)
#         im_converted = tf_convert_im(image, dark_sample, light_fraction, reflect_fraction, gain_corr, seed)
#         return im_converted

#     ###############################################
#     # Section 3: Preprocessing Layer
#     ###############################################

#     class PreprocessingLayer(tf.keras.layers.Layer):
#         def __init__(self, smoothing_sigma=3.5, m_dark=None, example_dark_list=None,
#                     target_size=(224, 224), light_fraction=0.34 * 0.23 * 0.34,  # using your calc_light_fraction result as an example
#                     reflect_fraction=0, gain_corr=10, **kwargs):
#             """
#             Parameters:
#             smoothing_sigma: standard deviation for Gaussian smoothing.
#             m_dark: master dark image (2D numpy array).
#             example_dark_list: list/array of example dark images (each a 2D numpy array).
#             target_size: desired final [height, width].
#             light_fraction: scalar float.
#             reflect_fraction: scalar float.
#             gain_corr: scalar.
#             """
#             super().__init__(**kwargs)
#             self.smoothing_sigma = smoothing_sigma
#             self.target_size = target_size
#             self.light_fraction = light_fraction
#             self.reflect_fraction = reflect_fraction
#             self.gain_corr = gain_corr

#             if m_dark is not None:
#                 self.m_dark = tf.convert_to_tensor(m_dark, dtype=tf.float32)
#             else:
#                 self.m_dark = None
#             if example_dark_list is not None:
#                 # Expect example_dark_list as a 3D array: [N, H, W].
#                 self.example_dark_list = tf.convert_to_tensor(example_dark_list, dtype=tf.float32)
#             else:
#                 self.example_dark_list = None

#         def call(self, inputs):
#             """
#             inputs: tuple (images, original_shapes)
#             images: tensor of shape [B, H_pad, W_pad] or [B, H_pad, W_pad, C].
#             original_shapes: tensor of shape [B, 2] containing [width, height] for each image.
#             """
#             images, original_shapes = inputs

#             def process_single_image(args):
#                 image, orig_shape = args
#                 orig_shape = tf.cast(orig_shape, tf.int32)
#                 # Expect orig_shape as [width, height]
#                 w, h = orig_shape[0], orig_shape[1]
#                 # Crop to original size.
#                 image = image[:h, :w]
#                 # If image has an extra channel dimension, squeeze it (we expect a 2D image for noise addition).
#                 if tf.rank(image) > 2:
#                     image = tf.squeeze(image)
#                 # Apply noise addition if dark images are provided.
#                 if (self.m_dark is not None) and (self.example_dark_list is not None):
#                     image = tf_noise_adder(image, self.m_dark, self.example_dark_list,
#                                         self.light_fraction, self.reflect_fraction, self.gain_corr)
#                 # Apply smoothing.
#                 image = tf_smooth_operator(image, self.smoothing_sigma)
#                 # If needed, expand dims for resizing.
#                 if tf.rank(image) == 2:
#                     image = tf.expand_dims(image, axis=-1)
#                 image = tf.image.resize(image, self.target_size)
#                 return image

#             processed_images = tf.map_fn(
#                 process_single_image,
#                 (images, original_shapes),
#                 fn_output_signature=tf.float32
#             )
#             return processed_images

#         def get_config(self):
#             config = super().get_config()
#             config.update({
#                 "smoothing_sigma": self.smoothing_sigma,
#                 "target_size": self.target_size,
#                 "light_fraction": self.light_fraction,
#                 "reflect_fraction": self.reflect_fraction,
#                 "gain_corr": self.gain_corr,
#             })
#             return config
# # it didn't work. wrapped it in a function to hide it
