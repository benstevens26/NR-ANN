import csv
import os

import scipy.ndimage as nd
from convert_sim_ims import *
from event import Event
from tqdm import tqdm
import numpy as np


def bin_event(event, N, parse_image=False):
    """A function to bin an event's image.
    IMPORTANT: The noise has to be binned as well which is NOT handled by this function.

    Args:
        event (object): The instance of the Event class to be binned
        N (int): The degree of binning. E.g. if you want to bin the image into 2x2 pixels, pass N=2.
        parse_image (bool): adds hacky functionality to bin an image that's not attached to an event. This is useful for darks. To use it, just parse the image where you would normally put the event.

    Returns:
        object: the event object with the image updated to the binned version.
    """
    if parse_image:
        image = event
    else:
        image = event.image
    height, width = image.shape

    new_height = (height // N) * N
    new_width = (width // N) * N

    trimmed_image = image[:new_height, :new_width]

    binned_image = trimmed_image.reshape(new_height // N, N, new_width // N, N).sum(
        axis=(1, 3)
    )

    if parse_image:
        return binned_image
    else:
        event.image = binned_image
        return event


def smooth_operator(event, smoothing_sigma=5):
    """
    Apply Gaussian smoothing to an event image.

    Parameters
    ----------
    event : Event
        The Event object containing the image to smooth.
    smoothing_sigma : float, optional
        The standard deviation for Gaussian kernel, controlling the smoothing level (default is 5).

    Returns
    -------
    Event
        The Event object with the smoothed image.
    """
    event.image = nd.gaussian_filter(event.image, sigma=smoothing_sigma)

    return event


def noise_adder(event, m_dark=None, example_dark_list=None, noise_index=None):
    """
    Add noise to an event image based on a master dark image and random sample from example darks.

    Parameters
    ----------
    event : Event
        The Event object with an image to which noise will be added.
    m_dark : np.ndarray
        2D array (image) containing the master dark.
    example_dark_list : list of np.ndarray
        List of example dark images from which a random sample is selected for noise addition.

    Returns
    -------
    Event
        The Event object with noise added to the image.
    """

    if m_dark is None or example_dark_list is None:
        print("WARNING: Noise isn't being added.")
        return event

    if noise_index is None:
        noise_index = np.random.randint(0, len(example_dark_list) - 1)

    event.noise_index = noise_index
    event.image = convert_im(
        event.image,
        get_dark_sample(
            m_dark,
            [len(event.image[0]), len(event.image)],
            example_dark_list[noise_index],
        ),
    )
    return event


def noise_remover(event, threshold=50):
    denoised_image = np.copy(event.image)

    denoised_image = np.nan_to_num(denoised_image, nan=0.0)
    # Zero out pixels below the threshold
    denoised_image[denoised_image < threshold] = 0.0

    event.image = denoised_image

    return event


def extract_features(event, num_segments=15):
    """
    Extract key features from an event for classification.

    Parameters
    ----------
    event : Event
        The Event object to extract features from.
    num_segments : int, optional
        The number of segments for intensity profiling along the principal axis (default is 15).

    Returns
    -------
    np.ndarray
        Array of features including [name, length, energy, max_den, recoil_angle].
    """
    axis, mean_x, mean_y = event.get_principal_axis()
    if type(axis) is str:
        return 'fup'

    recoil_angle = event.get_recoil_angle()

    segment_distances, segment_intensities = event.get_intensity_profile(num_segments)
    length = event.get_track_length(
        num_segments, segment_distances, segment_intensities
    )
    total_intensity = event.get_track_intensity()
    max_den = event.get_max_den()
    name = event.name
    int_mean, int_median, int_skew, int_kurt, int_std = event.get_intensity_parameters(
        segment_intensities
    )

    return np.array(
        [
            name,
            length,
            total_intensity,
            max_den,
            recoil_angle,
            int_mean,
            int_skew,
            int_kurt,
            int_std,
        ]
    )


def event_processor(
    events,
    chunk_size,
    output_csv,
    dark_dir="../../../../MIGDAL/sim_ims/darks",
    binning=1,
    num_bisector_segments=15,
):
    """
    Process events in chunks and write extracted features to a CSV file.

    Parameters
    ----------
    events : generator
        A generator yielding Event objects to be processed.
    chunk_size : int
        Number of events to process and write at once.
    output_csv : str
        Path to the output CSV file where features will be saved.
    dark_dir : str
        Path to the darks folder which should contain both the master darks and the dark lists
    binning : int
        An integer N reflecting the NxN binning of the image

    Writes
    ------
    CSV file
        Updates the output CSV file with feature data for each processed chunk.
    """
    dark_list_number = 0
    m_dark = np.load(f"{dark_dir}/master_dark_{str(binning)}x{str(binning)}.npy")
    example_dark_list_unbinned = np.load(
        f"{dark_dir}/quest_std_dark_{dark_list_number}.npy"
    )
    example_dark_list = [
        bin_event(i, binning, parse_image=True) for i in example_dark_list_unbinned
    ]

    with open(output_csv, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # Optionally write a header row if your feature extraction has fixed feature names
        writer.writerow(
            [
                "name",
                "length",
                "total_intensity",
                "max_den",
                "recoil_angle",
                "int_mean",
                "int_skew",
                "int_kurt",
                "int_std",
            ]
        )  # Example headers

        chunk = []
        count = 0

        for event in tqdm(events):
            event = noise_adder(event, m_dark, example_dark_list)
            # event = noise_remover(event)
            event = smooth_operator(event)
            features = extract_features(event, num_segments=num_bisector_segments)
            if type(features) is str:
                continue
            chunk.append(features)

            if len(chunk) >= chunk_size:
                writer.writerows(chunk)
                chunk = []
                count += chunk_size
                if (
                    count // 5000 > dark_list_number
                ):  # Change the 5000 here if things go wrong
                    dark_list_number += 1
                    example_dark_list = np.load(
                        f"{dark_dir}/quest_std_dark_{dark_list_number}.npy"
                    )
                    # print("Success!")

        if chunk:
            writer.writerows(chunk)


def yield_events(base_dirs):
    """
    Generator function to load and yield Event objects from .npy files within specified directories.

    Parameters
    ----------
    base_dirs : list of str
        List of base directory paths containing .npy event files.

    Yields
    ------
    Event
        An Event object for each .npy file found in the specified directories.
    """

    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            # Sort directories and files to ensure consistent order
            dirs.sort()  # Sort directories alphabetically
            files = sorted(
                f for f in files if f.endswith(".npy")
            )  # Sort and filter files for .npy

            # np.random.seed(5)
            np.random.shuffle(files)

            for file in files:
                file_path = os.path.join(root, file)
                # Load the event data from the .npy file
                image = np.load(file_path)
                event = Event(file, image)

                yield event


def load_event(event_name, cluster_path="../../../../MIGDAL/sim_ims"):
    # Determine if the event is in the "C" or "F" folder
    element_folder = "C" if "C" in event_name else "F"

    # Extract the energy value (before "keV") from the event name
    energy_str = event_name.split("keV")[0]
    energy_value = float(energy_str)  # Convert energy to float

    # Determine the energy range folder
    lower_bound = int(energy_value // 20) * 20
    upper_bound = lower_bound + 20
    energy_folder = f"{lower_bound}-{upper_bound}keV"

    # Construct the full path to the event file
    full_path = os.path.join(cluster_path, element_folder, energy_folder, event_name)

    # Check if the file exists and load it
    if os.path.exists(full_path):
        event_data = np.load(full_path)
        event = Event(event_name, event_data)
        return event
    else:
        raise FileNotFoundError(
            f"Event file '{event_name}' not found in the expected directory '{full_path}'."
        )


def load_events(file_path):
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
            event = Event(filename, image)

            # Append to events list
            events.append(event)

    return events
