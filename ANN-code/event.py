"""
This module contains the `Event` class, which endows a nuclear recoil image with attributes.
"""

import json
import re

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from numpy.linalg import svd
from scipy.stats import skew, kurtosis
import os


script_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(script_dir, "matplotlibrc.json")
with open(config_path, "r") as file:  # For reading the matplotlibrc.json file
    custom_params = json.load(file)

plt.rcParams.update(custom_params)


class Event:
    """
    A class to hold information and process data for an individual event.

    Attributes:
    ----------
    name : str
        The filename of the event
    image : np.ndarray
        2D array representing the image data of the event.
    species : str
        The extracted species (C or F) from the filename.
    energy : float
        The extracted energy in keV from the filename.
    depth : float
        The extracted drift length in cm from the filename.
    plot_name : str
        Event information for plot tiles.
    principal_axis : np.ndarray or None
        The principal axis direction vector.
    bisectors : list or None
        A list of bisectors calculated along the principal axis.
    mean_x : float or None
        The mean x-coordinate of non-zero pixels in the image.
    mean_y : float or None
        The mean y-coordinate of non-zero pixels in the image.
    """

    def __init__(self, name, image):
        """
        Initialize an Event instance.

        Parameters:
        ----------
        name : str
            The filename of the event
        image : np.ndarray
            2D array representing the image data of the event. Could be raw or processed.
        """
        self.name = name
        self.image = image
        self.noise_index = None
        self.error = False

        self.species = self.get_species_from_name()
        self.energy = self.get_energy_from_name()
        self.depth = self.get_depth_from_name()

        self.plot_name = str(self.energy) + "keV" + " " + self.species

        self.principal_axis = None
        self.bisectors = None
        self.mean_x = None
        self.mean_y = None

    def get_energy_from_name(self):
        """
        Extract the energy in keV from the filename.

        Returns:
        -------
        float
            The energy in keV.
        """
        match = re.search(r"(\d+\.?\d*)keV", self.name)
        if match:
            return float(match.group(1))
        else:
            raise ValueError("Could not extract energy.")

    def get_species_from_name(self):
        """
        Extract the species (C for carbon or F for fluorine) from the filename.

        Returns:
        -------
        str
            The species (either 'C' or 'F').
        """
        match = re.search(r"_([C|F])_", self.name)
        if match:
            return match.group(1)
        else:
            raise ValueError("Could not extract species.")

    def get_depth_from_name(self):
        """
        Extract the depth (drift length) in cm from the filename.

        Returns:
        -------
        float
            The depth in cm.
        """
        match = re.search(r"(\d+\.?\d*)cm", self.name)
        if match:
            return float(match.group(1))
        else:
            raise ValueError("Could not extract length.")

    def get_attributes(self):
        """
        Return primary attributes of the event.

        Returns:
        -------
        tuple
            name, energy, species, and depth
        """
        return self.name, self.energy, self.species, self.depth

    def get_principal_axis(self):
        """
        Calculate the principal axis of the image via SVD

        Returns:
        -------
        tuple
            Principal axis vector (np.ndarray), mean x-coordinate, and mean y-coordinate.
        """

        image = self.image
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

        return principal_axis, mean_x, mean_y

    def get_track_length(
        self, num_segments=15, segment_distances=None, segment_intensities=None
    ):
        """
        Calculate the track length from segmented distances and intensities.

        Parameters:
        ----------
        num_segments : int, optional
            Number of segments along the principal axis. Defaults to 15.
        segment_distances : list, optional
            Cumulative distances along the principal axis. Defaults to None.
        segment_intensities : list, optional
            Total intensities for each segment. Defaults to None.

        Returns:
        -------
        float
            The calculated track length.
        """

        if segment_distances is None:
            segment_distances, segment_intensities = self.get_intensity_profile(
                num_segments
            )

        non_zero_indices = [
            i for i, intensity in enumerate(segment_intensities) if intensity > 0
        ]

        if not non_zero_indices:
            # If there are no non-zero segments, return zero length
            return 0.0

        # First and last non-zero intensity segment indices
        first_non_zero_index = non_zero_indices[0]
        last_non_zero_index = non_zero_indices[-1]

        # Calculate the midpoints of the first and last non-zero segments
        try:
            midpoint_first = (
                segment_distances[first_non_zero_index]
                + segment_distances[first_non_zero_index + 1]
            ) / 2
        except:
            print(
                "Something has gone wrong with getting the midpoints of the non-zero segments. Is there only one non-zero segment?"
            )
            midpoint_first = segment_distances[first_non_zero_index]
            self.error = True
        if last_non_zero_index < len(segment_distances) - 1:
            midpoint_last = (
                segment_distances[last_non_zero_index]
                + segment_distances[last_non_zero_index + 1]
            ) / 2
        else:
            # If last_non_zero_index is the last segment, use only that segment's distance
            midpoint_last = segment_distances[last_non_zero_index]
        # Calculate the track length as the distance between the two midpoints
        track_length = abs(midpoint_last - midpoint_first)

        return track_length

    def get_track_intensity(self):
        """
        Calculates a proxy for track energy by summing pixel intensities.

        Returns:
        -------
        float
            The total track intensity
        """
        return np.sum(self.image)

    def get_max_den(self):
        """
        Calculate the maximum density as 1 over the max pixel value.

        Returns:
        -------
        float
            The maximum density.
        """
        max_den = 1 / np.max(self.image)
        return max_den

    def get_recoil_angle(self):
        """
        Calculate the angle between the principal axis and the +x direction.

        Returns:
        -------
        float
            The angle in degrees.
        """

        if self.principal_axis is None:
            self.principal_axis, self.mean_x, self.mean_y = self.get_principal_axis()

        principal_axis = self.principal_axis

        x_direction = np.array([1, 0])

        # Normalize the principal axis vector
        principal_axis = np.array(principal_axis)
        principal_axis_norm = principal_axis / np.linalg.norm(principal_axis)

        # Calculate the angle using the dot product formula
        dot_product = np.dot(principal_axis_norm, x_direction)
        angle_rad = np.arccos(dot_product)  # Angle in radians

        # Convert the angle to degrees
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def get_bisectors(self, num_segments=15):
        """
        Calculate bisectors along the principal axis

        Parameters:
        ----------
        num_segments : int, optional
            Number of segments to divide the principal axis into. Defaults to 15.

        Returns:
        -------
        list of tuple
            List of bisector start and end points as tuples.
        """

        image = self.image

        if self.principal_axis is None:
            self.principal_axis, self.mean_x, self.mean_y = self.get_principal_axis()

        principal_axis, mean_x, mean_y = self.principal_axis, self.mean_x, self.mean_y

        height, width = image.shape

        # Calculate the length of the principal axis extended over the whole image
        line_length = np.sqrt(width**2 + height**2)

        # Extend the principal axis over the full image dimensions
        x_start = mean_x - principal_axis[0] * line_length / 2
        y_start = mean_y - principal_axis[1] * line_length / 2
        x_end = mean_x + principal_axis[0] * line_length / 2
        y_end = mean_y + principal_axis[1] * line_length / 2

        # Divide the principal axis into equal segments
        segment_points_x = np.linspace(x_start, x_end, num_segments + 1)
        segment_points_y = np.linspace(y_start, y_end, num_segments + 1)

        # List to store bisectors' start and end points
        bisectors = []

        # Calculate the perpendicular bisectors
        for i in range(num_segments + 1):
            # Get the current segment boundary point
            x_point = segment_points_x[i]
            y_point = segment_points_y[i]

            # Perpendicular vector direction (rotate 90 degrees)
            perp_vector = np.array([-principal_axis[1], principal_axis[0]])

            # Extend the perpendicular bisector for calculation
            perp_length = line_length / 2  # Length of perpendicular bisector
            x_bisector_start = x_point - perp_vector[0] * perp_length
            y_bisector_start = y_point - perp_vector[1] * perp_length
            x_bisector_end = x_point + perp_vector[0] * perp_length
            y_bisector_end = y_point + perp_vector[1] * perp_length

            # Append the bisector start and end points as a tuple
            bisectors.append(
                ((x_bisector_start, y_bisector_start), (x_bisector_end, y_bisector_end))
            )

        return bisectors

    def get_intensity_profile(self, num_segments=15):
        """
        Calculate the (segmented) intensity profile along the principal axis

        Parameters:
        ----------
        num_segments : int, optional
            Number of segments along the principal axis. Defaults to 15.

        Returns:
        -------
        tuple
            Lists of segment distances and intensities.
        """

        image = self.image
        if self.bisectors is None:
            self.bisectors = self.get_bisectors(num_segments)

        bisectors = self.bisectors

        segment_intensities = np.zeros(num_segments)

        # Identify non-zero pixels
        non_zero_y, non_zero_x = np.nonzero(
            image
        )  # Get y and x indices of non-zero pixels
        non_zero_pixels = zip(
            non_zero_x, non_zero_y
        )  # Create a list of (x, y) tuples for non-zero pixels

        # Function to calculate the cross product of two vectors
        def cross_product_2d(v1, v2):
            return v1[0] * v2[1] - v1[1] * v2[0]

        # Loop through each non-zero pixel
        for x, y in non_zero_pixels:
            # Get the intensity value of the current pixel
            intensity = image[y, x]

            # Convert pixel position to array
            pixel_pos = np.array([x, y])

            # Check which segment this pixel belongs to by using cross products
            for i in range(num_segments):
                # Get current bisector and the next bisector (defining the segment boundaries)
                (x_bisector_start, y_bisector_start), (
                    x_bisector_end,
                    y_bisector_end,
                ) = bisectors[i]
                (x_next_bisector_start, y_next_bisector_start), (
                    x_next_bisector_end,
                    y_next_bisector_end,
                ) = bisectors[i + 1]

                # Define the vectors from the pixel to the start of the bisectors
                vector_to_bisector = pixel_pos - np.array(
                    [x_bisector_start, y_bisector_start]
                )
                vector_to_next_bisector = pixel_pos - np.array(
                    [x_next_bisector_start, y_next_bisector_start]
                )

                # Define the direction vectors of the bisectors
                bisector_vector = np.array(
                    [
                        x_bisector_end - x_bisector_start,
                        y_bisector_end - y_bisector_start,
                    ]
                )
                next_bisector_vector = np.array(
                    [
                        x_next_bisector_end - x_next_bisector_start,
                        y_next_bisector_end - y_next_bisector_start,
                    ]
                )

                # Calculate the cross products to check which side of the bisector the pixel is on
                cross_current = cross_product_2d(bisector_vector, vector_to_bisector)
                cross_next = cross_product_2d(
                    next_bisector_vector, vector_to_next_bisector
                )

                # Check if the pixel lies between the two bisectors
                if cross_current <= 0 <= cross_next:
                    # Add the pixel intensity to the corresponding segment
                    segment_intensities[i] += intensity
                    break  # Once classified, move to the next pixel

        # Calculate actual distances along the principal axis
        segment_distances = []
        total_distance = 0

        for i in range(num_segments):
            # Get the current segment's start and next bisector
            (x_bisector_start, y_bisector_start), (
                x_bisector_end,
                y_bisector_end,
            ) = bisectors[i]
            (x_next_bisector_start, y_next_bisector_start), (
                x_next_bisector_end,
                y_next_bisector_end,
            ) = bisectors[i + 1]

            # Calculate the distance between current bisector and next bisector
            distance = np.sqrt(
                (x_next_bisector_start - x_bisector_start) ** 2
                + (y_next_bisector_start - y_bisector_start) ** 2
            )
            total_distance += distance
            segment_distances.append(total_distance)

        return segment_distances, segment_intensities

    def get_intensity_parameters(self, segment_intensities):
        mean = np.mean(segment_intensities)
        skew = scipy.stats.skew(segment_intensities)
        kurt = kurtosis(segment_intensities)
        median = np.median(segment_intensities)
        std = np.std(segment_intensities)

        return mean, median, skew, kurt, std

    def plot_image(self):
        """
        Display the event image.
        """
        image = self.image

        fig, ax = plt.subplots()

        ax.grid(False)
        ax.imshow(image)

        plt.title(self.plot_name + " image")
        plt.show()

    def plot_image_with_principal_axis(self):
        """
        Plot the event image overlaid with the principal axis.
        """
        image = self.image

        if self.principal_axis is None:
            self.principal_axis, self.mean_x, self.mean_y = self.get_principal_axis()

        height, width = image.shape

        # Calculate the length of the principal axis extended over the whole image
        line_length = np.sqrt(
            width**2 + height**2
        )  # Diagonal length to ensure it covers the whole image

        # Extend the principal axis over the full image dimensions
        x_start = self.mean_x - self.principal_axis[0] * line_length / 2
        y_start = self.mean_y - self.principal_axis[1] * line_length / 2
        x_end = self.mean_x + self.principal_axis[0] * line_length / 2
        y_end = self.mean_y + self.principal_axis[1] * line_length / 2

        # Plot the image

        fig, ax = plt.subplots()

        ax.grid(False)
        ax.imshow(image)
        ax.plot(
            [x_start, x_end],
            [y_start, y_end],
            color="red",
            label="Principal Axis",
            linewidth=2,
        )
        plt.title(self.plot_name + " with principal axis")
        plt.legend()
        plt.show()

    def plot_image_with_bisectors(self, num_segments=15):
        """
        Plot the event image with the principal axis and bisectors.

        Parameters:
        ----------
        num_segments : int, optional
            Number of segments to divide the principal axis. Defaults to 15.
        """

        image = self.image

        if self.principal_axis is None:
            self.principal_axis, self.mean_x, self.mean_y = self.get_principal_axis()

        principal_axis, mean_x, mean_y = self.principal_axis, self.mean_x, self.mean_y

        if self.bisectors is None:
            self.bisectors = self.get_bisectors(num_segments)
        bisectors = self.bisectors

        # Calculate the length of the principal axis extended over the whole image
        height, width = image.shape
        line_length = np.sqrt(width**2 + height**2)

        # Extend the principal axis over the full image dimensions
        x_start = mean_x - principal_axis[0] * line_length / 2
        y_start = mean_y - principal_axis[1] * line_length / 2
        x_end = mean_x + principal_axis[0] * line_length / 2
        y_end = mean_y + principal_axis[1] * line_length / 2

        fig, ax = plt.subplots()
        plt.imshow(image, origin="lower")
        ax.plot(
            [x_start, x_end],
            [y_start, y_end],
            color="red",
            label="Principal Axis",
            linewidth=2,
        )

        for bisector in bisectors:
            (x_bisector_start, y_bisector_start), (
                x_bisector_end,
                y_bisector_end,
            ) = bisector
            ax.plot(
                [x_bisector_start, x_bisector_end],
                [y_bisector_start, y_bisector_end],
                "blue",
                linestyle="-",
                label="Bisector" if bisector == bisectors[0] else "",
            )

        plt.title(self.plot_name + " with principal axis and bisectors")
        plt.legend()
        plt.show()

    def plot_intensity_profile(
        self, num_segments=15, segment_distances=None, segment_intensities=None
    ):
        """
        Plot the intensity profile as a function of distance along the principal axis.

        Parameters:
        ----------
        num_segments : int, optional
            Number of segments along the principal axis. Defaults to 15.
        segment_distances : list, optional
            Distances along the principal axis. Defaults to None.
        segment_intensities : list, optional
            Total intensities for each segment. Defaults to None.
        """

        if segment_distances is None:
            segment_distances, segment_intensities = self.get_intensity_profile(
                num_segments
            )
        # Plot the intensity profile as a function of the actual distance along the principal axis
        plt.bar(
            segment_distances,
            segment_intensities,
            width=segment_distances[1] - segment_distances[0],
        )
        plt.xlabel("Distance Along Principal Axis (pixels)")
        plt.ylabel("Total Intensity")
        plt.title(self.plot_name + " with intensity profile")
        plt.show()
