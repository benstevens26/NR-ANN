"""
This module contains the `Event` class, which endows a nuclear recoil image with attributes.
"""

import json
import re
import matplotlib.pyplot as plt
import os
import numpy as np


# script_dir = os.path.dirname(os.path.realpath(__file__))
# config_path = os.path.join(script_dir, "matplotlibrc.json")
# with open(config_path, "r") as file:  # For reading the matplotlibrc.json file
#     custom_params = json.load(file)

# plt.rcParams.update(custom_params)


class Event:
    """
    A class to represent an event and hold its attributes.

    Attributes:
    ----------
    name : str
        The filename of the event.
    image : np.ndarray
        2D array representing the image data of the event.
    species : str
        The species ('C' for carbon or 'F' for fluorine) extracted from the filename.
    """

    def __init__(self, name, image):
        """
        Initialize an Event instance.

        Parameters:
        ----------
        name : str
            The filename of the event.
        image : np.ndarray
            2D array representing the image data of the event, either raw or processed.
        """

        self.name = name
        self.image = image
        self.species = self.get_species_from_name()

    def get_energy_from_name(self):
        """
        Extract the energy in keV from the filename.

        Parameters:
        ----------
        None

        Returns:
        -------
        float
            The energy in keV extracted from the filename.

        Raises:
        -------
        ValueError
            If the energy cannot be determined from the filename.
        """

        match = re.search(r"(\d+\.?\d*)keV", self.name)
        if match:
            return float(match.group(1))
        else:
            raise ValueError("Could not extract energy.")

    def get_species_from_name(self):
        """
        Extract the species (C for carbon or F for fluorine) from the filename.

        Parameters:
        ----------
        None

        Returns:
        -------
        str
            The species (either 'C' or 'F') extracted from the filename.

        Raises:
        -------
        ValueError
            If the species cannot be determined from the filename.
        """

        if "C" in self.name:
            return "C"

        if "F" in self.name:
            return "F"

        raise ValueError("Could not extract species.")

    def plot_image(self):
        """
        Display the event image.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """

        image = self.image
        energy = self.get_energy_from_name()
        plot_name = str(energy) + "keV" + " " + self.species

        fig, ax = plt.subplots()

        ax.grid(False)
        ax.imshow(image)

        plt.title(plot_name)
        plt.show()


def load_events_bb(file_path):
    """
    Load barebones events from a folder, processing .npy files into Event objects.

    Parameters:
    ----------
    file_path : str
        Path to the folder containing .npy files.

    Returns:
    -------
    events : list of Event
        A list of Event objects created from the .npy files in the specified folder.
    """

    events = []

    for filename in os.listdir(file_path):
        if filename.endswith(".npy"):
            # construct the full file path for each .npy file in directory
            full_path = os.path.join(file_path, filename)

            image = np.load(full_path)
            event = Event(filename, image)
            events.append(event)

    return events
