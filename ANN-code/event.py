"""
This module contains the `Event3D` class, which endows an event with attributes.
"""

class Event3D:
    """
    A class to represent a 3D event and hold its attributes.

    Attributes:
    ----------
    name : str
        The filename of the event.
    image : np.ndarray
        2D array representing the image data of the event.
    species : str
        The species ('C' for carbon or 'F' for fluorine) extracted from the filename.
    """

    def __init__(self, cam_path, cam_image, ito_path, ito_image):
        """
        Initialize an Event instance.

        Parameters:
        ----------
        name : str
            The filename of the event.
        image : np.ndarray
            2D array representing the image data of the event, either raw or processed.
        """

        self.cam_path = cam_path
        self.cam_image = cam_image
        self.ito_path = ito_path
        self.ito_image = ito_image

 