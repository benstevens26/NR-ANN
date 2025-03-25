"""
Module for data management
"""

from tqdm import tqdm
import os
import numpy as np
import pandas as pd

def create_file_paths(base_dirs):
    """
    Create a list of file paths from a list of directories.

    Parameters:
    ----------
    base_dirs : list
        A list of directories containing .npy (image) files.

    Returns:
    -------
    file_paths : list
        A (randomised) list of file paths to the .npy files in the specified directories.

    """

    file_paths = []

    if not isinstance(base_dirs, list):
        base_dirs = [base_dirs]

    for dir in base_dirs:
        for filename in tqdm(os.listdir(dir)):
            if filename.endswith(".npy"):

                file_paths.append(os.path.join(dir, filename))

    np.random.shuffle(file_paths)

    return file_paths