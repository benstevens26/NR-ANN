#!/usr/bin/env python3

print("Testing Python environment.")

# Test imports
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import scipy as sp
    from tqdm import tqdm
    import os
    import sys
    from image_preprocessing import noise_adder, gaussian_smoothing
    from bb_event import Event3D
    from feature_extraction import preprocess_3d, extract_R, extract_axis_3d, extract_recoil_angle_3d

    print("All packages imported successfully!")
except ImportError as e:
    print("Import failed:", e)

