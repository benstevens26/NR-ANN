import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from convert_sim_ims import *
from feature_extraction import *

folder_path = "Data/C/300-320keV"  # Change to whichever data you want to use
events = load_events(folder_path)


m_dark_list = [np.load(f) for f in glob.glob("Data/darks/master_dark_*.npy")]
m_dark = m_dark_list[
    1
]  # This is only for while I don't care about the others. I cba to implement ordering them by the numbers at the end of the file names
example_dark_list = np.load(
    "Data/darks/quest_std_dark_1.npy"
)  # This is an array of 200 dark images. I believe the other 9 are the same as well
example_dark = example_dark_list[np.random.randint(0, len(example_dark_list) - 1)]

dark_sample = get_dark_sample(
    m_dark, [len(events[0].image[0]), len(events[0].image)], example_dark
)
# plt.imshow(dark_sample)

# events[0].noisy_image = convert_im(events[0].image, dark_sample)
# plt.imshow(events[0].noisy_image)

test_events = [events[np.random.randint(0, len(events) - 1)] for i in range(5)]


for e in test_events:
    # e.plot_image_with_axis()
    e.image = convert_im(  # THIS SHOULD NOT OVERWRITE THE IMAGE REALLY THIS IS A TEMP FIX UNTIL I NEXT MERGE
        e.image,
        get_dark_sample(
            m_dark,
            [len(e.image[0]), len(e.image)],
            example_dark_list[np.random.randint(0, len(example_dark_list) - 1)],
        ),
    )
    e.image = np.array([[j if j > 50 else 0 for j in i] for i in e.image])
    e.plot_image_with_axis()
    e.plot_intensity_profile(num_segments=200)
