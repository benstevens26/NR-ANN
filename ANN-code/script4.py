from event_processor import yield_events
import numpy as np
from tqdm import tqdm

base_dirs = ["/vols/lz/tmarley/GEM_ITO/run/im0", "/vols/lz/tmarley/GEM_ITO/run/im1/C", "/vols/lz/tmarley/GEM_ITO/run/im1/F",
             "/vols/lz/tmarley/GEM_ITO/run/im2", "/vols/lz/tmarley/GEM_ITO/run/im3", "/vols/lz/tmarley/GEM_ITO/run/im4" ]

# base_dirs = ["Data/im0", "Data/im2"]

events = yield_events(base_dirs)

max_width = 0
max_height = 0

for event in tqdm(events):
    shape = np.shape(event.image)
    # shape is (height, width)
    if shape[0] > max_height:
        max_height = shape[0]
    if shape[1] > max_width:
        max_width = shape[1]

print(max_height, max_width)
