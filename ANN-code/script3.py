from bb_event import load_events_bb
import numpy as np

base_dirs = ["/vols/lz/tmarley/GEM_ITO/run/im0", "/vols/lz/tmarley/GEM_ITO/run/im1/C", "/vols/lz/tmarley/GEM_ITO/run/im1/F",
             "/vols/lz/tmarley/GEM_ITO/run/im2", "/vols/lz/tmarley/GEM_ITO/run/im3", "/vols/lz/tmarley/GEM_ITO/run/im4" ]

max_width = 0
max_height = 0

for path in base_dirs:
    events = load_events_bb(path)

    for event in events:
        shape = np.shape(event.image)
        # shape is (height, width)
        if shape[0] > max_height:
            max_height = shape[0]
        if shape[1] > max_width:
            max_width = shape[1]

print(max_height, max_width)
