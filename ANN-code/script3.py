from event_processor import event_processor, yield_events
import numpy as np

base_dirs = ["/vols/lz/tmarley/GEM_ITO/run/im0"]
# base_dirs = ['Data/im0']
events = yield_events(base_dirs)

max_width = 0
max_height = 0

for event in events:
    shape = np.shape(event.image)
    # shape is (height, width)
    if shape[0] > max_height:
        max_height = shape[0]
    if shape[1] > max_width:
        max_width = shape[1]

print(max_height, max_width)