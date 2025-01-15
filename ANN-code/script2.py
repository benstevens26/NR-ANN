from event_processor import event_processor, yield_events

base_dirs = ["/vols/lz/tmarley/GEM_ITO/run/im4"]
dark_dir = "/vols/lz/MIGDAL/sim_ims/darks"

chunk_size = 30
output_csv = "features_im4.csv"

events = yield_events(base_dirs)

event_processor(events, chunk_size, output_csv, dark_dir)
