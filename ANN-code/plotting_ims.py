from event_processor import event_processor, yield_events

base_dirs = ['Data/im0']
dark_dir = 'Data/darks'
chunk_size = 20
output_csv = 'test_data.csv'

events = yield_events(base_dirs)


event_processor(events, chunk_size=20, output_csv=output_csv, dark_dir=dark_dir)



