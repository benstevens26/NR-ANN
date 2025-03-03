import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
import datetime
import glob
import random

from bb_event import *
from cnn_processing import (
    NoiseAdder,
    SmoothOperator,
    load_data,
    load_data_yield,
    load_data_yield_bb,
    PreprocessingLayer,
    yield_preprocessed_data
)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.activations import softmax  # type: ignore
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input  # type: ignore
from tensorflow.keras.layers import *  # type: ignore

use_working_version = True
use_preprocessed = True

print(
    """
      -=+=-
      Checkpoint #1
      -=+=-
      """
)

# def load_all_bb_events(base_dirs: list):
#     """
#     Generator function to load and yield Event objects from .npy files within specified directories.

#     Parameters
#     ----------
#     base_dirs : list of str
#         List of base directory paths containing .npy event files.

#     Yields
#     ------
#     Event
#         An Event object for each .npy file found in the specified directories.
#     """
#     events = []
#     for base_dir in base_dirs:
#         for root, dirs, files in os.walk(base_dir):
#             # Sort directories and files to ensure consistent order
#             dirs.sort()  # Sort directories alphabetically
#             files = sorted(
#                 f for f in files if f.endswith(".npy")
#             )  # Sort and filter files for .npy

#             for file in files:
#                 file_path = os.path.join(root, file)
#                 # Load the event data from the .npy file
#                 image = np.load(file_path)
#                 event = BB_Event(file, image)
#                 events.append(event)
#     return events


# def load_image_subset(
#     directory: str = "/vols/lz/MIGDAL/sim_ims",
#     frac: float = "0.4",
#     even_split: bool = True,
#     N_C: int = 9925,
#     N_F: int = 39647,
# ):
#     """loads a subset of the images of size frac. Provide the data directory and it will load the appropriate subset.

#     Args:
#         directory (str): directory of the Data folder. The folder should contain a C folder and an F folder, each with events.
#         frac (float): the fraction in [0,1] of data that is to be loaded. We have ~50,000 images by default so e.g. frac=0.2 would total 10,000.
#         even_split (bool): determines if the function loads an equal number of each element. If False, it will load a representative sample by default
#         N_C (int): total number of carbon events
#         N_F (int): total number of fluorine events
#     """
#     if even_split and frac > (2 * N_C / (N_C + N_F)):
#         raise Exception("Not enough carbon events to do an even split.")

#     C_dir = f"{directory}/C"
#     F_dir = f"{directory}/F"

#     loaded_N = (N_C + N_F) * frac
#     loaded_N_C = (
#         int(0.5 * loaded_N // 1)
#         if even_split
#         else int((N_C / (N_C + N_F)) * loaded_N // 1)
#     )
#     loaded_N_F = (
#         int(0.5 * loaded_N // 1)
#         if even_split
#         else int((N_F / (N_C + N_F)) * loaded_N // 1)
#     )

#     # Collect all files from subdirectories
#     event_dirs = []
#     for type in [[C_dir, loaded_N_C], [F_dir, loaded_N_F]]:
#         base_dir = type[0]
#         N = type[1]
#         all_events = []
#         for subdir in os.scandir(base_dir):
#             if subdir.is_dir():
#                 # Use glob to get all files in the current subdirectory
#                 files_in_subdir = glob.glob(os.path.join(subdir.path, "*"))
#                 all_events.extend(files_in_subdir)

#         # Ensure no duplicates and enough files are available
#         all_events = list(set(all_events))  # Remove any duplicates

#         # Randomly select the specified number of files
#         selected_files = random.sample(all_events, N)
#         event_dirs.extend(selected_files)

#     for i in range(len(event_dirs)):
#         image = np.load(event_dirs[i])
#         image = image * 255 / np.max(image)
#         image = np.stack([image, image, image], axis=-1)
#         image = preprocess_input(image)
#         event_dirs[i] = BB_Event(event_dirs[i], image)

#     return event_dirs


print("==========================================")
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
print("GPU Device Name:", tf.test.gpu_device_name())
print("==========================================")
# List available GPUs
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

# Check if TensorFlow is using the GPU
if tf.test.gpu_device_name():
    print("Default GPU Device: ", tf.test.gpu_device_name())
else:
    print("GPU not detected.")
print("==========================================")


gpus = tf.config.list_physical_devices("GPU")
print(gpus)
if gpus:
    try:
        # Set memory growth to prevent TensorFlow from allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, False)
            print(f"Using GPU: {gpus[0].name}")
            memory_info = tf.config.experimental.get_memory_info(gpu.name)
            current_mb = memory_info['current'] / (1024 ** 2)
            peak_mb = memory_info['peak'] / (1024 ** 2)
            print(f"GPU {gpu.name}:")
            print(f"  Current memory usage: {current_mb:.2f} MB")
            print(f"  Peak memory usage: {peak_mb:.2f} MB")
    except Exception as e:
            print(f"Could not retrieve memory info for GPU {gpu.name}: {e}")
    except RuntimeError as e:
        print(f"Error while setting memory growth: {e}")
# HOPEFULLY this means it will automatically use the gpu from this point?

print(
    """
      -=+=-
      Checkpoint #2
      -=+=-
      """
)

# Define base directories and batch size
# with tf.device(gpus[0].name):
if use_preprocessed:
    base_dirs = ["/vols/lz/twatson/ANN/preprocessed_images"]
else:
    base_dirs = [
    "/vols/lz/tmarley/GEM_ITO/run/im0/C",
    "/vols/lz/tmarley/GEM_ITO/run/im0/F",
    "/vols/lz/tmarley/GEM_ITO/run/im1/C",
    "/vols/lz/tmarley/GEM_ITO/run/im1/F",
    "/vols/lz/tmarley/GEM_ITO/run/im2/C",
    "/vols/lz/tmarley/GEM_ITO/run/im2/F",
    "/vols/lz/tmarley/GEM_ITO/run/im3/C",
    "/vols/lz/tmarley/GEM_ITO/run/im3/F",
    "/vols/lz/tmarley/GEM_ITO/run/im4/C",
    "/vols/lz/tmarley/GEM_ITO/run/im4/F",
]
# base_dirs = ["ANN-code/Data/C", "ANN-code/Data/F"]  # List your data directories here
# base_dirs = ["/vols/lz/MIGDAL/sim_ims/C", "/vols/lz/MIGDAL/sim_ims/F"]


batch_size = 16
dark_list_number = 0
binning = 1
dark_dir = "/vols/lz/MIGDAL/sim_ims/darks"
# dark_dir="Data/darks"
m_dark = np.load(f"{dark_dir}/master_dark_{str(binning)}x{str(binning)}.npy")
example_dark_list_unbinned = np.load(
    f"{dark_dir}/quest_std_dark_{dark_list_number}.npy"
)


########################trying yielding####################

print(
    """
      -=+=-
      Checkpoint #2.5
      -=+=-
      """
)

m_dark_tensor = tf.convert_to_tensor(m_dark, dtype=tf.float32)
example_dark_tensor = tf.convert_to_tensor(example_dark_list_unbinned, dtype=tf.float32)

if use_working_version:
    if use_preprocessed:
        full_dataset = tf.data.Dataset.from_generator(
            lambda: yield_preprocessed_data(base_dirs),
            output_signature=(
                tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),  # MAY NEED TO CHANGE
                tf.TensorSpec(shape=(), dtype=tf.int32),
        )
    else:
        full_dataset = tf.data.Dataset.from_generator(
            lambda: load_data_yield(base_dirs, example_dark_tensor, m_dark_tensor, 3),
            output_signature=(
                tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),  # MAY NEED TO CHANGE
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )
else:  # Failed layering approach:
    full_dataset = tf.data.Dataset.from_generator(
        lambda: load_data_yield_bb(base_dirs, 3),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),  # image
                tf.TensorSpec(shape=(2,), dtype=tf.int32),  # original_size
            ),
            tf.TensorSpec(shape=(), dtype=tf.int32),  # label
        ),
    )

#############################################################
print(
    """
      -=+=-
      Checkpoint #2.75
      -=+=-
      """
)

dataset_size = 99366 # 99989 without the  # CHANGE DEPENDING ON DATA USED
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size  # Ensure all data is used

train_dataset = full_dataset.take(train_size).repeat().batch(batch_size, drop_remainder=True) # First 70%
remaining = full_dataset.skip(train_size)  # Remaining 30%
val_dataset = full_dataset.skip(train_size).take(val_size).repeat().batch(batch_size, drop_remainder=False) # Next 15%
test_dataset = full_dataset.skip(train_size + val_size).batch(batch_size, drop_remainder=False) # Final 15%


# print(train_dataset.take(1))

print(
    """
      -=+=-
      Checkpoint #3
      -=+=-
      """
)

# Supposedly these lines will optimise the loading: https://www.tensorflow.org/guide/keras/preprocessing_layers
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

print(
    """
      -=+=-
      Checkpoint #4
      -=+=-
      """
)

# events = load_image_subset(frac=0.001)
# # data = load_all_bb_events(["/vols/lz/MIGDAL/sim_ims/C", "/vols/lz/MIGDAL/sim_ims/F"])
num_categories = 2  # Change to 3 if argon included

# X = [event.image for event in events]
# y = [event.get_species_from_name() for event in events]

# ================================OLD MODEL DEFINITION=======================================
if use_working_version:
    inputs = keras.Input(shape=(None, None, 3))  # (224, 224, 3)

    # x = NoiseAdder(m_dark=m_dark, example_dark_list=example_dark_list_unbinned)(inputs)
    # x = SmoothOperator(smoothing_sigma=3.5)(x)

    # x = tf.keras.layers.Resizing(
    #     224, 224, pad_to_aspect_ratio=True, fill_mode="constant", fill_value=0.0
    # )(
    #     inputs
    # )  # This should use tensorflow's inbuilt resizing

    ## Loading VGG16 model
    # base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    # features = base_model(x)

    # net = tf.keras.layers.Flatten()(features)
    # net = tf.keras.layers.Dense(256, activation="relu")(net)
    # net = tf.keras.layers.Dropout(0.5)(net)
    # preds = tf.keras.layers.Dense(num_categories, activation="softmax")(net)
    # model = tf.keras.Model(base_model.input, preds)

    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    net = base_model.output
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(256, activation=tf.nn.relu)(net)
    net = tf.keras.layers.Dropout(0.5)(net)
    preds = tf.keras.layers.Dense(num_categories, activation=tf.nn.softmax)(net)
    model = tf.keras.Model(base_model.input, preds)
# ======================================================================================
# model = tf.keras.models.load_model(
#     "/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/CNN_checkpoints/epoch-04.keras",
#     custom_objects={"softmax_v2": softmax}  # Map softmax_v2 to softmax
# )

elif not use_working_version:
    images_input = keras.Input(shape=(None, None, 3), name="images")
    shapes_input = keras.Input(shape=(2,), name="original_shape", dtype=tf.int32)

    # Apply your preprocessing layer.
    x = PreprocessingLayer(
        smoothing_sigma=3.5,
        m_dark=m_dark,  # Replace with your parameter
        example_dark_list=example_dark_list_unbinned,  # Replace with your list
        target_size=(224, 224),
    )([images_input, shapes_input])

    x = tf.keras.layers.Resizing(
        224, 224, pad_to_aspect_ratio=True, fill_mode="constant", fill_value=0.0
    )(
        x
    )  # This should use tensorflow's inbuilt resizing

    # Now x is a batch of images with shape (batch, 224, 224, 3).
    # You can feed it into your base model.
    base_model = VGG16(weights="imagenet", include_top=False)
    features = base_model(x)

    # Build your classification head.
    x = tf.keras.layers.Flatten()(features)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(10, activation="softmax")(
        x
    )  # adjust number of categories

    # Create the model.
    model = keras.Model(inputs=[images_input, shapes_input], outputs=predictions)


# Ensure input dtype is tf.float32
# model.build(input_shape=(None, 572, 562, 3))
# model.layers[0].input_dtype = tf.float32

freeze = False
# Freeze convolutional layers if needed
if freeze:
    for layer in model.layers[:-4]:
        layer.trainable = False

opt = tf.keras.optimizers.Adam(
    learning_rate=1e-6
)  # Default value from the paper I'm "leaning on". Good to have very low learning rate for transfer learning
loss = tf.keras.losses.SparseCategoricalCrossentropy()

# "binary_crossentropy" if num_categories == 2 else
model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

# Setup TensorBoard callback
log_dir = "/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs"
tb_callback = tf.keras.callbacks.TensorBoard(log_dir)

# Setup checkpoint callback
os.makedirs(os.path.join(log_dir, "ckpt"), exist_ok=True)
ckpt_path = os.path.join(log_dir, "ckpt", "epoch-{epoch:02d}.keras")

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path,
    save_weights_only=False,
    # period=1,
    save_best_only=False,
    monitor="val_loss",
)

print(
    """
      -=+=-
      Checkpoint #5
      -=+=-
      """
)

# # Split into 70% train, 15% validation, 15% test
# train_ratio = 0.70
# validation_ratio = 0.15
# test_ratio = 0.15

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=1 - train_ratio, random_state=42
# )

# X_val, X_test, y_val, y_test = train_test_split(
#     X_test,
#     y_test,
#     test_size=test_ratio / (test_ratio + validation_ratio),
#     random_state=42,
# )

## Preprocessing input
# X_train = preprocess_input(np.array(X_train))
# X_test = preprocess_input(np.array(X_test))
# X_val = preprocess_input(np.array(X_val))

epochs = 40

train_start_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

print("After loading dataset")
print(train_dataset)


print(
    """
      -=+=-
      Checkpoint #6
      -=+=-
      """
)


# for sample in train_dataset.take(1): # doesn't work
#     print(sample)
#     plt.imshow(sample[0])
#     plt.savefig("test",dpi=200)


early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

# load in epoch 1
# model.load_weights("/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/CoNNCR-R/v2/epoch-01.keras")

history = model.fit(
    train_dataset,
    epochs=epochs,
    # initial_epoch=1,
    steps_per_epoch=(train_size // batch_size),
    batch_size=batch_size,
    validation_data=val_dataset,
    verbose=1,
    class_weight=None,  # look into changing this, might be good to
    callbacks=[tb_callback, ckpt_callback, early_stopping],
)

print(
    """
      -=+=-
      Checkpoint #7
      -=+=-
      """
)

train_end_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

history_filename = os.path.join(log_dir, "history.json")

model_save_path = "/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/CoNNCR-R.keras"
try:
    model.save(model_save_path)
except:
    model.save("CoNNCR-R.keras")

print(
    """
      -=+=-
      Checkpoint #8
      -=+=-
      """
)
info_filename = os.path.join(log_dir, "info.txt")

with open(history_filename, "w") as file:
    json.dump(history.history, file)

with open(info_filename, "w") as file:
    file.write("***Training Info***\n")
    file.write("Training Start: {}".format(train_start_time))
    file.write("Training End: {}\n".format(train_end_time))
    file.write("Arguments:\n")
    for arg in sys.argv:
        file.write("\t{}\n".format(arg))
