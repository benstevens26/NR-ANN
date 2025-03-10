import os
import sys

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# physical_devices = tf.config.list_physical_devices("GPU")
# for gpu in physical_devices:
#     tf.config.experimental.set_memory_growth(gpu, True)
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
    yield_preprocessed_data,
    get_file_list
)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.activations import softmax  # type: ignore
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input  # type: ignore
from tensorflow.keras.layers import *  # type: ignore

use_working_version = False
use_preprocessed = True
use_unscaled = True
finetune = True

print(
    """
      -=+=-
      Checkpoint #1
      -=+=-
      """
)

def load_and_process(file_path):
    # file_path here is a tf.string tensor, so convert to numpy if needed
    file_path = file_path.numpy().decode("utf-8")
    image = np.load(file_path)
    label = 0 if "C" in os.path.basename(file_path) else 1
    return image, label

def tf_load_and_process(file_path):
    # Wrap the python function
    image, label = tf.py_function(func=load_and_process, inp=[file_path], Tout=[tf.float32, tf.int32])
    image.set_shape((224, 224, 3))
    label.set_shape(())
    return image, label


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
            memory_info = tf.config.experimental.get_memory_info("GPU:0")
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
    if use_unscaled:
        base_dirs = ["/vols/lz/twatson/ANN/preprocessed_images_unscaled"]
    else:
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

if use_working_version:
    file_list = get_file_list(base_dirs)
    np.random.seed(77)
    np.random.shuffle(file_list)

    # Create a dataset from the file list
    full_dataset = tf.data.Dataset.from_tensor_slices(file_list)
    full_dataset = full_dataset.map(tf_load_and_process, num_parallel_calls=tf.data.AUTOTUNE)
    full_dataset = full_dataset.shuffle(buffer_size=len(file_list),seed=77)
    # if use_preprocessed:
    #     full_dataset = tf.data.Dataset.from_generator(
    #         lambda: yield_preprocessed_data(base_dirs),
    #         output_signature=(
    #             tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),  # MAY NEED TO CHANGE
    #             tf.TensorSpec(shape=(), dtype=tf.int32),
    #     )
    #     )
    # else:
    #     m_dark_tensor = tf.convert_to_tensor(m_dark, dtype=tf.float32)
    #     example_dark_tensor = tf.convert_to_tensor(example_dark_list_unbinned, dtype=tf.float32)
    #     full_dataset = tf.data.Dataset.from_generator(
    #         lambda: load_data_yield(base_dirs, example_dark_tensor, m_dark_tensor, 3),
    #         output_signature=(
    #             tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),  # MAY NEED TO CHANGE
    #             tf.TensorSpec(shape=(), dtype=tf.int32),
    #         ),
    #     )
else:  # Tensor slice approach:
    file_list = get_file_list(base_dirs)
    np.random.seed(77)
    np.random.shuffle(file_list)

    # Create a dataset from the file list
    full_dataset = tf.data.Dataset.from_tensor_slices(file_list)
    full_dataset = full_dataset.map(tf_load_and_process, num_parallel_calls=tf.data.AUTOTUNE)
    full_dataset = full_dataset.shuffle(buffer_size=len(file_list),seed=77)
    # full_dataset = full_dataset.batch(batch_size, drop_remainder=True)
    # full_dataset = full_dataset.prefetch(tf.data.AUTOTUNE)

#############################################################
print(
    """
      -=+=-
      Checkpoint #3
      -=+=-
      """
)

dataset_size = 99366 # 99989 without the  # CHANGE DEPENDING ON DATA USED
train_size = (int(0.7 * dataset_size)//batch_size)*batch_size
val_size = (int(0.15 * dataset_size)//batch_size)*batch_size
test_size = ((dataset_size - train_size - val_size)//batch_size)*batch_size  # Ensure all data is used

train_dataset = full_dataset.take(train_size).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE) # First 70%
remaining = full_dataset.skip(train_size)  # Remaining 30%
val_dataset = remaining.take(val_size).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE) # Next 15%
test_dataset = remaining.skip(val_size).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE) # Final 15%


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

if use_working_version:
    inputs = keras.Input(shape=(224, 224, 3))
    base_model = VGG16(weights="imagenet", include_top=False, input_tensor=inputs)
    net = base_model.output
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(256, activation="relu")(net)
    net = tf.keras.layers.Dropout(0.5)(net)
    preds = tf.keras.layers.Dense(num_categories, activation="softmax")(net)
    model = tf.keras.Model(base_model.input, preds)
    num_new_layers = 4

elif not use_working_version:
    inputs = keras.Input(shape=(224, 224, 3))
    base_model = VGG16(weights="imagenet", include_top=False, input_tensor=inputs)
    net = base_model.output
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(256, activation="leaky_relu")(net)
    net = tf.keras.layers.Dropout(0.4)(net)
    net = tf.keras.layers.Dense(64, activation="leaky_relu")(net)
    net = tf.keras.layers.Dropout(0.4)(net)
    preds = tf.keras.layers.Dense(num_categories, activation="softmax")(net)
    model = tf.keras.Model(base_model.input, preds)
    num_new_layers = 6


# Ensure input dtype is tf.float32
# model.build(input_shape=(None, 572, 562, 3))
# model.layers[0].input_dtype = tf.float32


freeze = True # Freeze convolutional layers for initial training
if freeze:
    for layer in model.layers[:-num_new_layers]:
        layer.trainable = False
    opt = tf.keras.optimizers.Adam(
        learning_rate=1e-3
    )
else: # low learning rate
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


epochs = 30

print("After loading dataset")
print(train_dataset)


print(
    """
      -=+=-
      Checkpoint #6
      -=+=-
      """
)




early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True
)

# load in epoch 1
# model.load_weights("/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/CoNNCR-R/v2/epoch-01.keras")
# print(notavaraible)


train_start_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
#============================================================================

history = model.fit(
    train_dataset,
    epochs=epochs,
    # initial_epoch=1,
    # steps_per_epoch=(train_size // batch_size),
    # validation_steps = (val_size // batch_size),
    # batch_size=batch_size,
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

print(f"history stuff: {history.history.keys()}")

train_end_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

history_filename = os.path.join(log_dir, "history.json")

model_save_path = "/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/CoNNCR-R_untuned.keras"
try:
    model.save(model_save_path)
except:
    model.save("CoNNCR-R_untuned.keras")

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
#============================================================================

# with open(info_filename, "w") as file:
#     file.write("***Training Info***\n")
#     file.write("Training Start: {}".format(train_start_time))
#     file.write("Training End: {}\n".format(train_end_time))
#     file.write("Arguments:\n")
#     for arg in sys.argv:
#         file.write("\t{}\n".format(arg))

print(
    """
      -=+=-
      Checkpoint #9
      -=+=-
      """
)


# unfreeze layers

if finetune:
    freeze = False # Unfreeze convolutional layers for finetuning
    if freeze:
        for layer in model.layers[:-num_new_layers]:
            layer.trainable = False
        opt = tf.keras.optimizers.Adam(
            learning_rate=1e-3
        )
    else: # low learning rate
        for layer in model.layers[:-num_new_layers]:
            layer.trainable = True
        opt = tf.keras.optimizers.Adam(
            learning_rate=1e-6
        )
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    # "binary_crossentropy" if num_categories == 2 else
    model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

    # Setup TensorBoard callback
    log_dir = "/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs"
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir)

    # Setup checkpoint callback
    os.makedirs(os.path.join(log_dir, "ckpt","finetuned"), exist_ok=True)
    ckpt_path = os.path.join(log_dir, "ckpt","finetuned", "epoch-{epoch:02d}.keras")

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path,
        save_weights_only=False,
        # period=1,
        save_best_only=False,
        monitor="val_loss",
    )
    train_start_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    print(
        """
        -=+=-
        Checkpoint #10
        -=+=-
        """
    )
    finetuned_history = model.fit(
        train_dataset,
        epochs=epochs,
        # initial_epoch=1,
        # steps_per_epoch=(train_size // batch_size),
        # validation_steps = (val_size // batch_size),
        # batch_size=batch_size,
        validation_data=val_dataset,
        verbose=1,
        class_weight=None,  # look into changing this, might be good to
        callbacks=[tb_callback, ckpt_callback, early_stopping],
    )

    train_end_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

    finetuned_history_filename = os.path.join(log_dir, "finetuned_history.json")

    model_save_path = "/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/CoNNCR-R_tuned.keras"
    try:
        model.save(model_save_path)
    except:
        model.save("CoNNCR-R_tuned.keras")

    print(
        """
        -=+=-
        Checkpoint #11
        -=+=-
        """
    )
    info_filename = os.path.join(log_dir, "info.txt")

    with open(finetuned_history_filename, "w") as file:
        json.dump(finetuned_history.history, file)
        
    with open(info_filename, "w") as file:
        file.write("***Training Info***\n")
        file.write("Training Start: {}".format(train_start_time))
        file.write("Training End: {}\n".format(train_end_time))
        file.write("Arguments:\n")
        for arg in sys.argv:
            file.write("\t{}\n".format(arg))