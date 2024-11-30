import datetime

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import os
import glob
from bb_event import *
from tensorflow.keras.layers import *
import random


def load_all_bb_events(base_dirs: list):
    """
    Generator function to load and yield Event objects from .npy files within specified directories.

    Parameters
    ----------
    base_dirs : list of str
        List of base directory paths containing .npy event files.

    Yields
    ------
    Event
        An Event object for each .npy file found in the specified directories.
    """
    events = []
    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            # Sort directories and files to ensure consistent order
            dirs.sort()  # Sort directories alphabetically
            files = sorted(
                f for f in files if f.endswith(".npy")
            )  # Sort and filter files for .npy

            for file in files:
                file_path = os.path.join(root, file)
                # Load the event data from the .npy file
                image = np.load(file_path)
                event = BB_Event(file, image)
                events.append(event)
    return events


def load_image_subset(
    directory: str = "/vols/lz/MIGDAL/sim_ims",
    frac: float = "0.4",
    even_split: bool = True,
    N_C: int = 9925,
    N_F: int = 39647,
):
    """loads a subset of the images of size frac. Provide the data directory and it will load the appropriate subset.

    Args:
        directory (str): directory of the Data folder. The folder should contain a C folder and an F folder, each with events.
        frac (float): the fraction in [0,1] of data that is to be loaded. We have ~50,000 images by default so e.g. frac=0.2 would total 10,000.
        even_split (bool): determines if the function loads an equal number of each element. If False, it will load a representative sample by default
        N_C (int): total number of carbon events
        N_F (int): total number of fluorine events
    """
    if even_split and frac > (2 * N_C / (N_C + N_F)):
        raise Exception("Not enough carbon events to do an even split.")

    C_dir = f"{directory}/C"
    F_dir = f"{directory}/F"

    loaded_N = (N_C + N_F) * frac
    loaded_N_C = (
        int(0.5 * loaded_N // 1)
        if even_split
        else int((N_C / (N_C + N_F)) * loaded_N // 1)
    )
    loaded_N_F = (
        int(0.5 * loaded_N // 1)
        if even_split
        else int((N_F / (N_C + N_F)) * loaded_N // 1)
    )

    # Collect all files from subdirectories
    event_dirs = []
    for type in [[C_dir, loaded_N_C], [F_dir, loaded_N_F]]:
        base_dir = type[0]
        N = type[1]
        all_events = []
        for subdir in os.scandir(base_dir):
            if subdir.is_dir():
                # Use glob to get all files in the current subdirectory
                files_in_subdir = glob.glob(os.path.join(subdir.path, "*"))
                all_events.extend(files_in_subdir)

        # Ensure no duplicates and enough files are available
        all_events = list(set(all_events))  # Remove any duplicates

        # Randomly select the specified number of files
        selected_files = random.sample(all_events, N)
        event_dirs.extend(selected_files)

    for i in range(len(event_dirs)):
        image = np.load(event_dirs[i])
        image_3C = np.stack([image, image, image], axis=-1)
        event_dirs[i] = BB_Event(event_dirs[i], image_3C)

    return event_dirs

events = load_image_subset(frac=0.001)
# data = load_all_bb_events(["/vols/lz/MIGDAL/sim_ims/C", "/vols/lz/MIGDAL/sim_ims/F"])
num_categories = 2  # Change to 3 if argon included


# Making input data into 3 channels
for i in range(len(data)):
    data[i]


## Loading VGG16 model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(415, 559, 3))
net = base_model.output
net = tf.keras.layers.Flatten()(net)
net = tf.keras.layers.Dense(256, activation=tf.nn.relu)(net)
net = tf.keras.layers.Dropout(0.5)(net)
preds = tf.keras.layers.Dense(num_categories, activation=tf.nn.softmax)(net)
model = tf.keras.Model(base_model.input, preds)


freeze = False
# Freeze convolutional layers if needed
if freeze:
    for layer in model.layers[:-4]:
        layer.trainable = False


opt = tf.keras.optimizers.Adam(
    learning_rate=1e-6, decay=0
)  # Default values from the paper I'm "leaning on". Good to have very low learning rate for transfer learning
loss = "binary_crossentropy" if num_categories == 2 else "categorical_crossentropy"

model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

# Setup TensorBoard callback
log_dir = "/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs"
tb_callback = tf.keras.callbacks.TensorBoard(log_dir)


# Setup checkpoint callback
os.makedirs(os.path.join(log_dir, "ckpt"))
ckpt_path = os.path.join(log_dir, "ckpt", "epoch-{epoch:02d}.keras")


ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path,
    save_weights_only=False,
    # period=1,
    save_best_only=False,
    monitor="val_loss",
)


# Split into 70% train, 15% validation, 15% test
train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 - train_ratio, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_test,
    y_test,
    test_size=test_ratio / (test_ratio + validation_ratio),
    random_state=42,
)


## Preprocessing input
X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)
X_val = preprocess_input(X_val)


epochs = 10
batch_size = 32  # No clue if this is applicable. Again just guessing based on the code I'm "inspired by"

train_start_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

history = model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val, y_val),
    verbose=1,
    class_weight=None,  # look into changing this, might be good to
    callbacks=[tb_callback, ckpt_callback],
)

train_end_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

history_filename = os.path.join(log_dir, "history.json")
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