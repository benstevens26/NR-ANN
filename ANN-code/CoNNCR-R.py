import os
import sys

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from cnn_processing import get_file_list
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16  # type: ignore
from tensorflow.keras.layers import *  # type: ignore
import json

finetune = True
exclude_low_e = True

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



base_dirs = ["/vols/lz/twatson/ANN/final_ims"]
batch_size = 16
binning = 1



file_list = get_file_list(base_dirs) if exclude_low_e else get_file_list(base_dirs,min_energy=0)
file_list = sorted(file_list) # only shuffle ONCE so that sets are easily reproducable
np.random.seed(77) 
np.random.shuffle(file_list)
print(f"FIRST AND LAST ELEMENTS OF FILE LIST: {file_list[0], file_list[-1]}")




dataset_size = len(file_list) # i might be stupid lmao
train_size = (int(0.7 * dataset_size)//batch_size)*batch_size
val_size = (int(0.15 * dataset_size)//batch_size)*batch_size
test_size = ((dataset_size - train_size - val_size)//batch_size)*batch_size  # Ensure all data is used

train_file_list = file_list[:train_size]
val_file_list = file_list[train_size:train_size + val_size]
test_file_list = file_list[train_size + val_size : train_size + val_size + test_size]
print(f"FIRST AND LAST ELEMENTS OF TEST SET: {test_file_list[0], test_file_list[-1]}")



# First 70% for training
train_dataset = tf.data.Dataset.from_tensor_slices(train_file_list)
train_dataset = train_dataset.map(tf_load_and_process, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# Next 15% for val
val_dataset = tf.data.Dataset.from_tensor_slices(val_file_list)
val_dataset = val_dataset.map(tf_load_and_process, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE) # Next 15%

# Next 15 % for test
test_dataset = tf.data.Dataset.from_tensor_slices(test_file_list)
test_dataset = test_dataset.map(tf_load_and_process, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE) # Final 15%




num_categories = 2  # Change to 3 if argon included


# Define model
inputs = keras.Input(shape=(224, 224, 3))
base_model = VGG16(weights="imagenet", include_top=False, input_tensor=inputs)
net = base_model.output
net = tf.keras.layers.Flatten()(net)
net = tf.keras.layers.Dense(256, activation="leaky_relu")(net)
net = tf.keras.layers.Dropout(0.4)(net)
net = tf.keras.layers.Dense(64, activation="leaky_relu")(net)
net = tf.keras.layers.Dropout(0.4)(net)
preds = tf.keras.layers.Dense(1, activation="sigmoid")(net) if num_categories == 2 else tf.keras.layers.Dense(num_categories, activation="softmax")(net)
model = tf.keras.Model(base_model.input, preds)
num_new_layers = 6


# Freeze convolutional layers for initial training

for layer in model.layers[:-num_new_layers]:
    layer.trainable = False
opt = tf.keras.optimizers.Adam(
    learning_rate=1e-3
)

loss = tf.keras.losses.BinaryCrossentropy() if num_categories == 2 else tf.keras.losses.SparseCategoricalCrossentropy()

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


epochs = 40

print("After loading dataset")
print(train_dataset)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True
)

# load in latest epoch
# model.load_weights("/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/CoNNCR-R/v5/ckpt/finetuned/epoch-30.keras")


history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset,
    verbose=1,
    class_weight=None,  # look into changing this, might be good to for argon
    callbacks=[tb_callback, ckpt_callback, early_stopping],
)

history_filename = os.path.join(log_dir, "history.json")

model_save_path = "/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/CoNNCR-R_untuned.keras"
try:
    model.save(model_save_path)
except:
    model.save("CoNNCR-R_untuned.keras")


with open(history_filename, "w") as file:
    json.dump(history.history, file)



# unfreeze VGG16 layers for finetuning

for layer in model.layers[:-num_new_layers]:
    layer.trainable = True
opt = tf.keras.optimizers.Adam(
    learning_rate=1e-5
)
loss = tf.keras.losses.BinaryCrossentropy() if num_categories == 2 else tf.keras.losses.SparseCategoricalCrossentropy()

# "binary_crossentropy" if num_categories == 2 else
model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

# Setup checkpoint callback
os.makedirs(os.path.join(log_dir, "ckpt","finetuned"), exist_ok=True)
ckpt_path = os.path.join(log_dir, "ckpt","finetuned", "epoch-{epoch:02d}.keras")

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path,
    save_weights_only=False,
    save_best_only=False,
    monitor="val_loss",
)


finetuned_history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset,
    verbose=1,
    class_weight=None,  # look into changing this, might be good to
    callbacks=[tb_callback, ckpt_callback, early_stopping],
)


finetuned_history_filename = os.path.join(log_dir, "finetuned_history.json")

model_save_path = "/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/CoNNCR-R.keras"
try:
    model.save(model_save_path)
except:
    model.save("CoNNCR-R.keras")

with open(finetuned_history_filename, "w") as file:
    json.dump(finetuned_history.history, file)