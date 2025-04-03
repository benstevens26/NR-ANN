import os
import sys

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
# tf.config.run_functions_eagerly(True)

from preprocess import get_file_list
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16  # type: ignore
from tensorflow.keras.layers import *  # type: ignore
from keras import regularizers
import json

finetune = True
exclude_low_e = True
CoNNCR_version = 15
argon = False
weight_class = True


def load_and_process(file_path):
    # file_path here is a tf.string tensor, so convert to numpy if needed
    file_path = file_path.numpy().decode("utf-8")
    image = np.load(file_path)
    label = 0 if "C" in os.path.basename(file_path) else 1 if "F" in os.path.basename(file_path) else 2
    return image, label

def tf_load_and_process(file_path):
    # Wrap the python function
    image, label = tf.py_function(func=load_and_process, inp=[file_path], Tout=[tf.float32, tf.int32])
    image.set_shape((224, 224, 3))
    label.set_shape(())
    return image, label





base_dirs = ["/vols/lz/twatson/ANN/final_ims_Ar"] if argon else ["/vols/lz/twatson/ANN/old_final_ims"]
batch_size = 16
binning = 1



file_list, low_list = get_file_list(base_dirs,return_low_energies=True, argon=argon) if exclude_low_e else get_file_list(base_dirs,min_energy=0,return_low_energies=True, argon=argon)
file_list = sorted(file_list) # only shuffle ONCE so that sets are easily reproducable
np.random.seed(77) 
np.random.shuffle(file_list)
print(f"FIRST AND LAST ELEMENTS OF FILE LIST: {file_list[0], file_list[-1]}")




dataset_size = len(file_list) # i might be stupid lmao
train_size = (int(0.7 * dataset_size)//batch_size)*batch_size
val_size = (int(0.15 * dataset_size)//batch_size)*batch_size
test_size = ((dataset_size - train_size - val_size)//batch_size)*batch_size  # Ensure all data is used

train_list = file_list[:train_size]
val_list = file_list[train_size:train_size + val_size]
test_list = file_list[train_size + val_size : train_size + val_size + test_size]
print(f"FIRST AND LAST ELEMENTS OF TEST SET: {test_list[0], test_list[-1]}")
print(f"TEST SET SIZE: {len(test_list)}")



if weight_class:
    CF_ratio = 7.33
    num_F = int(sum(1 for row in val_list if "F" in os.path.basename(row)))
    num_C = int(num_F//CF_ratio)

    C_val = [row for row in val_list if "C" in os.path.basename(row)]
    F_val = [row for row in val_list if "F" in os.path.basename(row)]
    
    if len(C_val) > num_C:
        # Randomly shuffle and select only num_C elements
        np.random.seed(77)
        np.random.shuffle(C_val)
        C_val = C_val[:num_C]
    
    val_list = C_val + F_val
    np.random.shuffle(val_list)





# First 70% for training
train_dataset = tf.data.Dataset.from_tensor_slices(train_list)
train_dataset = train_dataset.map(tf_load_and_process, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# Next 15% for val
val_dataset = tf.data.Dataset.from_tensor_slices(val_list)
val_dataset = val_dataset.map(tf_load_and_process, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE) # Next 15%





# Next 15 % for test
test_dataset = tf.data.Dataset.from_tensor_slices(test_list)
test_dataset = test_dataset.map(tf_load_and_process, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE) # Final 15%



low_size = len(low_list)
low_test_size = (int(0.15 * low_size)//batch_size)*batch_size
low_test_list = low_list[:low_test_size] + test_list
low_test_dataset = tf.data.Dataset.from_tensor_slices(low_test_list)
low_test_dataset = low_test_dataset.map(tf_load_and_process, num_parallel_calls=tf.data.AUTOTUNE)
low_test_dataset = low_test_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE) # Final 15%



num_categories = 3 if argon else 2


# Define model
inputs = keras.Input(shape=(224, 224, 3))
base_model = VGG16(weights="imagenet", include_top=False, input_tensor=inputs)
net = base_model.output
net = tf.keras.layers.Flatten()(net)
net = tf.keras.layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(net)
net = tf.keras.layers.BatchNormalization()(net)
net = tf.keras.layers.Activation("leaky_relu")(net)
net = tf.keras.layers.Dropout(0.3)(net)
net = tf.keras.layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4))(net)
net = tf.keras.layers.BatchNormalization()(net)
net = tf.keras.layers.Activation("leaky_relu")(net)
net = tf.keras.layers.Dropout(0.3)(net)
preds = tf.keras.layers.Dense(2, activation="softmax")(net) if num_categories == 2 else tf.keras.layers.Dense(num_categories, activation="softmax")(net)
model = tf.keras.Model(base_model.input, preds)
num_new_layers = 9


# Freeze convolutional layers for initial training

for layer in model.layers[:-num_new_layers]:
    layer.trainable = False
opt = tf.keras.optimizers.Adam(
    learning_rate=1e-3
)

# loss = tf.keras.losses.BinaryCrossentropy() if num_categories == 2 else tf.keras.losses.SparseCategoricalCrossentropy()
loss = tf.keras.losses.SparseCategoricalCrossentropy()

# "binary_crossentropy" if num_categories == 2 else
model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

# Setup TensorBoard callback
log_dir = f"/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/CoNNCR-R/v{CoNNCR_version}"

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
    class_weight= { 0 : 1 , 1 : 4.04 } if weight_class else None,  # look into changing this, might be good to for argon
    callbacks=[tb_callback, ckpt_callback, early_stopping],
)

history_filename = os.path.join(log_dir, "history.json")

model_save_path = f"/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/CoNNCR-R/v{CoNNCR_version}/CoNNCR-R_untuned.keras"
try:
    model.save(model_save_path)
except:
    model.save("CoNNCR-R_untuned.keras")


with open(history_filename, "w") as file:
    json.dump(history.history, file)



# unfreeze VGG16 layers for finetuning

for layer in model.layers[-(num_new_layers + 3):]:
    layer.trainable = True
opt = tf.keras.optimizers.Adam(
    learning_rate=1e-6
)
# loss = tf.keras.losses.BinaryCrossentropy() if num_categories == 2 else tf.keras.losses.SparseCategoricalCrossentropy()
loss = tf.keras.losses.SparseCategoricalCrossentropy()

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

epochs = 30

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True
)

finetuned_history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset,
    verbose=1,
    class_weight={ 0 : 1 , 1 : 4.04 } if weight_class else None,  # look into changing this, might be good to
    callbacks=[tb_callback, ckpt_callback, early_stopping],
)


mid_tuning_history_filename = os.path.join(log_dir, "mid_tuning_history.json")

model_save_path = f"/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/CoNNCR-R/v{CoNNCR_version}/CoNNCR-R_mid_tuning.keras"
try:
    model.save(model_save_path)
except:
    model.save("CoNNCR-R_mid_tuning.keras")

for layer in model.layers[-(num_new_layers + 4):]:
    layer.trainable = True
opt = tf.keras.optimizers.Adam(
    learning_rate=5e-7
)
model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True
)

ckpt_path = os.path.join(log_dir, "ckpt","finetuned", "epoch-{epoch:02d}_2.keras")

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path,
    save_weights_only=False,
    save_best_only=False,
    monitor="val_loss",
)


finetuned_history_2 = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset,
    verbose=1,
    class_weight={ 0 : 1 , 1 : 4.04 } if weight_class else None,  # look into changing this, might be good to
    callbacks=[tb_callback, ckpt_callback, early_stopping],
)

finetuned_history_filename = os.path.join(log_dir, "finetuned_history.json")

model_save_path = f"/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/CoNNCR-R/v{CoNNCR_version}/CoNNCR-R.keras"
try:
    model.save(model_save_path)
except:
    model.save("CoNNCR-R.keras")

with open(finetuned_history_filename, "w") as file:
    json.dump(finetuned_history.history, file)


finetuned_history_2_filename = os.path.join(log_dir, "finetuned_history_2.json")

with open(finetuned_history_2_filename, "w") as file:
    json.dump(finetuned_history_2.history, file)

print("Predicting...")
import csv
predictions_file_path = f"/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/CoNNCR-R/v{CoNNCR_version}/CoNNCR-Rv{CoNNCR_version}_predictions.csv"
# batch_size = 32  # Adjust as needed based on available memory
batched_paths = []
with open(predictions_file_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["file_path", "prediction"])  # Write header

    for file_index in range(0, len(low_test_list), batch_size):
        # Get batch file paths
        batched_paths = low_test_list[file_index:file_index + batch_size]

        # Load images for the current batch
        batched_images = [np.load(path) for path in batched_paths]  # Shape: (batch_size, 224, 224, 3)
        batched_images = np.stack(batched_images, axis=0)  # Convert to numpy array

        # Run batch prediction
        predictions = model.predict(batched_images, verbose=0)  # Shape: (batch_size, 2)

        # Write each file's path with its respective prediction
        for path, pred in zip(batched_paths, predictions):
            writer.writerow([path, list(pred)])