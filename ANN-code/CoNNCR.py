"""
CoNNCR

Convolutional
Neural
Network
(for)
Classification
(of)
Recoils

"""

#%%

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

#%%

# Import data for training

# X is list of image arrays
# y are 0 or 1 (carbon or fluorine)


#%%

# Split into 70% train, 15% validation, 15% test

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

#%%

# Define CoNNCR

input_shape = (224, 224, 1)  # Example shape, you can change it based on your data

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#%%

# Assuming X_train, y_train, X_val, y_val are preprocessed and ready
# where X_train and X_val are the image datasets and y_train, y_val are the labels

# Training the model
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=20,
                    validation_data=(X_val, y_val),
                    verbose=1)

# Evaluating the model
test_loss, test_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")


