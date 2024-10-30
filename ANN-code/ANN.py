import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from analysis import *
from convert_sim_ims import *
from feature_extraction import *
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# folder_path = "../../../../MIGDAL/sim_ims/"  # on the ssh
folder_path = "Data/C/300-320keV"  # on local
C_events = load_events(folder_path)
F_events = load_events("Data/F/260-280keV")
events = C_events + F_events

features = [[event.get_energy_from_name()] for event in events]
labels = [event.get_species_from_name() for event in events]

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

classifier = MLPClassifier(random_state=42)
classifier.fit(features_train, labels_train)
labels_pred = classifier.predict(features_test)

# Calculate accuracy
accuracy = accuracy_score(labels_test, labels_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print confusion matrix
conf_matrix = confusion_matrix(labels_test, labels_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Print classification report
class_report = classification_report(labels_test, labels_pred)
print("Classification Report:")
print(class_report)
