import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import tensorflow as tf


save = False
ROC_curve = True
confusion_matrix = True
# which = "L"
which = "C"
which = "both"

if which == "L":
    model = tf.keras.models.load_model(
    "/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/LENRIv1.keras",
    ) # load LENRI
elif which == "C":
    model = tf.keras.models.load_model(
    "/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/CoNNCR-R/v2/CoNNCR-R.keras",
    custom_objects={"softmax_v2": tf.keras.activations.softmax}
    ) # load CoNNCR
elif which == "both":
    models = [tf.keras.models.load_model(
    "/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/LENRIv1.keras"
    ),
    tf.keras.models.load_model(
    "/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/CoNNCR-R/v2/CoNNCR-R.keras",
    custom_objects={"softmax_v2": tf.keras.activations.softmax}
    ) 
    ]


plt.style.use('seaborn-v0_8-whitegrid')
