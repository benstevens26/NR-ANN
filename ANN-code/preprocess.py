import numpy as np
import tensorflow as tf
from cnn_processing import noise_adder, smooth_operator
from tqdm import tqdm
import os

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
batch_size = 16
dark_list_number = 0
binning = 1
dark_dir = "/vols/lz/MIGDAL/sim_ims/darks"
# dark_dir="Data/darks"
m_dark = np.load(f"{dark_dir}/master_dark_{str(binning)}x{str(binning)}.npy")
example_dark_list_unbinned = np.load(
    f"{dark_dir}/quest_std_dark_{dark_list_number}.npy"
)
m_dark_tensor = tf.convert_to_tensor(m_dark, dtype=tf.float32)
example_dark_tensor = tf.convert_to_tensor(example_dark_list_unbinned[:1], dtype=tf.float32)



def preprocess_file_path(image,m_dark=m_dark_tensor,example_dark_list=example_dark_tensor):
    image1 = noise_adder(image, m_dark=m_dark, example_dark_list=example_dark_list)
    image2 = smooth_operator(image1)
    image3 = image2.astype(np.float32)
    max_val = np.max(image3)
    if max_val > 0:
        image3 = 255*image3 / max_val
    image3 = np.repeat(image3[:, :, np.newaxis], 3, axis=-1)
    image4 = tf.image.resize_with_pad(image3,224,224)
    image5 = tf.keras.applications.vgg16.preprocess_input(image4)
    image5/=np.max(abs(image5))
    image5 = tf.expand_dims(image5, axis=0)
    return image5

def get_file_list(seed=77):
    uncropped_error = np.loadtxt(
    "/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/uncropped_error.csv",
    delimiter=",",
    dtype=str,
    )
    min_dim_error = np.loadtxt(
        "/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/min_dim_error.csv",
        delimiter=",",
        dtype=str,
    )
    # Get all the .npy files from base_dirs
    errors = np.concatenate((uncropped_error, min_dim_error))

    # Get all the .npy files from base_dirs
    file_list = []
    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            files = [f for f in files if (f.endswith(".npy") and os.path.join(root, f) not in errors)]
            file_list.extend([os.path.join(root, file) for file in files])

    file_list.sort()
    np.random.seed(seed)
    np.random.shuffle(file_list)
    return file_list

file_list = get_file_list()


for file_path in tqdm(file_list):
    image = np.load(file_path)
    image = preprocess_file_path(image, m_dark_tensor, example_dark_tensor)[0]
    np.save(f"/vols/lz/twatson/ANN/preprocessed_images/{os.path.basename(file_path)}",image)