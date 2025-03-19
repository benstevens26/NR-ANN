import numpy as np
import tensorflow as tf
from cnn_processing import noise_adder, smooth_operator
from tqdm import tqdm
import os, re
from skimage.filters import threshold_otsu

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


if __name__ == "__main__":
    example_dark_tensor = tf.convert_to_tensor(
        example_dark_list_unbinned, dtype=tf.float32
    )
else: # hopefully this makes it so that when i import things it won't kill the kernel
    example_dark_tensor = tf.convert_to_tensor(
        example_dark_list_unbinned[0], dtype=tf.float32
    )    

def preprocess_file_path(
    image, m_dark=m_dark_tensor, example_dark_list=example_dark_tensor
):
    image1 = noise_adder(image, m_dark=m_dark, example_dark_list=example_dark_list)
    image2 = smooth_operator(image1)
    image3 = image2.astype(np.float32)
    max_val = np.max(image3)
    if max_val > 0:
        image3 = 255 * image3 / max_val
    image3 = np.repeat(image3[:, :, np.newaxis], 3, axis=-1)
    image4 = tf.image.resize_with_pad(image3, 224, 224)
    image5 = tf.keras.applications.vgg16.preprocess_input(image4)
    image5 /= np.max(abs(image5))
    image5 = tf.expand_dims(image5, axis=0)
    return image5


def preprocess_file_path_unscaled(
    image, m_dark=m_dark_tensor, example_dark_list=example_dark_tensor, steps = [1,2,3,4,5,6]
):
    """preprocesses an image ready for CoNNCR

    Args:
        image (2D array): the image to preprocess
        m_dark (array), optional): the master dark. Defaults to m_dark_tensor.
        example_dark_list (3D array, optional): array of example darks. Defaults to example_dark_tensor.
        steps (list, optional): which steps to use 1 is add noise, 2 is smooth, 3 is threshold, 4 is scale and stack, 5 is resize and pad, 6 is VGG16 preprocess. Defaults to [1,2,3,4,5,6] (all).

    Returns:
        _type_: _description_
    """
    if 1 in steps:
        image = noise_adder(image, m_dark=m_dark, example_dark_list=example_dark_list)
    if 2 in steps:
        image = smooth_operator(image).astype(np.float32)
    if 3 in steps:
        image *= image > threshold_otsu(image)
    if 4 in steps:
        image -= np.min(image)
        image = 255 * image / np.max(image)
        image = np.repeat(image[:, :, np.newaxis], 3, axis=-1)
    if 5 in steps:
        image = tf.image.resize_with_pad(image, 224, 224)
    if 6 in steps:
        try:
            image = tf.keras.applications.vgg16.preprocess_input(image)
        except:
            print("to VGG16 preprocess, you need to use step 4 (stack to 3 channels)")
    return image


def get_file_list(seed=77, min_energy=10, return_low_energies = False):
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
            files = [
                f
                for f in files
                if (f.endswith(".npy") and os.path.join(root, f) not in errors and float(re.search(r'/([\d.]+)keV', os.path.join(root, f)).group(1)) > min_energy)
            ]
            file_list.extend([os.path.join(root, file) for file in files])

    file_list.sort()
    np.random.seed(seed)
    np.random.shuffle(file_list)
    
    if return_low_energies:
        low_file_list = []
        for base_dir in base_dirs:
            for root, dirs, files in os.walk(base_dir):
                files = [
                    f
                    for f in files
                    if (f.endswith(".npy") and os.path.join(root, f) not in errors and float(re.search(r'/([\d.]+)keV', os.path.join(root, f)).group(1)) <= min_energy)
                ]
                low_file_list.extend([os.path.join(root, file) for file in files])
        return file_list, low_file_list
    else:
        return file_list

if __name__ == "__main__":
    file_list, low_list = get_file_list(return_low_energies = True)


    for i in range(10):
        example_dark_list_unbinned = np.load(
        f"{dark_dir}/quest_std_dark_{i}.npy"
        )
        example_dark_tensor = tf.convert_to_tensor(
            example_dark_list_unbinned, dtype=tf.float32
        )
        start_idx = i * (len(file_list) // 10)
        end_idx = (i + 1) * (len(file_list) // 10)
        for file_path in tqdm(file_list[start_idx:end_idx], desc=f"Processing chunk {i+1}/{10}"):
            image = np.load(file_path)
            image = preprocess_file_path_unscaled(image, m_dark_tensor, example_dark_tensor)
            np.save(
                f"/vols/lz/twatson/ANN/final_ims/{os.path.basename(file_path)}",
                image,
            )
            
            
    file_list = low_list
    for i in range(10):
        example_dark_list_unbinned = np.load(
        f"{dark_dir}/quest_std_dark_{i}.npy"
        )
        example_dark_tensor = tf.convert_to_tensor(
            example_dark_list_unbinned, dtype=tf.float32
        )
        start_idx = i * (len(file_list) // 10)
        end_idx = (i + 1) * (len(file_list) // 10)
        for file_path in tqdm(file_list[start_idx:end_idx], desc=f"Processing LOW LIST chunk {i+1}/{10}"):
            image = np.load(file_path)
            image = preprocess_file_path_unscaled(image, m_dark_tensor, example_dark_tensor)
            np.save(
                f"/vols/lz/twatson/ANN/final_ims/{os.path.basename(file_path)}",
                image,
            )
