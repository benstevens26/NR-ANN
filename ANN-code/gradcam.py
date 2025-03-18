import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras

# Display
from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt


# model_builder = keras.applications.xception.Xception
# img_size = (299, 299)
# preprocess_input = keras.applications.xception.preprocess_input
# decode_predictions = keras.applications.xception.decode_predictions

# last_conv_layer_name = "block14_sepconv2_act"

# # The local path to our target image
# img_path = keras.utils.get_file(
#     "cat_and_dog.jpg",
#     "https://storage.googleapis.com/petbacker/images/blog/2017/dog-and-cat-cover.jpg",
# )

# display(Image(img_path))


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()



# img_array = preprocess_input(get_img_array(img_path, size=img_size))

# # Make model
# model = model_builder(weights="imagenet")

# # Remove last layer's softmax
# model.layers[-1].activation = None

# # Print what the top predicted class is
# preds = model.predict(img_array)
# print("Predicted:", decode_predictions(preds, top=1)[0])

# # Generate class activation heatmap
# heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

# # Display heatmap
# plt.matshow(heatmap)
# plt.show()






def save_and_display_gradcam(img_array, heatmap, cam_path="cam.jpg", alpha=0.65):
    # Load the original image
    img = img_array

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    
    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # # Save the superimposed image
    superimposed_img.save(cam_path)

    # # Display Grad CAM
    display(Image(cam_path))
    return superimposed_img


# save_and_display_gradcam(img_path, heatmap)

# import cv2
# import numpy as np
# import matplotlib as mpl
# import keras.utils
# from PIL import Image

# def overlay_heatmap(img_array, heatmap, alpha=0.4):
#     """
#     Overlays a heatmap on an image.

#     Parameters:
#         img_array (np.ndarray): A 224x224x3 image (RGB). Can be in range [0,1] or [0,255].
#         heatmap (np.ndarray): A 14x14 array with values between 0 and 1.
#         alpha (float): Opacity of the heatmap overlay (default 0.4).

#     The function scales the heatmap to 0-255, applies the jet colormap, resizes it to match 
#     the image dimensions using INTER_NEAREST, and then overlays it on the image.
#     It displays the resulting image.
#     """
#     # Convert img_array to uint8 if necessary
#     if img_array.dtype != np.uint8:
#         # If the maximum value is <= 1, assume the image is in [0,1] range
#         if img_array.max() <= 1:
#             img_array = np.uint8(img_array * 255)
#         else:
#             img_array = np.uint8(img_array)
    
#     # Scale heatmap from 0-1 to 0-255
#     heatmap_scaled = np.uint8(heatmap * 255)
    
#     # Apply the jet colormap (OpenCV uses BGR by default)
#     heatmap_color = cv2.applyColorMap(heatmap_scaled, cv2.COLORMAP_JET)
    
#     # Resize the heatmap to match the image dimensions using INTER_NEAREST to reduce blurring
#     heatmap_color = cv2.resize(heatmap_color, (img_array.shape[1], img_array.shape[0]), interpolation=cv2.INTER_NEAREST)
    
#     # Convert heatmap from BGR to RGB for correct color display with matplotlib
#     heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
#     # Superimpose the heatmap on the original image using weighted addition
#     superimposed_img = cv2.addWeighted(img_array, 1 - alpha, heatmap_color, alpha, 0)
    
#     # Display the final superimposed image
#     plt.imshow(superimposed_img)
#     plt.axis('off')
#     plt.show()