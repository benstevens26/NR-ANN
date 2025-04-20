import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import tensorflow as tf
import keras

# Display
from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt


from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image






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



def superimpose_path(img_path, heatmap, alpha=0.4):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

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

    # Save the superimposed image

    # Display Grad CAM
    display(superimposed_img)

def superimpose_array(img, heatmap, alpha=0.4):
    # Load the original image

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
    superimposed_img2 = keras.utils.array_to_img(superimposed_img)

    superimposed_img = keras.utils.img_to_array(superimposed_img)
    superimposed_img = superimposed_img - np.min(superimposed_img)
    superimposed_img = superimposed_img / np.max(superimposed_img) * 255
    

    # Display Grad CAM
    display(superimposed_img2)
    return superimposed_img.astype(np.uint8)


# superimpose_path(img_path, heatmap)




if __name__ == "__main__":
    img_path = r'C:\Users\Tom\Desktop\School\Imperial\Year 4\MSci Project\Report\images\old\Dogs.png'  # Change this to your image file path
    last_conv_layer_name  = 'block5_conv3'  

    # img_path = r'C:\Users\Tom\Desktop\School\Imperial\Year 4\MSci Project\NR-ANN\ANN-code\Data\im0\F\214.710keV_0.000_0.000_F_1.358cm_4632_im.npy'

    # model = VGG16(weights='imagenet')
    model = keras.saving.load_model(r"C:\Users\Tom\Desktop\School\Imperial\Year 4\MSci Project\CoNNCR-R.keras")
    img_size = (224, 224)

    event = np.load(r"C:\Users\Tom\Desktop\School\Imperial\Year 4\MSci Project\NR-ANN\ANN-code\Data\im0\F\214.710keV_0.000_0.000_F_1.358cm_4632_im.npy")
    event = np.repeat(event[:, :, np.newaxis], 3, axis=-1)
    event = tf.image.resize_with_pad(event, 224, 224)
    event = np.expand_dims(event, axis=0)
    event = event/np.max(event)*255
    
    
    img_array = preprocess_input(get_img_array(img_path, size=img_size))
    img_array = preprocess_input(event)
    # model.layers[-1].activation = None
    preds = model.predict(img_array)
    print("Predicted:", preds[0])
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    plt.matshow(heatmap)
    plt.show()
    event_sup = superimpose_array(img_array[0], heatmap)
    plt.matshow(event_sup)