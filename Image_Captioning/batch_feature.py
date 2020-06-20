from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle 
import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
from tqdm import tqdm


# This function provided with the image path, reads it, processes the image
# makes it ready for the CNN and returns the image
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # Resizing to proper INception v3 training size
    img = tf.image.resize(img, (299, 299))
    #img = tf.image.resize(img, (224,224))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    #img = tf.keras.applications.vgg16.preprocess_input(img)
    return img, image_path

# Creating an instance of prebuilt inception v3 model, defining the inputs and outputs
def image_features_model():
    image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
    #defining input and output of model
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    return image_features_extract_model

# This function takes in the image name vector at a batch size, converts them into activation
# outputs and saves them 
def batch_feature_processing(img_name_vector): 
 
    image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
    #image_model = tf.keras.applications.VGG16(include_top=False, weights = 'imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    print("summary", image_features_extract_model.summary)

    # Get unique images
    encode_train = sorted(set(img_name_vector))

    Enc_len = len(encode_train)
    np.save('encode_train.npy', encode_train)
    print("encode train", encode_train[-1], "len", Enc_len)
    
    # taking tensor slices from the tensorflow data pipeline dataset
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    # load all the images using map function
    image_dataset = image_dataset.map(
    load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    # iterating through the dataset 
    for img, path in tqdm(image_dataset):
        # extracting the batch features and resizing them
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3]))

        # Decoding the image names and using that to store the batch features
        for bf, p in zip(batch_features, path):
            # decoding the image path
            path_of_feature = p.numpy().decode("utf-8")
            print("path of feature", path_of_feature[35:])

            # saving the batch features with the corresponding image names, which is useful
            # later to load them back again
            np.save('Batch_feature'+ path_of_feature[35:], bf.numpy())

    return Enc_len