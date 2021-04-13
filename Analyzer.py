# -*- coding: utf-8 -*-
import os
import cv2
import random
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# np.set_printoptions(threshold=sys.maxsize)
from sklearn.decomposition import PCA
from matplotlib.figure import Figure
from IPython import get_ipython
from IPython.display import display
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model

import pickle
# tensorboard --logdir=logs/
NAME = "CNN_FHDW"
IMG_SIZE = 125
_globalgpu = None


def set_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def init_gpu():
    # set GPU memory which can be used for object detection
    config = tf.compat.v1.ConfigProto(gpu_options=
                                      tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)
                                      # device_count = {'GPU': 1}
                                      )
    # config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    globalgpu = tf.compat.v1.Session(config=config)
    return globalgpu


async def shuffle_training_data(training_data):
    random.shuffle(training_data)
    return training_data


async def standardize(model):
    # standard score, z-tranformation
    # Z = (X - E(X)) / Standardabweichung // https://en.wikipedia.org/wiki/Standard_score
    mean = np.mean(model, axis=0)
    std = np.std(model, axis=0)
    model_trans = model - mean # -= does not work because they need to be the same type
    model_trans = model_trans / std
    # you want to save your mean and std for more data
    return model_trans


async def normalize(model):
    # normalization
    max = np.amax(model, axis=0)
    min = np.amin(model, axis=0)

    model_trans = (model - min) / (max - min)
    return model_trans


async def create_training_data_grey(categories, data_path):
    training_data_grey = []
    for category in categories:
        path = os.path.join(data_path, category)
        class_index = categories.index(category)  # careful with the categoriy ( ex. cat first, dog second)
        for img in os.listdir(path):
            try:
                # convert it to greyscale, so we have a less complex data set
                # 2d array is easier to work with in this particular case
                img_array_grey = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                # resize and normalize the image
                new_array_grey = cv2.resize(img_array_grey, (IMG_SIZE, IMG_SIZE))
                training_data_grey.append([new_array_grey, class_index])
            except Exception as e:
                # print (e)
                pass

    print("taining_data: " + str(len(training_data_grey)))
    return training_data_grey


async def create_training_data_rgb(categories, data_path):
    training_data_rbg = []
    for category in categories:
        path = os.path.join(data_path, category)
        class_index = categories.index(category)  # careful with the categoriy ( ex. cat first, dog second)
        for img in os.listdir(path):
            try:
                # convert it to greyscale, so we have a less complex data set
                # 2d array is easier to work with in this particular case
                img_array_rbg = cv2.imread(os.path.join(path, img))
                # resize and normalize the image
                new_array_rgb = cv2.resize(img_array_rbg, (IMG_SIZE, IMG_SIZE))
                training_data_rbg.append([new_array_rgb, class_index])
            except Exception as e:
                # print (e)
                pass

    print("taining_data: " + str(len(training_data_rbg)))
    return training_data_rbg


async def prepare_training_data_np(training_data_grey, training_data_rgb):
    X_grey = []
    y_grey = []
    X_rgb = []
    y_rgb = []

    for features, label in training_data_grey:
        X_grey.append(features)
        y_grey.append(label)
    y_grey = np.array(y_grey)

    for features, label in training_data_rgb:
        X_rgb.append(features)
        y_rgb.append(label)
    y_rgb = np.array(y_rgb) # we need a numpy array

    # print(np.array(X))


    # Reduktion der "Dimensionen" eines Arrays für die
    # (24946, 125, 125)
    # 24946 Bilder, die 125 x 125 groß sind. (mit einem color channel S/W)
    X_train_grey = np.array(X_grey)

    X_train_grey = np.reshape(X_train_grey, (-1, X_train_grey.shape[1], X_train_grey.shape[2], 1))
    # Reduktion der "Dimensionen" eines Arrays für die
    # (24946, 125, 125)
    # 24946 Bilder, die 125 x 125 groß sind. (mit 3 color channel rgb)
    X_train_rgb = np.array(X_rgb)
    X_train_rgb = np.reshape(X_train_rgb, (-1, X_train_rgb.shape[1], X_train_rgb.shape[2], 3))
    # print(X_train.reshape(X_train, (X_train.shape[1], X_train.shape[2], -1)))
    print(X_train_grey.shape)
    print(X_train_rgb.shape)

    # x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # print(np.array(x).reshape(-1, shape=(IMG_SIZE, IMG_SIZE), 1))

    model_grey = {"x": X_train_grey,
                  "y": y_grey}

    model_rgb = {"x": X_train_rgb,
                 "y": y_rgb}

    return model_grey, model_rgb


async def normalize_model_grey(input_data_grey):

    print("---------------------------")
    print("---------------------------")
    input_data_grey["x"] = await standardize(input_data_grey["x"])
    model_grey = Sequential()
    # model_grey_x = tf.random.normal(input_data_grey["x"])
    # model_grey_y = tf.keras.layers.Conv2D(2, 3, activation='relu', input_shape=input_data_grey["x"][1:])(input_data_grey["x"])
    model_grey.add(Conv2D(256, (3, 3), input_shape=input_data_grey["x"].shape[1:]))
    model_grey.add(Activation('relu'))
    model_grey.add(MaxPooling2D(pool_size=(2, 2)))

    model_grey.add(Conv2D(256, (3, 3)))
    model_grey.add(Activation('relu'))
    model_grey.add(MaxPooling2D(pool_size=(2, 2)))

    model_grey.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model_grey.add(Dense(64))
    model_grey.add(Activation('relu'))

    model_grey.add(Dense(1))
    model_grey.add(Activation('sigmoid'))

    data = {"x": input_data_grey["x"],
            "y": input_data_grey["y"]}


    return data, model_grey


async def normalize_model_rgb(input_data_rgb):
    ################################
    # print("-------------normalize--------------")
    # print(await normalize(input_data_rgb["x"]))
    input_data_rgb["x"] = await standardize(input_data_rgb["x"])
    # print("---------------------------")
    # print("-------------input_data_rgb--------------")
    # print(input_data_rgb["x"])

    print("---------------------------")
    print("---------------------------")
    print("inputshape: {} \n "
          "shape.[1:]: {}".format(input_data_rgb["x"].shape, input_data_rgb["x"].shape[1:]))
    print("---------------------------")
    print("---------------------------")

    model_rgb = Sequential()
    model_rgb.add(Conv2D(64, (3, 3), input_shape=input_data_rgb["x"].shape[1:]))
    model_rgb.add(Activation('relu'))
    model_rgb.add(MaxPooling2D(pool_size=(2, 2)))

    model_rgb.add(Conv2D(64, (2, 3)))
    model_rgb.add(Activation('relu'))
    model_rgb.add(MaxPooling2D(pool_size=(2, 2)))

    model_rgb.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model_rgb.add(Dense(64))
    model_rgb.add(Activation('relu'))

    model_rgb.add(Dense(1))
    model_rgb.add(Activation('sigmoid'))

    data = {"x": input_data_rgb["x"],
            "y": input_data_rgb["y"]}

    return data, model_rgb


async def train_model_grey(input_data_grey, model):
    print("input_data_grey[x]: {} \n".format(input_data_grey["x"].shape))
    tensorboard_grey = TensorBoard(log_dir="logs/{}".format(NAME + "_grey"))

    model.compile(loss='binary_crossentropy',
                                      optimizer='adam',
                                      metrics=['accuracy'])

    model.fit(input_data_grey["x"],
                   input_data_grey["y"],
                   batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard_grey])
    print("grey done")
    return model


async def train_model_rgb(input_data_rgb, model):
    print("input_data_rgb[x]:  {} \n".format(input_data_rgb["x"].shape))
    tensorboard_rgb = TensorBoard(log_dir="logs/{}".format(NAME + "_rgb"))

    model.compile(loss='binary_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])
    model.fit(input_data_rgb["x"],
                  input_data_rgb["y"],
                  batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard_rgb])
    print("rgb done")

    return model


#### save and load methods ####

async def save_model_training_data(model, name):
    with open("{}_training_x.pickle".format(name), "wb") as pickle_out:
        pickle.dump(model["x"], pickle_out)

    with open("{}_training_y.pickle".format(name), "wb") as pickle_out:
        pickle.dump(model["y"], pickle_out)


async def save_normalized_model(model, data, name):
    with open("{}_normalized_x.pickle".format(name), "wb") as pickle_out:
        pickle.dump(data["x"], pickle_out)
    with open("{}_normalized_y.pickle".format(name), "wb") as pickle_out:
        pickle.dump(data["y"], pickle_out)
    model.save('{}_normalized_model.h5'.format(name))


async def save_trained_model(model, name):
    model.save('{}_trained_model.h5'.format(name))


async def load_model_training_data(name):
    with open("{}_training_x.pickle".format(name), "rb") as pickle_in:
        x = pickle.load(pickle_in)

    with open("{}_training_y.pickle".format(name), "rb") as pickle_in:
        y = pickle.load(pickle_in)

    model = {"x": x,
             "y": y,
             }
    return model


async def load_normalized_model(name):
    with open("{}_normalized_x.pickle".format(name), "rb") as pickle_in:
        x = pickle.load(pickle_in)

    with open("{}_normalized_y.pickle".format(name), "rb") as pickle_in:
        y = pickle.load(pickle_in)

    model = load_model('{}_normalized_model.h5'.format(name))

    data = {
             "x": x,
             "y": y,
             }

    return model, data


async def load_trained_model(name):
    with open("{}_normalized_x.pickle".format(name), "rb") as pickle_in:
        x = pickle.load(pickle_in)

    with open("{}_normalized_y.pickle".format(name), "rb") as pickle_in:
        y = pickle.load(pickle_in)
    model = load_model('{}_trained_model.h5'.format(name))
    data = {
             "x": x,
             "y": y,
             }

    return model, data




