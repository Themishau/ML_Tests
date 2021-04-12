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

import pickle

IMG_SIZE = 125


async def create_training_data(categories, data_path):
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


async def shuffle_training_data(training_data):
    random.shuffle(training_data)
    return training_data


async def create_model(training_data_grey, training_data_rgb):
    X_grey = []
    y_grey = []
    X_rgb = []
    y_rgb = []

    for features, label in training_data_grey:
        X_grey.append(features)
        y_grey.append(label)

    for features, label in training_data_rgb:
        X_rgb.append(features)
        y_rgb.append(label)

    # print(np.array(X))
    print(np.array(X_grey).shape)
    print(np.array(X_rgb).shape)

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

async def normalize_model(input_model_grey, input_model_rgb):



    print("---------------------------")
    print("---------------------------")
    input_model_grey["x"] = await standardize(input_model_grey["x"])
    model_grey = Sequential()

    model_grey.add(Conv2D(256, (3, 3), input_shape=input_model_grey["x"].shape[1:]))
    model_grey.add(Activation('relu'))

    model_grey.add(MaxPooling2D(pool_size=(2, 2)))
    model_grey.add(Conv2D(256, (3, 3)))
    model_grey.add(Activation('relu'))

    model_grey.add(MaxPooling2D(pool_size=(2, 2)))

    model_grey.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model_grey.add(Dense(64))

    model_grey.add(Dense(1))

    model_grey.add(Activation('sigmoid'))

    ################################
    print("-------------normalize--------------")
    print(await normalize(input_model_rgb["x"]))
    input_model_rgb["x"] = await standardize(input_model_rgb["x"])
    print("---------------------------")
    print("-------------input_model_rgb--------------")
    print(input_model_rgb["x"])

    print("---------------------------")
    print("---------------------------")

    model_rgb = Sequential()

    model_rgb.add(Conv2D(256, (3, 3), input_shape=input_model_rgb["x"].shape[1:]))
    model_rgb.add(Activation('relu'))

    model_rgb.add(MaxPooling2D(pool_size=(2, 2)))
    model_rgb.add(Conv2D(256, (3, 3)))
    model_rgb.add(Activation('relu'))

    model_rgb.add(MaxPooling2D(pool_size=(2, 2)))

    model_rgb.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model_rgb.add(Dense(64))

    model_rgb.add(Dense(1))

    model_rgb.add(Activation('sigmoid'))

    model_rgb = {"model": model_grey,
                 "x": input_model_grey["x"],
                 "y": input_model_grey["y"]}

    model_grey = {"model": model_rgb,
                  "x": input_model_rgb["x"],
                  "y": input_model_rgb["y"]}

    return model_grey, model_rgb


async def train_model(input_model_grey, input_model_rgb):
    input_model_grey["model"].compile(loss='binary_crossentropy',
                                      optimizer='adam',
                                      metrics=['accuracy'])

    input_model_grey.fit(input_model_grey["x"], input_model_grey["y"], batch_size=32, epochs=3, validation_split=0.3)

    input_model_rgb["model"].compile(loss='binary_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])

    input_model_rgb.fit(input_model_rgb["x"], input_model_rgb["y"], batch_size=32, epochs=3, validation_split=0.3)


async def save_model(model, name):
    print(model["y"])

    with open("{}_x.pickle".format(name), "wb") as pickle_out:
        pickle.dump(model["x"], pickle_out)

    with open("{}_y.pickle".format(name), "wb") as pickle_out:
        pickle.dump(model["y"], pickle_out)


async def load_model(name):
    with open("{}_x.pickle".format(name), "rb") as pickle_in:
        x = pickle.load(pickle_in)

    with open("{}_y.pickle".format(name), "rb") as pickle_in:
        y = pickle.load(pickle_in)

    model = {"x": x,
             "y": y,
             }

    return model


def createplot(datax, datay, datadict, output_path):
    print("creating plot")
    fig, ax = plt.subplots(figsize=(5, 5))  # Create a figure and an axes.
    ax.plot(datax, datay, label='linear')  # Plot some data on the axes.
    # ax.plot(x, x ** 2, label='quadratic')  # Plot more data on the axes...
    # ax.plot(x, x ** 3, label='cubic')  # ... and some more.
    ax.set_xlabel('x label')  # Add an x-label to the axes.
    ax.set_ylabel('y label')  # Add a y-label to the axes.
    ax.set_title("DATA")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    plt.grid(axis='both', color='0.95')
    fig.savefig(output_path + 'dfdataPLOT.png')

    # plt.show()
    dfdata = pd.DataFrame.from_dict(datadict)
    dfdata.to_csv(output_path + 'sorted_data.csv', header=True, quotechar=' ', index=True, sep=';', mode='a',
                  encoding='utf8')
    return fig, ax
    # self.view.plt.plot([1, 2, 3, 4])
    # self.view.plt.ylabel('some numbers')
    # self.view.plt.show()


def create_img(Image):
    return
    # display(Image.fromarray(image_np))
    # plt.figure(figsize=IMAGE_SIZE)
    # plt.imshow(image_np)
    # # plt.show(output_dict)
    # # plt.show()


def analyze_data(training_data):
    return
