# -*- coding: utf-8 -*-
import os
import cv2
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.figure import Figure
from IPython import get_ipython
from IPython.display import display
import pickle
IMG_SIZE = 125

async def create_training_data(categories, data_path):
    training_data_grey = []
    for category in categories:
        path = os.path.join(data_path, category)
        class_index = categories.index(category) # careful with the categoriy ( ex. cat first, dog second)
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
        class_index = categories.index(category) # careful with the categoriy ( ex. cat first, dog second)
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

async def create_model(training_data):
    x = []
    y = []

    for features, label in training_data:
        x.append(features)
        y.append(label)

    # x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    print(np.array(x).reshape(IMG_SIZE, IMG_SIZE))

    model = { "x" : x,
              "y" : y
            }
    return model
async def save_model(model):

    print(model["y"])

    with open("x.pickle", "wb") as pickle_out:
        pickle.dump(model["x"], pickle_out)

    with open("y.pickle", "wb") as pickle_out:
        pickle.dump(model["y"], pickle_out)


async def load_model():

    with open("x.pickle", "rb") as pickle_in:
        x = pickle.load(pickle_in)

    with open("y.pickle", "rb") as pickle_in:
        y = pickle.load(pickle_in)

    model = { "x" : x,
              "y" : y,
            }

    print(model["y"])

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

    #plt.show()
    dfdata = pd.DataFrame.from_dict(datadict)
    dfdata.to_csv(output_path + 'sorted_data.csv', header=True, quotechar=' ', index=True, sep=';', mode='a', encoding='utf8')
    return fig, ax
    #self.view.plt.plot([1, 2, 3, 4])
    #self.view.plt.ylabel('some numbers')
    #self.view.plt.show()

def create_img(Image):
    return
    # display(Image.fromarray(image_np))
    # plt.figure(figsize=IMAGE_SIZE)
    # plt.imshow(image_np)
    # # plt.show(output_dict)
    # # plt.show()

def analyze_data (training_data):
    return