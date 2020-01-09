import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import time 


DATADIR = "D:/NN/Cats&Dogs/PetImages"
CATEGORIES = ["Cat", "Dog"]


IMG_SIZE = 128


training_data = []

def create_training_data(dir, arr):
    for category in CATEGORIES:
        path = os.path.join(dir, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                rs_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                arr.append([rs_array, class_num])
            except Exception as e:
                pass

create_training_data(DATADIR, training_data)
random.shuffle(training_data)
print(len(training_data))

X = []
y = []

for feature, label in training_data:
    X.append(feature)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

X = X/255.0

print(len(X))
print(len(y))


np.save('features.npy', X)
np.save('labels.npy', y)




TEST_DIR = "D:/NN/Cats&Dogs/PetImages/test"

testing_data = []
create_training_data(TEST_DIR, testing_data)

random.shuffle(testing_data)
print(len(testing_data))

X_test = []
y_test = []

for feature, label in testing_data:
    X_test.append(feature)
    y_test.append(label)

X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

X_test = X_test/255.0

print(len(X_test))
print(len(y_test))

np.save('features_test.npy', X_test)
np.save('labels_test.npy', y_test)