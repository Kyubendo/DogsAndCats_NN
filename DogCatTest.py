import matplotlib.pyplot as plt
from keras import optimizers
import matplotlib.image as mpimg
from keras.layers import Input, Average
from keras import Model
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import keras

CATEGORIES = ["cat", "dog"]


model = keras.models.load_model('FinalModel_aftT-last.model')
for i in range (19):
    filepath = "D:/NN/Cats&Dogs/myTest/test{}.jpg".format(i)
    val_path = 'D:/NN/Cats&Dogs/PetImages/val'
    pre_img = mpimg.imread(filepath)

    IMG_SIZE = 86

    img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0

    

    prediction = model.predict(img)

    for i in range(40):
        print('`')

    animal = 'cat' if (prediction[0])[1] < (prediction[0])[0] else 'dog'
    print(prediction)
    print("It's a " + animal + ".")

    plt.imshow(pre_img)
    plt.show()