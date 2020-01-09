from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, LeakyReLU, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import time 


train_path = 'D:/NN/Cats&Dogs/PetImages'
val_path = 'D:/NN/Cats&Dogs/PetImages/val'

train_batches = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   rescale=1./255,
                                   zoom_range = 0.15,
                                   shear_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest').flow_from_directory(train_path,
                                                                        target_size=(86,86), color_mode="grayscale",
                                                                        classes=['Cat', 'Dog'],
                                                                        batch_size=16)

val_batches = ImageDataGenerator(rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 rescale=1./255,
                                 zoom_range = 0.15,
                                 shear_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest').flow_from_directory(val_path,
                                                                        target_size=(86,86), color_mode="grayscale",
                                                                        classes=['Cat', 'Dog'],
                                                                        batch_size=16)



NAME = "3conv"
tensorboard = TensorBoard(log_dir="D:\\NN\\Cats&Dogs\\logs\\{}".format(NAME))
print(NAME)
input_shape = (86, 86, 1)
model = Sequential()

model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=input_shape, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(2))
model.add(Activation('sigmoid'))


learning_rate = 0.001
decay_rate = learning_rate / 30

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=learning_rate, decay=decay_rate),
              metrics=['accuracy'])

checkpoint = ModelCheckpoint('FinalModel.model', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tensorboard = TensorBoard(log_dir="D:\\NN\\Cats&Dogs\\logs\\Final")
callbacks_list = [checkpoint, tensorboard]

model.fit_generator(train_batches, steps_per_epoch=1374,
                    validation_data=val_batches, validation_steps=187,
                    epochs=50,
                    callbacks=callbacks_list,
                    max_queue_size=1)

#model.save('FinalModel.model')