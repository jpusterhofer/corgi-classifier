from dataset import *

#Suppress all the tensorflow warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#Read in training and testing data
img_size = 50
train_data = read_train_sets("./data/training_data", img_size, ['pembroke', 'cardigan'], 0)
test_data = read_train_sets("./data/testing_data", img_size, ['pembroke', 'cardigan'], 0)

#Add layers to CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

#Train model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data.train.images(), train_data.train.labels(), epochs=15, verbose=2, validation_split=0.1)

#Test model
model.evaluate(test_data.train.images(), test_data.train.labels(), verbose=1)

