import csv
#import cv2
import os
import numpy as np
import matplotlib.image as mpimg

lines = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
 
images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('\\')[-1]
        current_path = 'IMG/' + filename
        #image = cv2.imread(current_path)
        image = mpimg.imread(current_path)
        image_flipped = np.fliplr(image)
        images.append(image_flipped)
        measurement = float(line[3])
        measurement_flipped = -measurement
        measurements.append(measurement_flipped)
    
X_train = np.array(images)
Y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.7))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.7))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
#model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
