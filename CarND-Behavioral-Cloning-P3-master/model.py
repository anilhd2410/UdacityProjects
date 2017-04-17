import csv
import os
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import sklearn
from random import shuffle

#batch size
BATCH_SIZE = 64
#Number of Epoch
EPOCH = 15

#Function : datagenerator
# Parameters : samples,rows from cvs file ; batch_size, size of the batch
def datagenerator(samples, batch_size):
    num_samples = len(samples)
    correction = 0.15 # this is a parameter to tune for steering difference between center and others
    path = 'IMG/' # fpath to your training IMG directory
    while 1: # Loop forever so the generator never terminates
        shuffle(samples) # Shuffle Samples
        for offset in range(0, num_samples, batch_size):
            #get the batch from the sample
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            # get the data for the batch
            for batch_sample in batch_samples:
                # read in images from center, left and right cameras
                for i in range(3):    
                    source_path = batch_sample[i]
                    filename = source_path.split('\\')[-1]
                    img =np.asarray(Image.open(path + filename))
                    images.append(img)
                 
                #read steering angles
                steering_center = float(batch_sample[3])
                angles.append(steering_center)
                # create adjusted steering measurements for the side camera images
                steering_left = steering_center + correction
                angles.append(steering_left)
                steering_right = steering_center - correction
                angles.append(steering_right)
         

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


#Read csv file and store rows into list
lines = []            
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        lines.append(row)

#split that data set into training and validation set 80-20%
from sklearn.model_selection  import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

#compile and train the model using the generator function
train_generator = datagenerator(train_samples, batch_size=BATCH_SIZE)
validation_generator = datagenerator(validation_samples,batch_size=BATCH_SIZE)

 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

#Network consists of 10 layers, including a normalization  amd s
# a Cropping layer, 5 convolutional layers, and 3 fully connected layers
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# cropping the images, only keep road view
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))

model.add(Convolution2D(24,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation='relu'))
# Dropout to reduce overfitting 
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
# Dropout to reduce overfitting 
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
#model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

#compile the model with odam optimizer, learning rate tuning not needed 
model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator, samples_per_epoch= 
            len(train_samples)*3, validation_data=validation_generator, 
            nb_val_samples=len(validation_samples)*3,nb_epoch=EPOCH)

#Save the model
model.save('model.h5')

# plot the history for loss
#import matplotlib.pyplot as plt

#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('EPOCH')
#plt.legend(['training set', 'validation set'], loc='upper left')
#plt.show()
