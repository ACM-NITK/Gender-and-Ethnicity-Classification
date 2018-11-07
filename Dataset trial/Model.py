import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import pickle


sns.set(style='white', context='notebook', palette='deep')
# Load the data
train = pd.read_csv("FINAL.csv")
#print(train.head())
Y_train = train["784"]
X_train = train.drop(labels = ["784"],axis = 1) 
 
X_train=X_train.values.reshape(-1,28,28,1)

X_train,X_val,Y_train,Y_val=train_test_split(X_train,Y_train,test_size=0.05,random_state=3)
print(len(X_train))
print(len(X_val))
Y_train=tf.keras.utils.to_categorical(Y_train,10)
Y_val=tf.keras.utils.to_categorical(Y_val,10)

filter_input=(5,5)
filter_hidden=(3,3)
data_format='channels_last'
pool=(2,2)


model=keras.Sequential()

#model = Sequential()
model.add(keras.layers.Conv2D(64, (3,3), activation='relu',input_shape = (28,28,1),name='block1_conv1'))
model.add(keras.layers.Conv2D(64, (5,5), activation='relu', name='block1_conv2'))
model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))


model.add(tf.layers.Flatten())
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dropout(0.15))
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()
datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)
model.load_weights("Trained_model.h5")
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=50),
                              epochs = 1, validation_data = (X_val,Y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0])

model.save('Trained_model1.h5')