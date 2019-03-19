# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import cv2
train_files=os.listdir("../input/g-and-e-9696/train_resized_1_images/train_resized_1_images/")#/G and E (96*96)/train_resized_images.tar.gz/train_resized_images/")
#print(train_files)
test_files=os.listdir("../input/g-and-e-9696/test_resized_1_images/test_resized_1_images/")
# Any results you write to the current directory are saved as output.
print(len(train_files))
#type(train_files)




#TRAIN_FILES
train_df=pd.DataFrame()
i=0
for file in train_files :
    if file.endswith('.jpg'):
        img = cv2.imread("../input/g-and-e-9696/train_resized_1_images/train_resized_1_images/"+file)
        st=file.split('_')
        img=img.reshape(1,96*96*3)
        #print(img.shape)
        #print(st)
        #st.pop()
        if( (not st[2].endswith('.jpg')) and len(st)==4 and (st[1]) and (st[2])) :
            df=pd.DataFrame(img)
            df[96*96*3]=int(st[1])*5+int(st[2])
            train_df=train_df.append(df)
            #print(train_df.shape)
        i=i+1
    #if(i==1):
    #    st=file.split('_')
    #    print(int(st[1])*5+int(st[2]))
print(i)
print(train_df.shape)
train_df.head()
train_df.to_csv('train9696.csv',index=False)



#TEST_FILES
test_df=pd.DataFrame()
i=0
for file in test_files :
    if file.endswith('.jpg'):
        img = cv2.imread("../input/g-and-e-9696/test_resized_1_images/test_resized_1_images/"+file)
        st=file.split('_')
        img=img.reshape(1,96*96*3)
        #print(img.shape)
        #print(st)
        #st.pop()
        if( (not st[2].endswith('.jpg')) and len(st)==4 and (st[1]) and (st[2])) :
            df=pd.DataFrame(img)
            df[96*96*3]=int(st[1])*5+int(st[2])
            test_df=test_df.append(df)
            #print(train_df.shape)
        i=i+1
    #if(i==1):
    #    st=file.split('_')
    #    print(int(st[1])*5+int(st[2]))
print(i)
print(test_df.shape)
test_df.head()
test_df.to_csv('test9696.csv',index=False)



#MODEL ARCHITECTURE
#MODEL ARCHITECTURE
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers, optimizers
from keras.utils import to_categorical
height=28
width=28

n_filters=16
weight_decay=1e-4
num_classes=10

model=Sequential()
model.add(Conv2D(n_filters,(5,5),kernel_regularizer=regularizers.l2(weight_decay),
                 input_shape=(96,96,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(2*n_filters,(5,5),kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(2*n_filters,(5,5),kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(4*n_filters, (5,5), kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(4*n_filters, (5,5), kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(8*n_filters, (5,5), kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(8*n_filters, (5,5), kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

opt = optimizers.SGD(lr = 0.001, momentum = 0.9, nesterov = True)
model.compile(optimizer=opt,loss='categorical_crossentropy',
             metrics=['accuracy'])

model.summary()

#FIT THE MODEL
from keras.utils import to_categorical
y_train=train_df[96*96*3]
x_train=train_df.drop(columns=[96*96*3])
x_train=np.array(x_train)
y_train=to_categorical(y_train)
x_train=x_train.reshape(x_train.shape[0],96,96,3)
model.fit(x_train,y_train,epochs=15)




#VALIDATE
model.save('weights.h5')
y_test=test_df[96*96*3]
x_test=test_df.drop(columns=[96*96*3])
y_test=to_categorical(y_test)
x_test=np.array(x_test)
x_test=x_test.reshape(x_test.shape[0],96,96,3)
preds=model.evaluate(x_test,y_test)
print('Loss: ',preds[0],' Accuracy: ',preds[1]*100,'%')

#MODEL 2
model=Sequential()
model.add(Conv2D(n_filters,(5,5),kernel_regularizer=regularizers.l2(weight_decay),
                 input_shape=(96,96,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(n_filters,(5,5),kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(n_filters,(5,5),kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(2*n_filters, (5,5), kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(2*n_filters, (5,5), kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(4*n_filters, (5,5), kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Conv2D(4*n_filters, (5,5), kernel_regularizer=regularizers.l2(weight_decay)))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

opt = optimizers.SGD(lr = 0.001, momentum = 0.9, nesterov = True)
model.compile(optimizer=opt,loss='categorical_crossentropy',
             metrics=['accuracy'])