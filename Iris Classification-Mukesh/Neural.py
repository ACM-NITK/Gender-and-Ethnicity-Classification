from tensorflow import keras
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize


#Read the file

data=pd.read_csv("Iris.csv")
print(data["Species"].unique())
data.loc[data["Species"]=="Iris-setosa","Species"]=0
data.loc[data["Species"]=="Iris-versicolor","Species"]=1
data.loc[data["Species"]=="Iris-virginica","Species"]=2


X=data.iloc[:,1:5].values
y=data.iloc[:,5].values


data=data.iloc[np.random.permutation(len(data))]

#Training Set=60% and Test set=10% of Total values

train=int(0.6*len(data))
test=int(0.3*len(data))

x_train=X[:train]
x_test=X[10:test+10]

#Convert from values to one-hot vector

y1=[]

j=0
for i in y:
    result=[0,0,0]
    result[i]=1
    y1.append(result)

y2=np.array(y1)


y_train1=y1[:train]
y_test1=y1[10:test+10]


y_train=np.array(y_train1)
y_test=np.array(y_test1)

#Use Keras to initialise the model with architecture:
#       1 Hidden Layer : No.of units=1000       


model=keras.Sequential()

model.add(keras.layers.Dense(1000,input_dim=4,activation='sigmoid'))
model.add(keras.layers.Dense(3,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#Fit the parameters with 4 Epochs or runs
model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=10,epochs=4,verbose=1)

prediction=model.predict(x_test)
length=len(prediction)
y_label=np.argmax(y_test,axis=1)
predict_label=np.argmax(prediction,axis=1)

accuracy=np.sum(y_label==predict_label)/length * 100 
print("Accuracy of the dataset",accuracy )