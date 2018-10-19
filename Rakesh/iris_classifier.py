#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import random

def sigmoid(z):
    ''' Calculates the Sigmoid ( Logistic ) function'''
    return 1.0/(1.0 + np.exp(-z))  

def sigmoid_prime(z):
    ''' Calculates the derivative of the Sigmoid Function '''
    return sigmoid(z)*(1-sigmoid(z))
        


# In[5]:


class Network():
    '''This Class defines the Neural Network '''
    
    def __init(self):
        ''' This constructor initializes the members , sizes is the vector of number of neurons in each layer'''
        self.layers=0
        self.sizes = []
        self.biases = [ np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        '''We randomly initialize the weights and the biases '''
    
    def set_val(self,sizes):
        self.layers=len(sizes)
        self.sizes = sizes
        self.biases = [ np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
    def feedforward(self,a):
        """ Return the output of the network if "a" is input."""
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a
    
    def SGD(self,train_data,epochs,batch_size,rate):
        '''  Train the model using stochastic gradient descent algorithm '''
        n = len(test_data)
        S = 0
        for i in range(epochs):
            random.shuffle(test_data)
            batch = [ test_data[k:k+batch_size] for k in range(0,n,batch_size)]
            for x in batch :
                self.update(x,rate)
            S = S+self.evaluate(test_data)
        S = S/epochs
        print('Avg Accuracy ',S)
                
    def update(self,batch,eta):
        ''' Update the weights and biases using backward propagation and gradient descent on mini batches '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
    
    def backprop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def evaluate(self,test_data):
        ''' Evaluates the Model by calculatin number of tests that match'''
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
        


# In[9]:


import pandas as pd

iris = pd.read_csv('iris.csv')

#Create numeric classes for species (0,1,2) 
iris.loc[iris['species']=='virginica','species']=1
iris.loc[iris['species']=='versicolor','species']=2
iris.loc[iris['species']=='setosa','species'] = 0
iris = iris[iris['species']!=2]

Y = iris.species.values
X = iris[[ 'sepal_length' , 'sepal_width' , 'petal_length' , 'petal_width']].values

test_data  = [ (x,y) for x,y in zip(X,Y)]
nnet = Network()
nnet.set_val([4,4,2,2,1])
#print(test_data)
nnet.SGD(test_data,20,10,0.5)

#nnet.evaluate(test_data)




