#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:50:30 2019

@author: abhigyan
"""

"""import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim"""
import numpy as np
import matplotlib.pyplot as plt


# Load Dataset to a NumPy array
# The dataset must be in the following format:
# 1. The data should be sperated by a ',' if it contains multiple
#    columns and the starting line number should be mentioned
# 2. The [start:stop] should be mentioned for all the columns
#    for each line
# 3. Each line should have a single data entry.
# 4. All numbers should be provided as per Python indexing conditions

"""Function to load data into NumPy array, takes in [start:stop], 
   starting line number, and prediction varibale column number and 
   returns all the input variables and the output variable as NumPy arrays"""

def _load_data(path, startLine, start, stop, prediction):
    f = open(path, 'r')
    data = f.readlines()
    data = data[startLine:]
    inp = []
    dataNp = np.array([])
    for i in data:
        a = i.split(',')[start:stop]
        for j in a:
            dataNp = np.append(dataNp, float(j))
    dataNp = np.reshape(dataNp, (-1, stop-start))
    for i in range(stop-start):
        inp.append(dataNp[:, i])
    op = inp[prediction]
    
    return inp,op
        

"""A class describing an Echo State Network that can be trained on a single dataset
   and will contain the weights to the training based on the data provided. Thus we 
   can create several instances of ESN each trained on different data. The ESN accepts
   only a single variable data and generates a single variable data as a result."""
class ESN(object):
    
    def __init__(self, leakRate, inSize, outSize, reservoirSize, spectralRadius = 1.25):
        self.leakRate = leakRate
        self.inSize = inSize
        self.outSize = outSize
        self.reservoirSize = reservoirSize
        self.spectralRadius = spectralRadius
        self.initialize_weights()
        
        
    def initialize_weights(self):
        np.random.seed(42)
        self.Win = np.random.random((self.reservoirSize, 1+self.inSize))
        self.W = np.random.random((self.reservoirSize, self.reservoirSize))
        # Normalizing and setting spectral radius:
        print('Computing spectral radius....'),
        rhoW = max(abs(np.linalg.eig(self.W)[0]))
        print('Done !!!')
        self.W *= self.spectralRadius / rhoW
        
        
    def train(self, regCoeff, trainSet, initLen, target, method = 'ridge'):
        if(method == 'ridge'):
            x = self.ridge_regression(trainSet, regCoeff, initLen, target)
            return x
        elif(method == 'ml'):
            self.ml_regression(trainSet, regCoeff, initLen, target)
            
            
    def ridge_regression(self, trainSet, regCoeff, initLen, target):
        # Memory allocation for collected states
        trainLen = len(trainSet) - 1
        X = np.zeros((1+self.inSize+self.reservoirSize, trainLen-initLen))
        # Set the corresponding target matrix directly
        Yt = target[None,initLen+1:]
        # Run the reservoir with the data and collect all the x
        x = np.zeros((self.reservoirSize,1))
        for t in range(trainLen):
            u = trainSet[t]
            x = (1-self.leakRate)*x + self.leakRate*np.tanh(np.dot(self.Win, np.vstack((1,u))) \
                 + np.dot(self.W, x))
            if t >= initLen:
                X[:,t-initLen] = np.vstack((1,u,x))[:,0]
                
        # Train the output by ridge regression
        
        X_T = X.T
        self.Wout = np.dot(np.dot(Yt,X_T), np.linalg.inv(np.dot(X,X_T) + \
                             regCoeff*np.eye(1+self.inSize+self.reservoirSize) ) )
        print('W_out has been learned by Ridge Regression method !!!')
        return x
    
    
    # First we run some of the data to initialize the echo chamber, then we run the 
    # rest of the test set. This should be used for cross validation
    def test(self, testSet, initLen, labels, initialize = False, initializer = None, 
             lossType = 'MSE', plot = True):
        testLen = len(testSet)
        Y = np.array([])
        if initialize:
            x = initializer
            print(x)
            initLen = 0
        else:
            x = np.zeros((self.reservoirSize,1))
        u = testSet[0]
        for t in range(testLen):
            x = (1-self.leakRate)*x + self.leakRate*np.tanh(np.dot(self.Win, np.vstack((1,u))) \
                 + np.dot(self.W, x))
            y = np.dot(self.Wout, np.vstack((1,u,x)))
            u = np.copy(y)
            Y = np.append(Y, y)
            """if t >= initLen:
                Y = np.append(Y, y)"""
        if(lossType == 'MSE'):
            loss = np.mean(np.square(Y[initLen:] - labels[initLen:]))
            print('Mean Squared Error: ' + str(loss))
        elif(lossType == 'CE'):
            loss = -np.mean(labels[:]*np.log(Y))
            print('Mean Cross Entropy Loss: ' + str(loss))
            
        if plot:
            # Use this function to plot the number of datapoints you eant to plot
            plt.plot(range(len(labels[0:100])), labels[0:100], label ='Actual')
            plt.plot(range(len(labels[0:100])), Y[0:100], label = 'Generated')
            plt.legend(loc = 'best')
            plt.show()
        return loss



# Standard dat here represents Mackey-Glass data used by the original creator of
# the program to test the data
def load_data(data='standard'):
    if data == 'standard':
        path = '../data/Mackey-Glass.txt'
        i,o = _load_data(path, 0, 0, 1, 0)
        return i,o
    path = data
    i,o = _load_data(path, startLine = 1, start = 1, stop = 3, prediction = 0)
    return i,o



initLen = 100
leak_rate = 0.1
reservoir_size = 100
spectral_radius = 1.25


i,o = load_data() #'../data/CPU_Mem_Data1Yr.txt')
esn = ESN(leak_rate, 1, 1, reservoir_size, spectral_radius)
x = esn.train(10**-8, i[0][0:2000], initLen, i[0][0:2000])
loss = esn.test(i[0][2000:2500], initLen, i[0][2000:2500], initialize = True, initializer = x)
