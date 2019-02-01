# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 00:57:28 2019

@author: Nandhagopalan
"""

import numpy as np
from scipy import misc
import cv2
import glob
from PIL import Image
import os, sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


dogpath = "C:/Users/Nandhagopalan/Desktop/DataScience/DLImpl/DogsVsCats/Dog/"
catpath="C:/Users/Nandhagopalan/Desktop/DataScience/DLImpl/DogsVsCats/Cat/"

dirs = os.listdir( catpath )


'''
Resizing the images
'''
def resize():
    for item in dirs:
        if os.path.isfile(catpath+item):
            f, e = os.path.splitext(catpath+item)
            Image.open(catpath+item).convert('RGB').save(f+'.jpg')
            im = Image.open(catpath+item)
            imResize = im.resize((200,200), Image.ANTIALIAS)
            imResize.save(f + '.jpg', 'JPEG', quality=90)

resize()

'''
Reading the image converting shape and labelling the same
'''

dogs = [cv2.imread(file) for file in glob.glob(dogpath+'/*jpg')]
dogs=[dog.reshape(dog.shape[0]*dog.shape[1]*dog.shape[2],1)  for dog in dogs ]
dogsresized=np.concatenate( dogs, axis=1 )

cats = [cv2.imread(file) for file in glob.glob(catpath+'/*jpg')]
cats=[cat.reshape(cat.shape[0]*cat.shape[1]*cat.shape[2],1)  for cat in cats ]
catssresized=np.concatenate( cats, axis=1 )

data=np.concatenate([dogsresized,catssresized],axis=1)

doglabels=np.zeros([dogsresized.shape[1],1])
catlabels=np.ones([catssresized.shape[1],1])
labels=np.concatenate([doglabels,catlabels], axis=0)
labels=labels.T

'''
Reading a random sample of cats and dogs
'''

i=203

plt.imshow(data[:,i].reshape(200,200,3),cmap=plt.cm.binary)
print(labels[:,i])

'''
Shuffling the index values
'''
m=data.shape[1]
np.random.seed(138)
shuffle_index=np.random.permutation(m)

data,labels=data[:,shuffle_index],labels[:,shuffle_index]

'''
Normalise the data
'''
data=data/255

'''
Splitting the data
'''
X_train, X_test, y_train, y_test = train_test_split(data.T, labels.T, test_size=0.2)

X_train=X_train.T
X_test=X_test.T
y_train=y_train.T
y_test=y_test.T

'''
Helper functions for building the L layer architecture
'''
def sigmoid(z):
    return (1/(1+np.exp(-z))),z

def relu(z):
    return (z* (z > 0)),z

'''
Initialization function (HE initialization) for deep layer model
'''
def initialize_parameters_he(layers_dims):
     
    parameters = {}
    L = len(layers_dims)  # integer representing the number of layers
     
    for l in range(1,L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(1/layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
        
    return parameters


'''
Forward Propogation model with Linear--RELU[l-1] & linear--Sigmoid[l] layers
'''

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
        
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = np.dot(W,A)+b
    
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
       
    if activation == "sigmoid":
       
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)
      
    
    elif activation == "relu":
        
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)
        
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters)//2                  # number of layers in the neural network
   
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],"relu")
        caches.append(cache)
       
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    
    AL, cache = linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid")
    caches.append(cache)
    
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches

'''
Computing the cost
'''
def compute_cost(AL, Y):
    """
    
    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = -(np.sum(np.multiply(Y,np.log(AL))+np.multiply((1-Y),np.log(1-AL)),axis=1,keepdims=True))/m
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    
    return cost

'''
BackPropogation Function
'''

def linear_backward(dZ, cache):
    """

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ,A_prev.T)/m
    db = np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def sigmoid_backward(da,activation):
    sig,_ = sigmoid(activation)
    return da* sig * (1 - sig)

def relu_backward(da,activation):
    #derivative of relu
     activation[activation<=0] = 0
     activation[activation>0] = 1
     return da*activation

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
       
        dZ = relu_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
                
    elif activation == "sigmoid":
        
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
    
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
   
    # Initializing the backpropagation
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
   
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,current_cache,"sigmoid")
   
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)],current_cache,"relu")
        grads["dA" + str(l)] = dA_prev_temp[l]
        grads["dW" + str(l + 1)] = dW_temp[l]
        grads["db" + str(l + 1)] = db_temp[l]
        
    return grads


"""
Update parameters using gradient descent
"""
def update_parameters(parameters, grads, learning_rate):
    
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)] 
        parameters["b" + str(l+1)] = parameters["b"+str(l+1)]-learning_rate*grads["db"+str(l+1)]
    
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate = 0.001, num_iterations = 3000, print_cost=False):#lr was 0.009

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
   
    parameters = initialize_parameters_he(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.        
        AL, caches = L_model_forward(X,parameters)
        
        # Compute cost.        
        cost = compute_cost(AL,Y)
            
        # Backward propagation.        
        grads = L_model_backward(AL,Y,caches)
         
        # Update parameters.
        parameters = update_parameters(parameters,grads,learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

layers_dims = [X_train.shape[0],100,75,60,40,10, y_train.shape[0]] #  4-layer model
parameters = L_layer_model(X_train,y_train, layers_dims, num_iterations = 2500, print_cost = True)




