# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 08:14:31 2019

@author: Nandhagopalan
"""
from sklearn.datasets import make_moons
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import math

'''
Sample inputs
'''

X, y = make_moons(n_samples = 5000, noise=0.2, random_state=100)


'''
Splitting the data
'''
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

X_train=X_train.T
X_test=X_test.T
y_train=y_train.reshape(-1,1).T
y_test=y_test.reshape(-1,1).T


'''
Defining the architecture
'''
networkArchitecture = [
    {"input_dim": X_train.shape[0], "output_dim":5, "activation": "relu"},
    {"input_dim": 5, "output_dim": 2, "activation": "relu"},
    {"input_dim": 2, "output_dim": 1, "activation": "sigmoid"}
]

'''
Activations and its derivatives
'''

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

'''
Accuracy
'''

def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_


def get_accuracy_value(Y,Y_hat):
    Y_hat = convert_prob_into_class(Y_hat)
    return (Y_hat == Y).all(axis=0).mean()

'''
Random initialization
'''

def initialize_parameters(nn_architecture):

    # number of layers in our neural network
    print("Total no of layers",len(nn_architecture))
    
    # parameters storage initiation
    params_values = {}
    
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        
        # extracting the number of units in layers
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        # initiating the values of the W matrix and vector b for subsequent layers
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1
        
    return params_values

'''
Forward Propogation
'''

def forward_propagation(X, parameters):
    
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    Z1 = np.dot(W1, X) + b1

    A1 = relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    
    return A3, cache

def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
                                                        # Steps 1-4 below correspond to the Steps 1-4 described above. 
    D1 = np.random.rand(A1.shape[0],A1.shape[1])      # Step 1: initialize matrix D1 = np.random.rand(..., ...)
    D1 = D1<0.5                                         # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
    A1 = np.multiply(A1,D1)                                         # Step 3: shut down some neurons of A1
    A1 = A1/keep_prob                                         # Step 4: scale the value of neurons that haven't been shut down
    
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    
    D2 = np.random.rand(A2.shape[0],A2.shape[1])                                         # Step 1: initialize matrix D2 = np.random.rand(..., ...)
    D2 = D2<0.5                                         # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
    A2 = np.multiply(A2,D2)                                       # Step 3: shut down some neurons of A2
    A2 = A2/keep_prob                                         # Step 4: scale the value of neurons that haven't been shut down
    
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    
    return A3, cache

'''
Computing cost
'''
def compute_cost(Y_hat, Y):
    # number of examples
    m = Y_hat.shape[1]
    # calculation of the cost according to the formula
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)


def compute_cost_with_regularization(A3, Y, parameters, lambd):
    
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
   
    cross_entropy_cost = compute_cost(A3, Y) # This gives you the cross-entropy part of the cost
    
    L2_regularization_cost = (np.sum(np.square(W1))+np.sum(np.square(W2))+np.sum(np.square(W3)))*lambd/(2*m)
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost

'''
BackPropogation
'''

def backward_propagation(X, Y, cache):
    
    
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    dA2 = np.dot(W3.T, dZ3)
    
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients


def backward_propagation_with_regularization(X, Y, cache, lambd):
    
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    
    dW3 = 1./m * np.dot(dZ3, A2.T) + (lambd/m)*W3
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T) + (lambd/m)*W2
 
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
   
    dW1 = 1./m * np.dot(dZ1, X.T) + (lambd/m)*W1
    
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients

def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    dA2 = np.dot(W3.T, dZ3)
    
    dA2 = np.multiply(D2,dA2)              # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
    dA2 = dA2/keep_prob              # Step 2: Scale the value of neurons that haven't been shut down
   
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
   
    dA1 = np.multiply(D1,dA1)              # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
    dA1 = dA1/keep_prob              # Step 2: Scale the value of neurons that haven't been shut down
   
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients

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

'''
Model Building
'''

def model_regularizer(X, Y,networkArchitecture,optimizer='gd', mini_batch_size = 64, beta = 0.9,
           beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8,
           learning_rate = 0.3, num_iterations = 1000, print_cost=True, lambd = 0, keep_prob = 1):
   
    t=0
    grads = {}
    costs = []                            # to keep track of the cost
    m = X.shape[1]                        # number of examples
    
    # Initialize parameters dictionary.
    parameters = initialize_parameters(networkArchitecture)
    
    # Initialize the optimizer
    if optimizer == "gd":
        pass # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
        
        
    # Loop (gradient descent)

    for i in range(0, num_iterations):
        
        minibatches = random_mini_batches(X, Y, mini_batch_size)

        for minibatch in minibatches:

            # Select a minibatch
          (minibatch_X, minibatch_Y) = minibatch

          if keep_prob == 1:
            a3, cache = forward_propagation(minibatch_X, parameters)
          elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(minibatch_X, parameters, keep_prob)
        
          # Cost function
          if lambd == 0:
            cost = compute_cost(a3, minibatch_Y)
          else:
            cost = compute_cost_with_regularization(a3, minibatch_Y, parameters, lambd)
            
          # Backward propagation.
          assert(lambd==0 or keep_prob==1)    # it is possible to use both L2 regularization and dropout, 
                                            # but this assignment will only explore one at a time
          if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(minibatch_X, minibatch_Y, cache)
          elif lambd != 0:
            grads = backward_propagation_with_regularization(minibatch_X, minibatch_Y, cache, lambd)
          elif keep_prob < 1:
            grads = backward_propagation_with_dropout(minibatch_X, minibatch_Y, cache, keep_prob)
        
          # Update parameters
          if optimizer == "gd":
                parameters = update_parameters(parameters, grads, learning_rate)
          elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
          elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2,  epsilon)
        
        # Print the loss every 10000 iterations
        if print_cost and i % 50 == 0:
            print("Cost after iteration",i,cost)
        if print_cost and i % 50 == 0:
            costs.append(cost)
    
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

parameters = model_regularizer(X_train,y_train,networkArchitecture)

#Regularized
parameters=model_regularizer(X_train,y_train,networkArchitecture,lambd = 0.7)

#Drop-out
parameters=model_regularizer(X_train,y_train,networkArchitecture,keep_prob = 0.86)


Y_test_hat, _ = forward_propagation(X_test, parameters)

acc_test = get_accuracy_value(y_test,Y_test_hat)
print("Test set accuracy: {:.2f} ".format(acc_test))





'''
Implementing along with the mini batches 
'''

def random_mini_batches(X, Y, mini_batch_size = 64):
      
    m = X.shape[1]                  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))
   
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
   
    for k in range(0, num_complete_minibatches):
        
        mini_batch_X = shuffled_X[:,(k*mini_batch_size):(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:,(k*mini_batch_size):(k+1)*mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
       
        mini_batch_X = shuffled_X[:,(num_complete_minibatches*mini_batch_size):((m-mini_batch_size)*num_complete_minibatches)]
        mini_batch_Y = shuffled_Y[:,(num_complete_minibatches*mini_batch_size):((m-mini_batch_size)*num_complete_minibatches)]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

'''
Momentum optimizer along with the gradient update
'''

def initialize_velocity(parameters):
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    
    # Initialize velocity
    for l in range(L):
       
        v["dW" + str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
        
        
    return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
   

    L = len(parameters) // 2 # number of layers in the neural networks
    
    # Momentum update for each parameter
    for l in range(L):
        
        # compute velocities
        v["dW" + str(l+1)] = (beta*v["dW"+str(l+1)])+((1-beta)*grads["dW"+str(l+1)])
        v["db" + str(l+1)] = (beta*v["db"+str(l+1)])+((1-beta)*grads["db"+str(l+1)])
        # update parameters
        parameters["W" + str(l+1)] = parameters["W"+str(l+1)]-learning_rate*v["dW"+str(l+1)] 
        parameters["b" + str(l+1)] = parameters["b"+str(l+1)]-learning_rate*v["db"+str(l+1)]
        
    return parameters, v

'''
Adam initialization
'''

def initialize_adam(parameters) :
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
    
        v["dW" + str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
    
    return v, s

'''
Adam Update
'''

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):

    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        
        v["dW" + str(l+1)] = (beta1*v["dW"+str(l+1)])+(1-beta1)*grads["dW"+str(l+1)]
        v["db" + str(l+1)] = (beta1*v["db"+str(l+1)])+(1-beta1)*grads["db"+str(l+1)]

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        
        v_corrected["dW" + str(l+1)] = v["dW"+str(l+1)]/(1-(np.power(beta1,t)))
        v_corrected["db" + str(l+1)] = v["db"+str(l+1)]/(1-(np.power(beta1,t)))
        

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        
        s["dW" + str(l+1)] = (beta2*s["dW"+str(l+1)])+(1-beta2)*(np.square(grads["dW"+str(l+1)]))
        s["db" + str(l+1)] = (beta2*s["db"+str(l+1)])+(1-beta2)*(np.square(grads["db"+str(l+1)]))


        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        
        s_corrected["dW" + str(l+1)] = s["dW"+str(l+1)]/(1-(np.power(beta2,t)))
        s_corrected["db" + str(l+1)] = s["db"+str(l+1)]/(1-(np.power(beta2,t)))

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-(learning_rate*((v_corrected["dW"+str(l+1)])/np.sqrt(s_corrected["dW"+str(l+1)]+epsilon))) 
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-(learning_rate*((v_corrected["db"+str(l+1)]/np.sqrt(s_corrected["db"+str(l+1)])+epsilon))) 

    return parameters,v,s




param = model_regularizer(X_train,y_train, networkArchitecture, optimizer = "momentum")


param = model_regularizer(X_train,y_train, networkArchitecture, optimizer = "adam")


