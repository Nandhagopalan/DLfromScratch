# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 20:57:39 2019

@author: Nandhagopalan
"""

from mlxtend.data import loadlocal_mnist  
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import classification_report,confusion_matrix 
%matplotlib inline

trainX,trainy = loadlocal_mnist(
        images_path='train-images.idx3-ubyte', 
        labels_path='train-labels.idx1-ubyte')

testX,testy = loadlocal_mnist(
        images_path='t10k-images.idx3-ubyte', 
        labels_path='t10k-labels.idx1-ubyte')

m=trainX.shape[0]
#Normalise the data

trainX=trainX/255
testX=testX/255

'''
Preparing for Zero classifier
'''
zerotrainy = np.zeros(trainy.shape)
zerotrainy[np.where(trainy == 0.0)] = 1
trainy=zerotrainy

zerotesty = np.zeros(testy.shape)
zerotesty[np.where(testy == 0.0)] = 1
testy=zerotesty

'''
Reshaping the x and y attributes
'''
trainX=trainX.T     # shape(nx,m)
trainy=trainy.reshape(1,trainy.shape[0])  # shape (1,m)
testX=testX.T
testy=testy.reshape(1,testy.shape[0])

'''
Shuffling the index values
'''
np.random.seed(138)
shuffle_index=np.random.permutation(m)

trainX,trainy=trainX[:,shuffle_index],trainy[:,shuffle_index]

'''
viewing one sample example
'''
i=44
plt.imshow(trainX[:,i].reshape(28,28),cmap=matplotlib.cm.binary)
print(trainy[:,i])


'''
Building the single neuron architecture
''''

def sigmoidfunction(z):
    return (1/(1+np.exp(-z)))

def compute_cost(y,yhat):
    m=y.shape[1]
    L= -(1/m)*np.sum(np.multiply(y,np.log(yhat))+ np.multiply((1-y),(np.log(1-yhat))))
    return L
'''
Training phase
'''

learningrate=1
X=trainX
Y=trainy

n_x=X.shape[0]

W=np.random.randn(n_x,1)*0.01
b=np.zeros((1,1))

for i in range(2000):
    Z=np.dot(W.T,X)+b
    A=sigmoidfunction(Z)
    
    cost=compute_cost(Y,A)
    dW=(1/m)*np.dot(X,(A-Y).T)
    db=(1/m)*np.sum(A-Y,axis=1)

    W=W-(learningrate*dW)
    b=b-(learningrate*db)
    
    if (i%100==0):
        print("Epoch of {} and the cost value is {}".format(i,cost))

print("Final Cost",cost)

'''
Testing 
'''
testZ=np.dot(W.T,testX)+b
testA=sigmoidfunction(testZ)

predictions=(testA>0.5)[0,:]
labels=(testy==1)[0,:]

print(confusion_matrix(predictions,labels))

print(classification_report(predictions,labels))





