# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 18:08:35 2019

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
Building the one hidden layer with 64 neuron architecture
'''

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
n_h=64

W1=np.random.randn(n_h,n_x)*0.01
b1=np.zeros((n_h,1))
W2=np.random.randn(1,n_h)*0.01
b2=np.zeros((1,1))


for i in range(2000):
    Z1=np.dot(W1,X)+b1
    A1=sigmoidfunction(Z1)
    Z2=np.dot(W2,A1)+b2
    A2=sigmoidfunction(Z2)
    
    
    cost=compute_cost(Y,A2)
    dZ2=A2-Y
    dW2=(1/m)*np.dot(dZ2,A1.T)
    db2=(1/m)*np.sum(dZ2,axis=1,keepdims=True)

    dA1=np.dot(W2.T,dZ2)
    dZ1=dA1*sigmoidfunction(Z1)*(1-sigmoidfunction(Z1))
    dW1=(1/m)*np.dot(dZ1,X.T)
    db1=(1/m)*np.sum(dZ1,axis=1,keepdims=True)

    W2=W2-(learningrate*dW2)
    b2=b2-(learningrate*db2)
    W1=W1-(learningrate*dW1)
    b1=b1-(learningrate*db1)
    
    
    if (i%100==0):
        print("Epoch of {} and the cost value is {}".format(i,cost))

print("Final Cost",cost)

'''
Testing 
'''
Z1=np.dot(W1,testX)+b1
A1=sigmoidfunction(Z1)
Z2=np.dot(W2,A1)+b2
A2=sigmoidfunction(Z2)


predictions=(A2>0.5)[0,:]
labels=(testy==1)[0,:]

print(confusion_matrix(predictions,labels))

print(classification_report(predictions,labels))


'''
Extending to multi class using softmax
'''

#One hot encode output labels

digits=10

trainyonehot=np.eye(digits)[trainy.astype('int32')]
trainyonehot=trainyonehot.T.reshape(digits,m)


testyonehot=np.eye(digits)[testy.astype('int32')]
testyonehot=testyonehot.T.reshape(digits,testy.shape[1])


'''
Shuffling the index values
'''

np.random.seed(138)
shuffle_index=np.random.permutation(m)


trainX,trainyonehot=trainX[:,shuffle_index],trainyonehot[:,shuffle_index]


'''
viewing one sample example
'''
i=12
plt.imshow(trainX[:,i].reshape(28,28),cmap=matplotlib.cm.binary)
print(trainyonehot[:,i])


'''
Updated cost function
'''

def multiclass_loss(y,yhat):
    L=-(1/m)*np.sum(np.multiply(y,np.log(yhat)))
    return L
    

learningrate=1
X=trainX
Y=trainyonehot

n_x=X.shape[0]
n_h=64

W1=np.random.randn(n_h,n_x)*0.01
b1=np.zeros((n_h,1))
W2=np.random.randn(digits,n_h)*0.01
b2=np.zeros((digits,1))


for i in range(2000):
    Z1=np.dot(W1,X)+b1
    A1=sigmoidfunction(Z1)
    Z2=np.dot(W2,A1)+b2
    A2=np.exp(Z2)/(np.sum(np.exp(Z2),axis=0))
    
    
    cost=multiclass_loss(Y,A2)
    dZ2=A2-Y
    dW2=(1/m)*np.dot(dZ2,A1.T)
    db2=(1/m)*np.sum(dZ2,axis=1,keepdims=True)

    dA1=np.dot(W2.T,dZ2)
    dZ1=dA1*sigmoidfunction(Z1)*(1-sigmoidfunction(Z1))
    dW1=(1/m)*np.dot(dZ1,X.T)
    db1=(1/m)*np.sum(dZ1,axis=1,keepdims=True)

    W2=W2-(learningrate*dW2)
    b2=b2-(learningrate*db2)
    W1=W1-(learningrate*dW1)
    b1=b1-(learningrate*db1)
    
    
    if (i%100==0):
        print("Epoch of {} and the cost value is {}".format(i,cost))

print("Final Cost",cost)

    
'''
Testing for multiclass
'''
Z1=np.dot(W1,testX)+b1
A1=sigmoidfunction(Z1)
Z2=np.dot(W2,A1)+b2
A2=np.exp(Z2)/(np.sum(np.exp(Z2),axis=0))


predictions=np.argmax(A2,axis=0)
labels=np.argmax(testyonehot,axis=0)

print(confusion_matrix(predictions,labels))

print(classification_report(predictions,labels))



    