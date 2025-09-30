import numpy as np
import pandas as pd

data = pd.read_csv('/Users/alekmelenski/Desktop/NeuralNet/train.csv')
print("raw data\n", data)
data = np.array(data)

labels = data[:, 0]
print("lables", labels)
training = data[:, 1:]
print(training)

BATCHSIZE = 3

np.random.seed(42)

def init_params(batchsize, neurons, neurons2):
    W1 = np.random.uniform(-0.01, 0.01, (784, neurons))   # weights from input → hidden
    b1 = np.random.uniform(-0.5, 0.5, (batchsize, neurons))   # biases for hidden layer
    W2 = np.random.uniform(-0.01, 0.01, (neurons, neurons2))   # weights from hidden → output, output is neurons 2
    b2 = np.random.uniform(-0.5, 0.5, (batchsize, neurons2))   # biases for output layer
    return W1, b1, W2, b2
    
def ReLU(x):
    return np.maximum(0, x)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True) #need this for exp overflow
                      #Also note for future, if using batches, need an axis parameter
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def forward_prop(W1, b1, W2, b2, input):
    print("input: ", input.shape)
    #print("weight1: ", W1.shape)
    Z1 = input @ W1 +  b1
    A1 = ReLU(Z1)
    #print("A1: ", A1.shape)
    #print("weight2: ", W2.shape)
    Z2 = A1 @ W2 + b2
    #print("y before softmax: ", Z2)
    y = softmax(Z2)
    #print("y after softmax: ", y)
    #print("Z2: ", Z2.shape)
    #print("y: ", y.shape)
    #print(Z2)
    return Z1, A1, Z2, y

def loss(y, labels):
    y = np.clip(y, 1e-7, 1 - 1e-7) # Sets a range so you dont take the log of 0 (inf)
    loss =  -np.log(y[range(len(y)), labels[:BATCHSIZE]]) #categorical cross entropy (-log)
    cost = np.mean(loss)
    return cost

def accuracy(y, labels):
    predictions = np.argmax(y , axis=1)
    return np.mean(predictions == labels[:BATCHSIZE])
        

W1, b1, W2, b2 = init_params(1, 10, 10)
Z1, A1, Z2, y = forward_prop(W1, b1, W2, b2, training[0:BATCHSIZE])
print("y: ", y)

print("loss: ", loss(y, labels))
print("accuracy: ", accuracy(y, labels))
