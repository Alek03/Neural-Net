import numpy as np
import pandas as pd

data = pd.read_csv('path')
testing = pd.read_csv('path')
data = np.array(data)

labels = data[:, 0]
training = data[:, 1:]

BATCHSIZE = 100
inputs = training[0:BATCHSIZE]

#np.random.seed(42)

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
    Z1 = input @ W1 +  b1
    A1 = ReLU(Z1)
    Z2 = A1 @ W2 + b2
    y = softmax(Z2)
    return Z1, A1, Z2, y

def loss(y, labels):
    y = np.clip(y, 1e-7, 1 - 1e-7) # Sets a range so you dont take the log of 0 (inf)
    loss = -np.log(y[range(len(y)), labels[:BATCHSIZE]]) #categorical cross entropy (-log) | selects the index of the correct label for each row (batch)
    cost = np.mean(loss)
    return cost

def dLoss(y ,labels):
    '''
    Derivative of loss with respect to Z2 (softmax inputs)
    We do not do dLoss * dSoftMax (chain rule) because dSoftMax is too complex
    Math works out so that we can combine these steps
    Formula: predicted - ground truth
    '''
    
    one_hot = np.eye(10)[labels[:BATCHSIZE]] #One hot encode labels

    #Gradient
    dLoss = y - one_hot
    #Normalize Gradient
    dLoss_norm = dLoss / BATCHSIZE

    return dLoss_norm

def dLayer(inputs, weights, dvalues):
    #Gradients on Weights + Bias
    dWeight = inputs.T @ dvalues
    dBias = np.sum(dvalues, axis = 0, keepdims=True)
    #Gradients on values
    dInput = dvalues @ weights.T
    return dWeight, dBias, dInput

def dReLU(dvalues, inputs):
    dInputs = dvalues.copy()
    dInputs[inputs <= 0] = 0
    return dInputs

def accuracy(y, labels):
    predictions = np.argmax(y , axis=1)
    return np.mean(predictions == labels[:BATCHSIZE])
        
W1, b1, W2, b2 = init_params(BATCHSIZE, 10, 10)

epochs = 100
LEARNING_RATE = 0.001
for epoch in range(epochs):
    print(f"\n=== Epoch {epoch+1} ===")
    
    #Forward pass
    Z1, A1, Z2, y = forward_prop(W1, b1, W2, b2, inputs)
    
    #Backward pass
    dLoss_output = dLoss(y, labels)
    dWeight2, dBias2, dInput2 = dLayer(A1, W2, dLoss_output)
    dReLU_output = dReLU(dInput2, Z1)
    dWeight1, dBias1, dInput1 = dLayer(inputs, W1, dReLU_output)
    
    #Weight & Bias update
    W1 -= LEARNING_RATE * dWeight1
    W2 -= LEARNING_RATE * dWeight2
    b1 -= LEARNING_RATE * dBias1
    b2 -= LEARNING_RATE * dBias2
    
    #Print
    current_loss = loss(y, labels)
    current_accuracy = accuracy(y, labels)
    print(f"Loss: {current_loss:.7f}, Accuracy: {current_accuracy:.4f}")
