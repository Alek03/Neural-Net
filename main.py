import numpy as np
import pandas as pd
import os

cwd = os.getcwd()

data = pd.read_csv(os.path.join(cwd, "train.csv"))
testing = pd.read_csv(os.path.join(cwd, "test.csv"))
data = np.array(data)
testing = np.array(testing)

labels = data[:, 0]
training = data[:, 1:]

#np.random.seed(42)

def init_params(neurons, neurons2):
    W1 = np.random.uniform(-0.01, 0.01, (784, neurons))   # weights from input → hidden
    b1 = np.random.uniform(-0.5, 0.5, (1, neurons))   # biases for hidden layer
    W2 = np.random.uniform(-0.01, 0.01, (neurons, neurons2))   # weights from hidden → output, output is neurons 2
    b2 = np.random.uniform(-0.5, 0.5, (1, neurons2))   # biases for output layer
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
    batch_size = len(labels)

    y = np.clip(y, 1e-7, 1 - 1e-7) # Sets a range so you dont take the log of 0 (inf)

    loss = -np.log(y[range(len(labels)), labels]) #categorical cross entropy (-log) | selects the index of the correct label for each row (batch)
    cost = np.mean(loss)

    return cost

def dLoss(y ,labels):
    '''
    Derivative of loss with respect to Z2 (softmax inputs)
    We do not do dLoss * dSoftMax because dSoftMax is too complex
    Math works out so that we can combine these steps
    Formula: predicted - groud truth
    '''
    
    one_hot = np.eye(10)[labels] #One hot encode labels
                                 #one_hot = np.eye(10)[labels[:BATCHSIZE]]
    #Gradient                       ^Interesting to note that this still works somehow
    dLoss = y - one_hot
    #Normalize Gradient
    dLoss_norm = dLoss / len(labels)

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
    return np.mean(predictions == labels)
        
W1, b1, W2, b2 = init_params(10, 10)

epochs = 10000
LEARNING_RATE = 0.001
BATCHSIZE = 500

for epoch in range(epochs):
    print(f"\nEpoch: {epoch+1}")
    
    indicies = np.random.randint(0, len(training), size = BATCHSIZE)
    training_batch = training[indicies]
    labels_batch = labels[indicies]

    # Forward pass
    Z1, A1, Z2, y = forward_prop(W1, b1, W2, b2, training_batch)
    
    # Backward pass
    dLoss_output = dLoss(y, labels_batch)
    dWeight2, dBias2, dInput2 = dLayer(A1, W2, dLoss_output)
    dReLU_output = dReLU(dInput2, Z1)
    dWeight1, dBias1, dInput1 = dLayer(training_batch, W1, dReLU_output)
    
    # gradient descent
    W1 -= LEARNING_RATE * dWeight1
    W2 -= LEARNING_RATE * dWeight2
    b1 -= LEARNING_RATE * dBias1
    b2 -= LEARNING_RATE * dBias2
    
    #prints
    current_loss = loss(y, labels_batch)
    current_accuracy = accuracy(y, labels_batch)
    print(f"Loss: {current_loss:.7f}, Accuracy: {current_accuracy:.4f}")


def kaggleTest(testing):
    
    _, _, _, predicted = forward_prop(W1, b1, W2, b2, testing)

    predicted = np.argmax(predicted, axis = 1)

    predictedDf = pd.DataFrame({
        "ImageId": np.arange(1, len(predicted) + 1),
        "Label": predicted
        })
    
    print(predictedDf)
    predictedDf.to_csv("submission.csv", index=False)

#kaggleTest(testing=testing)
