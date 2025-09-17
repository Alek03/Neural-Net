import numpy as np
import pandas as pd

data = pd.read_csv('C:/Users/al4k9/OneDrive/Desktop/Neural Net/train.csv')
print("raw data", data)
data = np.array(data)

labels = data[:, 0]
training = data[:, 1:]
print(training)

np.random.seed(42)

def init_params():
    W1 = np.random.uniform(-0.5, 0.5, (10, 784))  # weights from input → hidden
    b1 = np.random.uniform(-0.5, 0.5, (10, 1))   # biases for hidden layer
    W2 = np.random.uniform(-0.5, 0.5, (10, 10))   # weights from hidden → output
    b2 = np.random.uniform(-0.5, 0.5, (10, 1))    # biases for output layer
    return W1, b1, W2, b2
    
def ReLU(x):
    return np.maximum(0, x)

def softmax(x):
    x = x - np.max(x) #need this for exp overflow
                      #Also note for future, if using batches, need an axis parameter
    return np.exp(x) / np.sum(np.exp(x))

def forward_prop(W1, b1, W2, b2, input):
    Z1 = W1 @ input +  b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    y = softmax(Z2)
    #print(Z2)
    return Z1, A1, Z2, y

W1, b1, W2, b2 = init_params()
Z1, A1, Z2, y = forward_prop(W1, b1, W2, b2, training[0])
print(y)