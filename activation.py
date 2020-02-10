import numpy as np
"""
Activation function

More read .................
"""
def sigmoid(x):
    return 1/(1+np.exp(-x))
def relu(x):
    return np.maximum(0.0,x)
def softmax(x):
    expo = np.exp(x)
    expo_sum = np.sum(np.exp(x))
    return expo/expo_sum
