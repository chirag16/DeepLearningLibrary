from abc import ABC, abstractmethod
import numpy as np

"""
class: Activation
This is the base class for all activation functions.
It has 2 methods -
compute_output - this is used during forward propagation. Calculates A given Z
copute_grad - this is used during back propagation. Calculates dZ given dA and A
"""
class Activation(ABC):
    @abstractmethod
    def compute_output(self, Z):
        pass
    
    @abstractmethod
    def compute_grad(self, A, dA):
        pass
    

"""
class: Sigmoid
This activation is used in the last layer for networks performing binary classification.
"""
class Sigmoid(Activation):
    def __init__(self):
        pass
    
    def compute_output(self, Z):
        return 1. / (1 + np.exp(-Z))
    
    def compute_grad(self, Y, A, dA):
        return dA * A * (1 - A)
    
    
"""
class: Softmax
This activation is used in the last layer for networks performing multi-class classification.
"""
class Softmax(Activation):
    def __init__(self):
        pass
    
    def compute_output(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis=0)
    
    def compute_grad(self, Y, A, dA):
        return A - Y
    
"""
class ReLU
This activation is used in hidden layers.
"""
class ReLU(Activation):
    def __init__(self):
        pass
    
    def compute_output(self, Z):
        A = Z
        A[Z < 0] = 0
        return A
    
    def compute_grad(self, Y, A, dA):
        dZ = dA
        dZ[A == 0] = 0
        return dZ
    