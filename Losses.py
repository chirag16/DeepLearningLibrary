from abc import ABC, abstractmethod
import numpy as np

"""
class: Loss
This is the base class for all losses.
It has 2 methods - 
compute_cost - This calculates the sum of the losses over all training examples, given Y_hat, Y
compute_grad - This calculates dA for the last layer.
"""
class Loss(ABC):
    @abstractmethod
    def compute_cost(self, Y, A):
        pass
    
    @abstractmethod
    def compute_grad(self, Y, A):
        pass
    

"""
class: BinaryCrossentropy
This loss is used for binary classification.
"""
class BinaryCrossentropy(Loss):
    def __init__(self):
        pass
    
    def compute_cost(self, Y, A):
        m = A.shape[1]     # m = no. of training examples
        return (- 1. / m) * (Y * np.log(A) + (1 - Y) * np.log(1 - A)).sum()
        
    def compute_grad(self, Y, A):
        return - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))
    

"""
class: CategoricalCrossentropy
This loss is used for multi-class classification.
"""
class CategoricalCrossentropy(Loss):
    def __init__(self):
        pass
    
    def compute_cost(self, Y, A):
        m = A.shape[1]     # m = no. of training examples
        return (- 1. / m) * (Y * np.log(A)).sum()
        
    def compute_grad(self, Y, A):
        return - np.divide(Y, A)
    
    
    
    
    
    
    
    
    
    
    