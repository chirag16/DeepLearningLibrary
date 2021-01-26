from abc import ABC, abstractmethod

"""
class: Optimizer
This is the base class for all optimizers to use.
It has the method optimize to update parameters, given the parameters and gradients.
"""
class Optimizer(ABC):
    @abstractmethod
    def optimize(self, W, b, dW, db):
        pass
    
"""
class: GradientDescent optimizer
W = W - learning_rate * dW
b = b - learning_rate * db
"""
class GradientDescent(Optimizer):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        
    def optimize(self, W, b, dW, db):
        W = W - self.learning_rate * dW
        b = b - self.learning_rate * db
        
        return W, b