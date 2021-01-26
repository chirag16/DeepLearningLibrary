from abc import ABC, abstractmethod
import numpy as np

"""
class: Layer
This is the base class for any layer in the neural network.
You can define the number of neurons, (n_h)
the activation function (activation),
and the size of the input.
"""
class Layer(ABC):
    @abstractmethod
    def forward_propagation(self, A_prev):
        pass
    
    @abstractmethod
    def backward_propagation(self, dA):
        pass
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def update_parameters(self):
        self.W, self.b = self.optimizer.optimize(self.W, self.b, self.dW, self.db)
    

"""
class: DenseLayer
This class represents one dense layer in the neural network.
"""
class DenseLayer(Layer):
    def __init__(self, n_h, activation, n_prev):
        self.W = np.random.randn(n_h, n_prev) * 0.001
        self.b = np.zeros((n_h, 1))
        self.activation = activation
        
        # Stuff to cache to be used later for back propagation
        self.A_prev = None
        self.Z = None
        self.A = None
        
        # Gradients to cache to be used for future simultaneous update of parameters
        self.dW = None
        self.db = None
    
    def forward_propagation(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.dot(self.W, self.A_prev) + self.b
        
        self.A = self.activation.compute_output(self.Z)
        
        return self.A
    
    def backward_propagation(self, Y, dA):
        dZ = self.activation.compute_grad(Y, self.A, dA)
            
        m = dA.shape[1]     # m = no. of training examples
        self.dW = (1. / m) * np.dot(dZ, self.A_prev.T)
        self.db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
        
        dA_prev = np.dot(self.W.T, dZ)
        
        return dA_prev
        