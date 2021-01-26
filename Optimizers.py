from abc import ABC, abstractmethod
import numpy as np

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
    
"""
class: Momentum
V_dW = 0
V_dW = beta * V_dW + (1 - beta) * dW

V_db = 0
V_db = beta * V_db + (1 - beta) * db

W = W - learning_rate * V_dW
b = b - learning_rate * V_db
"""
class Momentum(Optimizer):
    def __init__(self, learning_rate, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        
        self.V_dW = None
        self.V_db = None
        
    def optimize(self, W, b, dW, db):
        # Initialize the moving averages if not done yet
        if np.any(self.V_dW == None):
            self.V_dW = np.zeros_like(W)
        if np.any(self.V_db == None):
            self.V_db = np.zeros_like(b)
        
        # Update the moving averages of the gradients
        self.V_dW = self.beta * self.V_dW + (1 - self.beta) * dW
        self.V_db = self.beta * self.V_db + (1 - self.beta) * db
        
        W = W - self.learning_rate * self.V_dW
        b = b - self.learning_rate * self.V_db
        
        return W, b
    
"""
class: RMSprop
S_dW = 0
S_dW = beta * S_dW + (1 - beta) * (dW) ** 2

S_db = 0
S_db = beta * S_db + (1 - beta) * (db) ** 2

W = W - learning_rate * dW / (S_dW) ** 0.5
b = b - learning_rate * db / (S_db) ** 0.5
"""
class RMSprop(Optimizer):
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        
        self.S_dW = None
        self.S_db = None
        
    def optimize(self, W, b, dW, db):
        # Initialize the moving RMSes if not done yet
        if np.any(self.S_dW == None):
            self.S_dW = np.zeros_like(W)
        if np.any(self.S_db == None):
            self.S_db = np.zeros_like(b)
                
        # Update the moving averages of the RMSes
        self.S_dW = self.beta * self.S_dW + (1 - self.beta) * np.square(dW)
        self.S_db = self.beta * self.S_db + (1 - self.beta) * np.square(db)
        
        W = W - self.learning_rate * dW / (np.sqrt(self.S_dW) + self.epsilon)
        b = b - self.learning_rate * db / (np.sqrt(self.S_db) + self.epsilon)
        
        return W, b
    
        
"""
class: Adam
V_dW = 0
V_dW = beta1 * V_dW + (1 - beta1) * dW

V_dW_corrected = V_dW / (1 - beta1 ** t)

V_db = 0
V_db = beta1 * V_db + (1 - beta1) * db

V_db_corrected = V_db / (1 - beta1 ** t)

S_dW = 0
S_dW = beta2 * S_dW + (1 - beta2) * (dW) ** 2

S_dW_corrected = S_dW / (1 - beta2 ** t)

S_db = 0
S_db = beta2 * S_db + (1 - beta2) * (db) ** 2

S_db_corrected = S_db / (1 - beta2 ** t)

W = W - learning_rate * V_dW / ((S_dW) ** 0.5 + epsilon)
b = b - learning_rate * V_db / ((S_db) ** 0.5 + epsilon)
"""
class Adam(Optimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.V_dW = None
        self.V_db = None
        self.S_dW = None
        self.S_db = None
        
        self.t = 1
        
    def optimize(self, W, b, dW, db):
        # Initialize the moving averages if not done yet
        if np.any(self.V_dW == None):
            self.V_dW = np.zeros_like(W)
        if np.any(self.V_db == None):
            self.V_db = np.zeros_like(b)
            
        # Initialize the moving RMSes if not done yet
        if np.any(self.S_dW == None):
            self.S_dW = np.zeros_like(W)
        if np.any(self.S_db == None):
            self.S_db = np.zeros_like(b)        
    
        # Update the moving averages of the gradients
        self.V_dW = self.beta1 * self.V_dW + (1 - self.beta1) * dW
        V_dW_corrected = self.V_dW / (1 - self.beta1 ** self.t)
        
        self.V_db = self.beta1 * self.V_db + (1 - self.beta1) * db
        V_db_corrected = self.V_db / (1 - self.beta1 ** self.t)
        
        # Update the moving averages of the RMSes
        self.S_dW = self.beta2 * self.S_dW + (1 - self.beta2) * np.square(dW)
        S_dW_corrected = self.S_dW / (1 - self.beta2 ** self.t)
        
        self.S_db = self.beta2 * self.S_db + (1 - self.beta2) * np.square(db)
        S_db_corrected = self.S_db / (1 - self.beta2 ** self.t)
        
        W = W - self.learning_rate * V_dW_corrected / (np.sqrt(S_dW_corrected) + self.epsilon)
        b = b - self.learning_rate * V_db_corrected / (np.sqrt(S_db_corrected) + self.epsilon)
        
        return W, b