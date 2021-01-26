import numpy as np
from Layers import DenseLayer

"""
This class represents a Neural Network with sequential layers.
You can add layers to this, (add)
define the loss, optimizer and metrics to calculate, (compile)
train the neural network (fit)
"""
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        
    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def fit(self, X, Y, num_epochs, print_after_steps=1):
        metrics = { 'cost': [] }
        
        # Repeat for num_epochs epochs
        for epoch in range(num_epochs):
            # Forward Propagation
            A = X
            for l in range(len(self.layers)):
                A = self.layers[l].forward_propagation(A)
                
            # Coumpute the overall cost
            cost = self.loss.compute_cost(Y, A)
            metrics['cost'].append(cost)
            
            # Compute dA for the last layer
            dA = self.loss.compute_grad(Y, A)
                
            # Backward Propagation
            for l in reversed(range(len(self.layers))):
                dA = self.layers[l].backward_propagation(Y, dA)
            
            # Simultaneously update the weights and biases
            for l in range(len(self.layers)):
                self.layers[l].update_parameters(self.optimizer)
                
            # Print status in colsole
            if epoch % print_after_steps == 0:
                print('Epoch:', epoch, 'Cost:', cost)
        
        return metrics
        
    def predict(self, X):
        # Forward Propagation
        A = X
        for l in range(len(self.layers)):
            A = self.layers[l].forward_propagation(A)
            
        return A
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    