import numpy as np
from copy import deepcopy

"""
This class represents a Neural Network with sequential layers.
You can add layers to this, (add)
define the loss, optimizer and metrics to calculate, (compile)
train the neural network (fit)
"""
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        
    def compile(self, loss, optimizer, metrics):
        self.loss = loss
        self.metrics = metrics
        
        # Set a unique optimizer for each layer
        for i in range(len(self.layers)):
            self.layers[i].set_optimizer(deepcopy(optimizer))
        
        # Initialize metrics to be computed
        self.computed_metrics = {}
        for metric in self.metrics:
            self.computed_metrics[metric] = []

    def fit(self, X, Y, validation_X, validation_Y, num_epochs, batch_size=16, print_after_steps=1):
        # Store validation data to use it later to compute validation metrics
        self.validation_X = validation_X
        self.validation_Y = validation_Y
        
        # Calculate steps per epoch
        m = X.shape[1]  # No. training samples
        steps_per_epoch = int(m / batch_size)
        
        if steps_per_epoch == 0:
            steps_per_epoch = 1
            batch_size = m
        
        # Repeat for num_epochs epochs
        for epoch in range(num_epochs):
            # Train on one mini-batch
            mini_batch_start = 0
            for step in range(steps_per_epoch):
                # Get next batch to train on
                X_mini_batch = X[:, mini_batch_start:mini_batch_start + batch_size]
                Y_mini_batch = Y[:, mini_batch_start:mini_batch_start + batch_size]
                mini_batch_start = mini_batch_start + batch_size
                
                # Forward Propagation
                A_mini_batch = X_mini_batch
                for l in range(len(self.layers)):
                    A_mini_batch = self.layers[l].forward_propagation(A_mini_batch)
                
                # Compute dA for the last layer
                dA_mini_batch = self.loss.compute_grad(Y_mini_batch, A_mini_batch)
                    
                # Backward Propagation
                for l in reversed(range(len(self.layers))):
                    dA_mini_batch = self.layers[l].backward_propagation(Y_mini_batch, dA_mini_batch)
                
                # Simultaneously update the weights and biases
                for l in range(len(self.layers)):
                    self.layers[l].update_parameters()
                    
                                    
                # Compute the metrics
                # This step is done at the end so as to avoid messing up the cahsed values 
                # of A, Z etc in the layers of the network.
                for metric in self.metrics:
                    computed_metric = self.compute_metric(metric, Y_mini_batch, A_mini_batch)
                    self.computed_metrics[metric].append(computed_metric)
                    
                # Print status in console
                if step % print_after_steps == 0:
                    self.print_metrics(epoch=epoch, steps_per_epoch=steps_per_epoch, step=step)
        
        return self.computed_metrics
    
    def compute_metric(self, metric, Y, A):
        if metric == 'cost':
            return self.loss.compute_cost(Y, A)
        elif metric == 'accuracy':
            Y_prediction = np.zeros_like(A)
            Y_prediction[A > 0.5] = 1
            
            batch_size = Y.shape[1]
            correct_predictions = np.min(Y == Y_prediction, axis=0).sum()
            return (correct_predictions / batch_size) * 100
        elif metric == 'validation_accuracy':
            validation_A = self.predict(self.validation_X)
            
            Y_prediction = np.zeros_like(validation_A)
            Y_prediction[validation_A > 0.5] = 1
            
            batch_size = self.validation_Y.shape[1]
            correct_predictions = np.min(self.validation_Y == Y_prediction, axis=0).sum()
            return (correct_predictions / batch_size) * 100
        
    def print_metrics(self, epoch, steps_per_epoch, step):
        print('\nEpoch:', epoch, 'Step:', step)
        for metric in self.metrics:
            print(metric, self.computed_metrics[metric][epoch * steps_per_epoch + step], end=' ')
        
    def predict(self, X):
        # Forward Propagation
        A = X
        for l in range(len(self.layers)):
            A = self.layers[l].forward_propagation(A)
            
        return A
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    