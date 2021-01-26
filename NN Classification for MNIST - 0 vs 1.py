# Read and pre-process the images
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

from NeuralNetwork import NeuralNetwork
from Layers import DenseLayer
from Optimizers import GradientDescent
from Losses import BinaryCrossentropy
from Activations import Sigmoid, ReLU

base_dir = os.path.join('D:', 'Datasets', 'MNIST', 'archive')
train_dir = os.path.join(base_dir, 'trainingSet', 'trainingSet')
test_dir = os.path.join(base_dir, 'testSet', 'testSet')

train_zeros_dir = os.path.join(train_dir, '0')
train_ones_dir = os.path.join(train_dir, '1')

zero_file_names = [os.path.join(train_zeros_dir, file) for file in os.listdir(train_zeros_dir)]
one_file_names = [os.path.join(train_ones_dir, file) for file in os.listdir(train_ones_dir)]

train_file_names = zero_file_names + one_file_names
random.shuffle(train_file_names)

print('Train file names:', train_file_names[:10])

def preprocess(X):
    return X / 255    # For now preprocessing only includes this step

IMG_SIZE = 28
NUM_CLASSES = 10

images = []
labels = []
for file in train_file_names:
    img = cv2.resize(cv2.imread(file, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
    images.append(img.flatten())
    
    if '\\0\\' in file: 
        labels.append(0)
    else: labels.append(1)
    
X, Y = np.array(images).T, np.array(labels).reshape(1, -1)

# Look at sample data
rows, cols = 1, 5
fig = plt.figure(figsize=(rows, cols))
for i in range(1, rows * cols + 1):
    img = X.T[i - 1].reshape(IMG_SIZE, IMG_SIZE)
    fig.add_subplot(rows, cols, i)
    plt.imshow(img)
plt.show()

print('Sample of training labels:', Y[:, :5])

# Split the training and testing data
X = preprocess(X)
X_train, Y_train = X[:, :-100], Y[:, :-100]
X_test, Y_test = X[:, -100:], Y[:, -100:]

print('Shape of training images', X_train.shape)
print('Shape of training labels', Y_train.shape)
print('Shape of testing images', X_test.shape)
print('Shape of testing labels', Y_test.shape)

# Define the model
model = NeuralNetwork([
            DenseLayer(256, activation=ReLU(), n_prev=X.shape[0]),
            DenseLayer(1, activation=Sigmoid(), n_prev=256)
        ])

model.compile(loss=BinaryCrossentropy(), optimizer=GradientDescent(learning_rate=7e-2))

# Train the model
metrics = model.fit(X_train, Y_train, num_epochs=800, print_after_steps=100)

# Display trend of costs

plt.plot(range(len(metrics['cost'])), metrics['cost'])
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.show()

# Make predictions
Y_predictions = model.predict(X_test)
print(Y_predictions)

Y_predictions[Y_predictions > 0.5] = 1
Y_predictions[Y_predictions <= 0.5] = 0

accuracy = np.ones_like(Y_predictions)[Y_predictions == Y_test].sum() / (Y_predictions.shape[0])
print('Accuracy on test data:', accuracy)