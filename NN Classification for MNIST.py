# Read and pre-process the images
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

from NeuralNetwork import NeuralNetwork
from Layers import DenseLayer
from Optimizers import GradientDescent, Momentum, RMSprop, Adam
from Losses import CategoricalCrossentropy
from Activations import Softmax, ReLU

base_dir = os.path.join('D:', 'Datasets', 'MNIST', 'archive')
train_dir = os.path.join(base_dir, 'trainingSet', 'trainingSet')
test_dir = os.path.join(base_dir, 'testSet', 'testSet')

train_zeros_dir = os.path.join(train_dir, '0')
train_ones_dir = os.path.join(train_dir, '1')
train_twos_dir = os.path.join(train_dir, '2')
train_threes_dir = os.path.join(train_dir, '3')
train_fours_dir = os.path.join(train_dir, '4')
train_fives_dir = os.path.join(train_dir, '5')
train_sixes_dir = os.path.join(train_dir, '6')
train_sevens_dir = os.path.join(train_dir, '7')
train_eights_dir = os.path.join(train_dir, '8')
train_nines_dir = os.path.join(train_dir, '9')

zero_file_names = [os.path.join(train_zeros_dir, file) for file in os.listdir(train_zeros_dir)]
one_file_names = [os.path.join(train_ones_dir, file) for file in os.listdir(train_ones_dir)]
two_file_names = [os.path.join(train_twos_dir, file) for file in os.listdir(train_twos_dir)]
three_file_names = [os.path.join(train_threes_dir, file) for file in os.listdir(train_threes_dir)]
four_file_names = [os.path.join(train_fours_dir, file) for file in os.listdir(train_fours_dir)]
five_file_names = [os.path.join(train_fives_dir, file) for file in os.listdir(train_fives_dir)]
six_file_names = [os.path.join(train_sixes_dir, file) for file in os.listdir(train_sixes_dir)]
seven_file_names = [os.path.join(train_sevens_dir, file) for file in os.listdir(train_sevens_dir)]
eight_file_names = [os.path.join(train_eights_dir, file) for file in os.listdir(train_eights_dir)]
nine_file_names = [os.path.join(train_nines_dir, file) for file in os.listdir(train_nines_dir)]

train_file_names = zero_file_names + one_file_names + one_file_names + two_file_names + three_file_names + four_file_names + six_file_names + seven_file_names + eight_file_names + nine_file_names 
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
    
    label = np.zeros((NUM_CLASSES, ))
    if '\\0\\' in file: 
        label[0] = 1
    elif '\\1\\' in file: 
        label[1] = 1
    elif '\\2\\' in file: 
        label[2] = 1
    elif '\\3\\' in file: 
        label[3] = 1
    elif '\\4\\' in file: 
        label[4] = 1
    elif '\\5\\' in file: 
        label[5] = 1
    elif '\\6\\' in file: 
        label[6] = 1
    elif '\\7\\' in file: 
        label[7] = 1
    elif '\\8\\' in file: 
        label[8] = 1
    elif '\\9\\' in file: 
        label[9] = 1
        
    labels.append(label)
    
X, Y = np.array(images).T, np.array(labels).T

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
X_train, Y_train = X[:, :-300], Y[:, :-300]
X_test, Y_test = X[:, -300:], Y[:, -300:]

print('Shape of training images', X_train.shape)
print('Shape of training labels', Y_train.shape)
print('Shape of testing images', X_test.shape)
print('Shape of testing labels', Y_test.shape)

# Define the model
model = NeuralNetwork([
            DenseLayer(256, activation=ReLU(), n_prev=X.shape[0]),
            DenseLayer(NUM_CLASSES, activation=Softmax(), n_prev=256)
        ])

model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(learning_rate=1e-2), metrics=['cost', 'accuracy', 'validation_accuracy'])

# Train the model
metrics = model.fit(
        X_train, 
        Y_train, 
        validation_X=X_test[:, :200],
        validation_Y=Y_test[:, :200],
        num_epochs=50, 
        batch_size=1024, 
        print_after_steps=10
    )

# Display trend of costs
plt.plot(range(len(metrics['cost'])), metrics['cost'])
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.show()

# Display trend of training and validation accuracy
plt.plot(range(len(metrics['accuracy'])), metrics['accuracy'], 'c-')
plt.plot(range(len(metrics['validation_accuracy'])), metrics['validation_accuracy'], 'm-')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.show()

# Make predictions
Y_predictions = model.predict(X_test)
Y_predictions = np.argmax(Y_predictions, axis=0)
print(Y_predictions)

Y_test = np.argmax(Y_test, axis=0)
accuracy = np.ones_like(Y_predictions)[Y_predictions == Y_test].sum() / len(Y_predictions)
print('Accuracy on test data:', accuracy)