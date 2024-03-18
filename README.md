# CS6910
:DEEP_LEARNING_ASSIGNMENT1:

# Problem:
    To implement a feedforward neural network and write the backpropagation code for training the network the Fashion-MNIST dataset by using numpy for all matrix/vector operations,without any automatic differentiation packages.

# Dataset: 
[Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) , [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) datasets

# Objective: 
    A Neural Network will be trained and tested using the Fashion-MNIST dataset. Specifically, given an input image (28 x 28 = 784 pixels) from the Fashion-MNIST dataset, the network will be trained to classify the image into 1 of 10 classes.

# About Datasets: 
    Fashion-MNIST is a dataset of images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes

    Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels

    Each training and test example is assigned to one of the following labels:

0: T-shirt/top

1: Trouser

2: Pullover

3: Dress

4: Coat

5: Sandal

6: Shirt

7: Sneaker

8: Bag

9: Ankle boot

    The MNIST database is, of same configuration as fashion_mnist, of handwritten digits.(0,1,2,3,4,5,6,7,8,9)


# Data Preprocessing:

    Final Training set: shape of (x_train) is (54000 x 784)  ;  shape of (y_train_encode), after one hot encoding, is (54000 x 10)

    Final Validation set: shape of (x_val) is (6000 x 784)  ;  shape of (y_val_encode), after one hot encoding, is (6000 x 10)
    
    Final Test set: shape of (x_test) is (10000 x 784)  ;  shape of (y_test_encode), after one hot encoding, is (10000 x 10)

# The code contains:
    Base Class:
    Class NeuralNetwork which initializes the layer sizes, activation function, weights by constructor. 
    This class does forward propagation method by feedforward function.


    Inheritence Class:
    Inheritance Class backward_optimizer from the base Class NeuralNetwork. 
    The Class backward_optimizer does the backpropagation method by the backward function.

    
    Optimizers:
    Creating SGD,MGD,NAG,RMSPROP,ADAM,NADAM optimizers for training the weights and biases by making different functions in the Class backward_optimizer. 
    
    Finding Loss and Accuracy:
    Defining the loss_accuracy function in the Class backward_optimizer for finding the loss and accuracy after every epoch.

    Training the model:
    The train function to calling the six optimizers one by one and the training will be done by the selected optimizer.

    Predicting the labels(For Plotting Confusion matrix):
    Creating the predict function to help to finding the predicted labels of x_test data for making the confusion matrix

# Hyperparameters:
    Using the sweep functionality provided by wandb to finding the best values for the hyperparameters listed below:

    number of epochs: 5, 10
    number of hidden layers: 3, 4, 5
    size of every hidden layer: 32, 64, 128
    weight decay (L2 regularisation): 0, 0.0005, 0.5
    learning rate: 1e-3, 1 e-4
    optimizer: sgd, momentum, nesterov, rmsprop, adam, nadam
    batch size: 16, 32, 64
    weight initialisation: random, Xavier
    activation functions: sigmoid, tanh, ReLU
    
# For MNIST dataset:
    I have predicted labels for x_test data of mnist dataset by the best three models that I have found by the fashion_mnist dataset.
    
# WANDB Report:
    Have Sweeped with Sweep Count: 439 with Bayes method
[Report](https://wandb.ai/parthasakhapaul/Deep_leraning_A1/reports/MA23M016-PARTHA-SAKHA-PAUL-CS6910-Assignment-1--Vmlldzo3MTgwMjIz)
