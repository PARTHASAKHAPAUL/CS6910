# CS6910
:DEEP_LEARNING_ASSIGNMENT1:

    To implement a feedforward neural network and write the backpropagation code for training the network the Fashion-MNIST dataset by using numpy for all matrix/vector operations,without any automatic differentiation packages.

# Dataset: 
    Fashion-MNIST, MNIST datasets

# Objective: 
    A Neural Network will be trained and tested using the Fashion-MNIST dataset. Specifically, given an input image (28 x 28 = 784 pixels) from the Fashion-MNIST dataset, the network will be trained to classify the image into 1 of 10 classes.

# Data Preprocessing:

    Final Training set: shape of (x_train) is (54000 x 784)  ;  shape of (y_train_encode), after one hot encoding, is (54000 x 10)

    Final Validation set: shape of (x_val) is (6000 x 784)  ;  shape of (y_val_encode), after one hot encoding, is (6000 x 10)
    
    Final Test set: shape of (x_test) is (10000 x 784)  ;  shape of (y_test_encode), after one hot encoding, is (10000 x 10)

# The code contains:
    Base Class:
    Class NeuralNetwork which initializes the layer_sizes, activation_func, weight_init by constructor. 
    This class has forward propagation method by feedforward function.


    Inheritence Class:
    Inheritance Class backward_optimizer from the base Class NeuralNetwork. 
    The Class backward_optimizer does the backpropagation method by the backward function.

    
    Optimizers:
    SGD,MGD,NAG,RMSPROP,ADAM,NADAM by making different functions in the Class backward_optimizer. 

And then I made:

    the loss_accuracy function for finding the loss and accuracy after every epoch
    
    the train function to calling the six optimizers one by one
    
    the predict function to help to finding the predicted labels of x_test data for making the confusion matrix

And for the last question 10: 

    same thing that I have done for the fashion_mnist dataset, I have predicted for mnist dataset by the three best model that I have found by the fashion_mnist dataset.



#  For Wandb Sweep : [link](https://docs.wandb.ai/guides)
