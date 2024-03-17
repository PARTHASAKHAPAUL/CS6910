# CS6910
DEEP_LEARNING:::

Question:
In this assignment I need to implement a feedforward neural network and write the backpropagation code for training the network the Fashion-MNIST dataset by using numpy for all matrix/vector operations,without any automatic differentiation packages. This network will be trained and tested using the Fashion-MNIST dataset. Specifically, given an input image (28 x 28 = 784 pixels) from the Fashion-MNIST dataset, the network will be trained to classify the image into 1 of 10 classes.



Answer:
I have taken:

Final Training set: shape of (x_train) is (54000 x 784)  ;  shape of (y_train_encode), after one hot encoding, is (54000 x 10)

Final Validation set: shape of (x_val) is (6000 x 784)  ;  shape of (y_val_encode), after one hot encoding, is (6000 x 10)

Final Test set: shape of (x_test) is (10000 x 784)  ;  shape of (y_test_encode), after one hot encoding, is (10000 x 10)

I have made the Class NeuralNetwork which initializes the layer_sizes, activation_func, weight_init by constructor. Then I have made the forward propagation in that Class by feedforward function.

And then I have made the inheritance Class backward_optimizer from the base Class NeuralNetwork. 

The Class backward_optimizer does the backpropagation method by the backward function and I have used the optimizers SGD,MGD,NAG,RMSPROP,ADAM,NADAM by making different functions in the Class backward_optimizer. 

And then I made the loss_accuracy, train, predict functions for finding the loss_accuracy, calling the optimizers, finding the confusion matrix respectively.

And for the last question 10: same thing I have done for the mnist dataset as I have done for fashion_mnist dataset.
