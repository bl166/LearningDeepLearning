"""
Write your own n layer neural network.py that builds and trains a neural network of n layers. 
Your code must be able to accept as parameters 

(1) the number of layers and 
(2) layer size. 

We provide you hints below to help you organize and implement the code, but if you have better ideas, please
feel free to implement them and ignore our hints. In your report, please tell us why you made the choice(s) you did.


Hints:

1. Create a new class, e.g DeepNeuralNetwork, that inherits NeuralNetwork in three layer neural network.py

2. In DeepNeuralNetwork, change function feedforward, backprop, calculate loss and fit model

3. Create a new class, e.g. Layer(), that implements the feedforward and 
back-prop steps for a single layer in the network

4. Use Layer.feedforward to implement DeepNeuralNetwork.feedforward

5. Use Layer.backprop to implement DeepNeuralNetwork.backprop         

6. Notice that we have L2 weight regularizations in the final loss function in addition to the cross entropy. 
Make sure you add those regularization terms in DeepNeuralNetwork.calculate loss and their derivatives in 
DeepNeuralNetwork.fit model.

Train your network on the Make Moons dataset using different number of layers, different layer sizes, 
different activation functions and, in general, different network configurations. In your report, 
include generated images and describe what you ob- serve and what you find interesting 
(e.g. decision boundary of deep vs shallow neural networks).

Next, train your network on another dataset different from Make Moons. You can choose datasets provided 
by Scikit-learn (more details here) or any dataset of your interest. Make sure that you have the correct 
number of input and output nodes. Again, play with different network configurations. In your report, 
describe the dataset you choose and tell us what you find interesting.

Be curious and creative!!! You are exploring Deep Learning. :)
"""

import three_layer_neural_network as nn
import numpy as np
import matplotlib.pyplot as plt


class Activation(object):
    def __init__(self, type):
        self.type = type

    def activ(self, z):
        '''
        actFun computes the activation functions
        :param type: Tanh, Sigmoid, or ReLU
        :return: activation functions
        '''

        # YOU IMPLMENT YOUR actFun HERE
        if self.type.lower()=="tanh":
            return np.tanh(z)
        elif self.type.lower()=="sigmoid":
            return 1./(1+np.exp(-z))
        elif self.type.lower()=="softmax":
            x = z-z.max(axis=0)
            return np.exp(x) / np.exp(x).sum(axis=0)
        elif self.type.lower()=="relu":
            return z*(z>0)
        else:
            self.type = 'none'
            return z

    def d_activ(self, z):
        '''
        diff_actFun computes the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''

        # YOU IMPLEMENT YOUR diff_actFun HERE
        if self.type.lower()=="tanh":
            return 1 - np.square(np.tanh(z))
        elif self.type.lower()=="sigmoid":
            return 1/(1+np.exp(-z))*(1-1/(1+np.exp(-z)))
        elif self.type.lower()=="softmax":
            return 1/len(z)
        elif self.type.lower()=="relu":
            return 1*(z>0)
        else:
            self.type = 'none'
            return np.ones_like(z)


class DeepNeuralNetwork(nn.NeuralNetwork):
    def __init__(self, layers_config, activ_config, loss, epsilon = 1e-2, reg_lambda = 1e-2, seed = 0):
        """
        :param layers_config:  (input, hidden, ..., output) list of layers dimensions
        :param activ_config:   ('tahn', ..., 'sigmoid') list of activation types
        """
        self.n_layers = len(layers_config) # number of layers + 1(input)
        assert self.n_layers == len(activ_config)+1, "Wrong layers / activ config"

        self.loss = loss
        self.epsilon = epsilon
        self.reg_lambda = reg_lambda

        # Weights and biases are initiated by index. For a one hidden layer net you will have a w[1] and w[2]
        self.W, self.b = {}, {}

        # Activation are also initiated by index. For the example we will have activ_config[2] and activ_config[3]
        np.random.seed(seed)
        self.activfunc = {}
        for i in range(1, len(layers_config)):
            self.W[i] = np.random.randn(layers_config[i-1], layers_config[i]) / np.sqrt(layers_config[i-1])
            self.b[i] = np.zeros((1, layers_config[i]))
            self.activfunc[i + 1] = Activation(activ_config[i-1])


    def feedforward(self, x):
        '''
        feedforward builds an {arbitrary number}-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :return:
        '''

        # store results in dict, keys being layer ibdices
        z = {}      # w * x + b
        a = {1: x}  # 1st layer output is just the input

        for i in range(1, self.n_layers):
            z[i+1] = a[i] @ self.W[i] + self.b[i]
            a[i+1] = self.activfunc[i+1].activ(z[i+1])
        return z, a


    def backprop(self, z, a, y):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return:
        '''

        # for final layer
        delta = (a[self.n_layers] - y) * self.activfunc[self.n_layers].d_activ(a[self.n_layers])
                        
        update_params = {self.n_layers-1: (
                            a[self.n_layers - 1].T @ delta, # dw
                            np.mean(delta, axis=0)          # db
                        )}

        # start from the final-1 layer and stop at 2nd layer (1st being X and does not update)
        for i in range(2, self.n_layers)[::-1]:
            # new delta
            delta = delta @ self.W[i].T * self.activfunc[i].d_activ(z[i])
            dW = a[i - 1].T @ delta
            db = np.mean(delta, axis=0)

            # Add regularization terms (b's don't have regularization terms)
            dW += self.reg_lambda * self.W[i-1]

            update_params[i-1] = (dW, db)

        # update w and b
        for k, v in update_params.items():
            self.W[k] += -self.epsilon * v[0]
            self.b[k] += -self.epsilon * v[1]


    def fit_model(self, x, y, num_passes, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        assert x.shape[0] == y.shape[0], "Length error!"

        for i in range(num_passes):
            # forward and backward
            z, a = self.feedforward(x)
            self.backprop(z, a, y)

            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(y, self.predict(x))))


    def calculate_loss(self, y, prob):
        """
        :param y:    one-hot ground truth
        :param prob: predicted probabilities
        :return:
        """
        if self.loss.lower() == 'xent':
            data_loss = -np.sum(y * np.log(prob))
        elif self.loss.lower() == 'mse':
            data_loss = np.sum((y - prob)**2)

        return (1 / len(y)) * data_loss


    def predict(self, x):
        """
        :param x: input
        :return: predicted probabilities of shape (n_cases, n_classes)
        """
        _, a = self.feedforward(x)
        return a[self.n_layers]


    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plots the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        # plot_decision_boundary(lambda x: self.predict(x), X, y)

        # Set min and max values and give it some padding
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole gid
        Z = np.argmax(self.predict(np.c_[xx.ravel(), yy.ravel()]), axis=1)
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.figure(dpi=150)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
        plt.show()


def main():
    # # generate and visualize Make-Moons dataset
    X, y = nn.generate_data()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

    model = DeepNeuralNetwork(layers_config = (2, 3, 3, 2),
                              activ_config = ('tanh', 'tanh', 'Sigmoid'),
                              loss = 'xent',
                              epsilon = 1e-3,
                              reg_lambda = 1e-2)
    model.fit_model(X, np.eye(2)[y], num_passes=20000)
    model.visualize_decision_boundary(X, y)

    
if __name__ == "__main__":
    main()

