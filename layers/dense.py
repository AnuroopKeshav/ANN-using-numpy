import numpy as np
import activation_functions

class Dense:

    # Initializing the layer (number of perceptrons and activation function)
    def __init__(self, n_pereptrons, activation_function=activation_functions.Sigmoid()) -> None:
        self.n_pereptrons = n_pereptrons
        self.activation_function = activation_function
    
    # Feed forward
    def feedForward(self, X, is_test=False):
        self.X = X
        self.features, self.entries = self.X.shape

        # Check for first iteration
        try:
            self.weights
            self.bias
        except:
            # Initializing weight and bias matrices if first iteration
            self.weights = np.random.rand(self.n_pereptrons, self.features) - 0.5
            self.bias = (np.random.rand(self.n_pereptrons) - 0.5).reshape(-1, 1)

        # Linear combination and activation function
        self.z = np.dot(self.weights, self.X) + self.bias
        self.a = self.activation_function.function(self.z)

        return self.weights, self.z, self.a, self.activation_function
    
    # Back propagate
    def backPropagate(self, dz) -> None:
        self.dz = dz

        # Derivatives of dz with respect to weights and biases
        self.dw = np.dot(self.dz, self.X.T) / self.entries
        self.db = np.sum(self.dz, axis=1).reshape(-1, 1) / self.entries

    # Gradient descent
    def adjust(self, learning_rate) -> None:
        self.weights = self.weights - (learning_rate * self.dw)
        self.bias = self.bias - (learning_rate * self.db)