import numpy as np

class Sequential:

    # List and count of layers
    layers = []
    layer_count = 0

    # List of input matrix for test
    A_test = []

    # Initializing the learning rate and the epochs
    def __init__(self, learning_rate=0.1, epochs=10) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs

    # Adding layers to the layer model
    def add(self, layer) -> None:
        self.layers.append(layer)
        self.layer_count += 1
    
    # Fitting the model
    def fit(self, X_train, y_train):

        for epoch in range(self.epochs):
            counter = 0

            # For every single entry from the dataset
            for X, y in zip(X_train, y_train):
                self.W = []
                self.Z = []
                self.A = []
                self.FN = []
                self.DZ = []

                X = X.reshape(1, -1)
                y = y.reshape(1, -1)

                self.A.append(X.T)

                # Feed Forward
                for i in self.layers:
                    a = self.A[-1]
                    w, z, a, fn = i.feedForward(a)
                    self.W.append(w)
                    self.Z.append(z)
                    self.A.append(a)
                    self.FN.append(fn)

                # Accuracy calculation
                counter += np.sum(np.argmax(self.A[-1], axis=0) == np.argmax(y.T, axis=0))

                self.DZ.append(self.A[-1] - y.T)
                self.A.pop(-1)
                self.Z.pop(-1)
                self.layers.reverse()

                # Making a list of DZ for all the layers
                for i in range((self.layer_count - 1), 0, -1):
                    dz = np.multiply(np.dot(self.W[i].T, self.DZ[-1]), self.FN[i].diffFunction(self.Z[i - 1]))
                    self.DZ.append(dz)

                # Back Propagation
                self.layers[0].backPropagate(self.DZ[0])

                for i in range(1, self.layer_count):
                    self.layers[i].backPropagate(self.DZ[i])

                # Adjusting all the weights and biases
                for i in self.layers:
                    i.adjust(self.learning_rate)

                # Undoing the list of layers for the next iteration
                self.layers.reverse()
                del self.W, self.Z, self.A, self.FN, self.DZ

            # Printing the accuracy of the model in the current epoch
            print(f"EPOCH {epoch + 1}\nAccuracy: {counter / y_train.shape[0]}\n\n")

    # Predict
    def predict(self, X_test):
            self.A_test.append(X_test.T)
            
            # Forward feed with the current set of optimal weights and biases
            for i in self.layers:
                a = self.A_test[-1]
                w, z, a, fn = i.feedForward(a)
                self.A_test.append(a)

            return self.A_test[-1]