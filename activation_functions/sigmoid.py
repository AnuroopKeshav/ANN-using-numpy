import numpy as np

class Sigmoid:

    def function(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def diffFunction(self, dz):
        return np.multiply((1.0 / (1.0 + np.exp(-dz))), (1.0 - (1.0 / (1.0 + np.exp(-dz)))))