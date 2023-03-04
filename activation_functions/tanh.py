import numpy as np

class Tanh:

    def function(self, z):
        return np.tanh(z)
    
    def diffFunction(self, dz):
        return 1.0 / (np.multiply((np.cosh(dz)), np.cosh(dz)))