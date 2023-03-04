import numpy as np

class LeakyReLU:

    def function(self, z):
        return np.maximum((0.1 * z), z)
    
    def diffFunction(self, dz):
        return np.where(dz > 0, 1.0, 0.1)