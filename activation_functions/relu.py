import numpy as np

class ReLU:

    def function(self, z):
        return np.maximum(0, z)
    
    def diffFunction(self, dz):
        return np.where(dz > 0, 1.0, 0)