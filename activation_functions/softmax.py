import numpy as np

class Softmax:

    def function(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=0)
    
    def diffFunction(self, dz):
        return np.multiply((np.exp(dz) / sum(np.exp(dz))), (1 - (np.exp(dz) / sum(np.exp(dz)))))