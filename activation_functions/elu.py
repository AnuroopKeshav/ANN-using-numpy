import numpy as np

class ELU:

    def __init__(self, alpha: float):
        if alpha <= 0:
            raise Exception("The value for alpha must be a positive number")
        else:
            self.alpha = alpha

    def function(self, z):
        return z if z >= 0 else self.alpha * (np.exp(z) - 1)
    
    def diffFunction(self, dz):
        return 1 if dz > 0 else self.alpha * np.exp(dz)