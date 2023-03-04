# import numpy as np

# class Dropout:
    
#     # Initializing the dropout layer with a probability parameter input
#     def __init__(self, p: float) -> None:
#         self.p = p

#     # Defining a function to return a list with specified probability distribution 
#     def _getBool(self):
#         bool_list = [True] * int((1 - self.p) * 100) + [False] * int(self.p  * 100)
#         return np.random.choice(bool_list)

#     # Feed forward
#     def feedForward(self, X, is_test=False):
#         self.X = X
#         if not is_test:
#             self.features, self.entries = self.X.shape
#             self.weights = np.where(self._getBool(), np.ones((self.entries, self.features)), 0)
#         else:
#             return self.X

import numpy as np

class Dropout:

    def __init__(self, p: float) -> None:
        self.p = p
    
    def _random_list(self):
        pass

    def feedForward(self, X):
        self.X = X
        self._random_list()