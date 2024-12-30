# Implementing activation functions allows us to branch out from linear functions
# Since originally we're just doing xw + b (linear)

import numpy as np

class Dense:

    def __init__(self, inputs, numberNuerones):
        self.weights = np.random.randn(numberNuerones, inputs)

        self.biases = np.zeros((1, numberNuerones))
    
    def forward(self, inputs):

        self.output = np.dot(inputs, self.weights)
        return self.output

class ReLU:

    def forward(self, inputs):

        # This removes all array items that are less than 0
        # so [-1, 3, -4, 5.5, 6] becomes
        # [3, 5.5, 6]
        return np.maximum(0, inputs)

class Softmax:

    def forward(self, inputs):

        # What this does is it normalises the outputs to be between 0 and 1
        # It uses the equation e^x / e^x + e^y + e^z
        # More e's added if more neurones present

        # np.exp does e^value for every item
        # The reason we do inputs - np.max(inputs) is to normalise the data/keep it small
        # Since if we have rediculously big or small numbers, they may cause problems
        # We keep dimensions to avoid padding, and axis = 1 for rows
        expValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # When using division in numpy, it divides each item in the array by that amount
        probabilities = expValues / np.sum(expValues, axis=1, keepdims=True)