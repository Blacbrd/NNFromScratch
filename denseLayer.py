# Each row represents what weights are connected from an input neurone to the rest of the output neurones
# So the amount of rows is the amount of input neurones
# Each column represents all the weights connecting to one neurone
# So the amount of columns is the amount of output neurones

import numpy as np

class DenseLayer:

    # (num input, num output)
    def __init__(self, numInputs, numNeurones):
        
        self.weights = np.random.randint(numInputs, numNeurones)
        
        # Zeros makes every item in the array a zero
        self.bias = np.zeros(1, numNeurones)
    
    def forward(self, inputs):

        # This essentially does what we were doing by hand in the previous file
        # This output will be used as an input for the next layer
        self.output = np.dot(inputs, self.weights) + self.bias

inputs = np.random.randint(1, 5)
dense1 = DenseLayer(1, 5)
dense2 = DenseLayer(5, 1)

# This is how to chain layers together
dense2.forward(dense1.forward(inputs))