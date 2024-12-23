# np.dot() for dot product

# For two vectors, lets say [a1, a2], [b1, b2]
# np.dot(a, b) = (a1*b1 + a2*b2)

# For matrices:
# Matrix multiplication can only occur if the first matrix's column number is the same as the second's row number
# 1x2 , 2x3  ->  valid, and output will have dimensions 1x3

# [1, 2, 3] x [[4, 5, 6],
#               7, 8, 9],
#               10, 11, 12]]

# Take the dot product of the rows and columns, so the first number will be (1*4 + 2*7 + 3*10)
# You have to remember to watch out when doing np.dot(a, b) or np.dot(b, a), as they are not the same

# Weights are labeled wji where i is the neurone number on the left and j is the neurone number on the right
# So the 2nd neurone going to the 4th neurone would be w42

import numpy as np

# Notice how this will give you the same value as the previous neurone
def exampleNeurone():

    weights = [0.2, -0.1, 0.7]
    inputs = [1, 2, 3]
    bias = 0.5

    return np.dot(inputs, weights) + bias

def exampleNeuroneLayer():

    inputs = [[1, 2, 3, 4],
              [11, 22, 33, 44]]

    weights = [[0.2, 0.4, -0.7, 0.9],
               [0.5, 0.9, 0.4, 0.3],
               [-0.4, -0.9, 0.9, 0.8],
               [0.5, 0.9, 0.4, 0.3]]
    
    biases = [0.3, 0.4, 0.5, 0.6]

    return np.dot(inputs, np.array(weights).T) + biases

# For this, I want to see if writing it in wij is better than wji
def neuroneLayerDiffNotation():

    inputs = [[1, 2, 3, 4],
              [11, 22, 33, 44]]

    weights = [[0.2, 0.4, -0.7, 0.9],
               [0.5, 0.9, 0.4, 0.3],
               [-0.4, -0.9, 0.9, 0.8],
               [0.5, 0.9, 0.4, 0.3]]

    biases = [0.3, 0.4, 0.5, 0.6]

    return  np.dot(weights, np.array(inputs).T) + np.array(biases).reshape(-1, 1)


x = exampleNeuroneLayer()

print(x)

y = neuroneLayerDiffNotation()

print(y)

# I've come to the conclusion that wji is better, as each output is represented as a row, and no reshaping is required