# When the input is represented as a matrix, you can think of each row as being a batch of data

# Below I am going to make a neural network with these parameters:
# 4 input neurones (with 3 batches of data)
# A hidden layer with 3 neurones
# Another hidden layer with 2 neurones
# An output layer with 1 neurone

# (3 rows, 4 colums)
import numpy as np

def multipleLayers():

    inputs = np.random.rand(3, 4)

    weights1 = np.random.rand(3, 4)
    bias1 = np.random.rand(1, 3)

    weights2 = np.random.rand(2, 3)
    bias2 = np.random.rand(1, 2)

    weights3 = np.random.rand(1, 2)
    bias3 = np.random.rand(1, 1)

    output1 = np.dot(inputs, np.array(weights1).T) + bias1
    output2 = np.dot(output1, np.array(weights2).T) + bias2
    output3 = np.dot(output2, np.array(weights3).T) + bias3

    # Since I put in 3 batches, I will end up getting 3 results
    print(output3)

multipleLayers()