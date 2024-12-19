# Weights are how much a neurone affects the next neurone
# Bias allows our neurones to be more flexible

def neuroneExample():

    x = [1, 2, 3]
    w = [0.2, -0.1, 0.7]
    b = 0.5

    output = 0
    for i in range(3):

        output += (x[i] * w[i])
    
    return output + b

def neuroneLayerExample():

    inputs = [1, 2, 3, 4]

    weights = [[0.2, 0.4, -0.7, 0.9],
               [0.5, 0.9, 0.4, 0.3],
               [-0.4, -0.9, 0.9, 0.8]]
    
    bias = [0.3, 0.4, 0.5, 0.6]

    output = 0
    for i in range (len(weights)):
        for j in range(len(inputs)):

            output += (inputs[i] * weights[i][j])
        
        output += bias[i]
    
    return output

e = neuroneExample()


print(e)

