import numpy as np

# sigmoid function to normalize inputs
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid derivatives to adjust synaptic weights
def sigmoid_derivative(x):
    return x * (1 - x)

# input dataset
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

# output dataset
training_outputs = np.array([[0,1,1,0]]).T

# seed random numbers to make calculation
np.random.seed(1)

# initialize weights randomly with mean 0 to create weight matrix, synaptic weights
synaptic_weights = 2 * np.random.random((3,1)) - 1

for iteration in range(20000):
    input_layer = training_inputs
    outputs  = sigmoid(np.dot(input_layer, synaptic_weights))

    error = training_outputs - outputs

    adj = error * sigmoid_derivative(outputs)
    synaptic_weights += np.dot(input_layer.T, adj)

print('Synapitic Weights after training: ')
print(synaptic_weights)
print('Outputs: ')
print(outputs)