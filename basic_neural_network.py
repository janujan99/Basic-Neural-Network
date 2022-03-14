import random
import math
import numpy as np

RANGE = 10


class Network:
    def __init__(self, weights, biases) -> None:
        self.weights = weights
        self.biases = biases


def activation(z):
    return z


def activation_prime(z):
    return 1


def create_data(epoch, test_data=False):
    w1 = float(random.randint(-1 * RANGE, 10))
    w2 = float(random.randint(-RANGE, RANGE))
    b1 = 0.0
    b2 = 0.0
    training_data = []
    for _ in range(epoch):
        a1 = float(random.randint(-RANGE, RANGE))
        a2 = activation(w1*a1+b1)
        # add (input, output) tuples to training_data
        training_data.append((a1, activation(w2*a2+b2)))

    return training_data, w1, b1, w2, b2


# get training and test data
epoch = 10000
training_data, actual_w1, actual_b2, actual_w2, actual_b3 = create_data(epoch)
# fit training data
iterations = 1
eta = 0.2
num_layers = 3
weights = [float(random.randint(-RANGE, RANGE)) for x in range(num_layers-1)]
biases = [float(random.randint(-RANGE, RANGE)) for x in range(num_layers-1)]

#print(f"actual weights: {actual_w1} {actual_w2} Weights: {weights}")
#print(f"actual biases: {actual_b2} {actual_b3} Biases: {biases}")

sample = (5, 206)
first_example = training_data[0]
inp = sample[0]
out = sample[1]
# print(first_example)
weights = [8.0, 41.0]
biases = [0, 0]
# for inp, out in training_data:
for _ in range(1):
    z = [None]*num_layers
    a = [inp] + [None]*(num_layers-1)
    dC_daI = [None]*(num_layers)
    dC_dwI = [None]*(num_layers-1)
    dC_dbI = [None]*(num_layers-1)
    # feed forward
    for index in range(num_layers-1):
        z[index+1] = weights[index]*a[index] + biases[index]
        a[index+1] = activation(z[index+1])
    cost1 = (a[-1] - out)**2
    print("z: ", z)
    print("a: ", a)
    # back propagate
    for index in range(num_layers-1)[::-1]:
        print("Iteration: ", index)
        if index == num_layers-2:
            dC_daI[index+1] = 2*(a[-1] - out)
        else:
            daI_dzI = activation_prime(z[index + 1])
            dC_daI[index+1] = dC_daI[index+2] * daI_dzI * weights[index+1]

        print("Activatoin prime: ", activation_prime(z[index + 1]))
        print(dC_daI[index+1])
        print("a_i, ", a[index])
        dC_dwI[index] = dC_daI[index+1] * \
            activation_prime(z[index + 1])*a[index]
        dC_dbI[index] = dC_daI[index+1]*activation_prime(z[index + 1])*1

    temp_weights = weights
    temp_biases = biases
    weights = [weights[x] - eta*dC_dwI[x] for x in range(len(weights))]
    biases = [biases[x] - eta*dC_dbI[x] for x in range(len(biases))]
    print("New weights: ", weights)
    print("New biases: ", biases)
    for index in range(num_layers-1):
        z[index+1] = weights[index]*a[index] + biases[index]
        a[index+1] = activation(z[index+1])
    cost2 = (a[-1] - out)**2
    print("Cost change(loop): ", cost2-cost1)


# ------------------------------------
print(f"actual weights: {actual_w1} {actual_w2} Weights: {weights}")
print(f"actual biases: {actual_b2} {actual_b3} Biases: {biases}")
print("dC/dwI: ", dC_dwI)
print("dC/dbI: ", dC_dbI)
print("dC/daI: ", dC_daI)
