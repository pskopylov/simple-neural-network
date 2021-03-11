import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class AbstractNeuron:
    def __init__(self, weights):
        self.weights = weights
        self.act = 0
        self.der = 0
        self.delta = 0


class Error(AbstractNeuron):
    def __init__(self, weights):
        super().__init__(weights)

    def feed_forward(self, input, result):
        self.act = 1 / 2 * (input - result) ** 2
        self.der = input - result

    def calculate_delta(self):
        self.delta = self.der


class Neuron(AbstractNeuron):
    def __init__(self, weights):
        super().__init__(weights)

    def feed_forward(self, inputs):
        self.act = sigmoid(np.dot(self.weights, inputs))
        self.der = self.act * (1 - self.act)
        return self.act

    def calculate_delta(self, input_delta, output_weights):
        self.delta = self.der * np.dot(input_delta, output_weights)

    def refresh_weights(self, input, rate):
        self.weights -= rate * input * self.delta

    def set_weights(self, weights):
        self.weights = weights
