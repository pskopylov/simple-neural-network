from neuron import Neuron, Error
import numpy as np


class Layer:
    def __init__(self, input_size, size, weights=None):
        if weights is None:
            self.neurons = [Neuron(np.random.rand(input_size + 1) - 0.5) for _ in range(size)]
        else:
            self.neurons = [Neuron(weights[i]) for i in range(size)]

    def feed_forward(self, inputs):
        return [n.feed_forward(inputs) for n in self.neurons]

    def calculate_deltas(self, next_neurons):
        weights = np.array([neuron.weights for neuron in next_neurons]).T
        deltas = np.array([neuron.delta for neuron in next_neurons])
        for i in range(len(self.neurons)):
            self.neurons[i].calculate_delta(deltas, weights[i])

    def refresh_weights(self, input, rate):
        for neuron in self.neurons:
            neuron.refresh_weights(input, rate)

    def get_weights(self):
        return [neuron.weights.copy() for neuron in self.neurons]

    def set_weights(self, weights):
        for i in range(weights):
            self.neurons[i].set_weights(weights[i])

    def get_der(self):
        return [neuron.der for neuron in self.neurons]


class ErrorLayer:
    def __init__(self, size):
        self.neurons = [Error(np.ones(size)) for _ in range(size)]

    def feed_forward(self, input, result):
        for i in range(len(self.neurons)):
            self.neurons[i].feed_forward(input[i], result[i])

    def calculate_deltas(self):
        for neuron in self.neurons:
            neuron.calculate_delta()

    def get_der(self):
        return [neuron.der for neuron in self.neurons]
