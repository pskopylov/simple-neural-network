import numpy as np
import math
import time
import cProfile


def sigmoid(args):
    # return 1 / (1 + np.exp(-args))
    return np.array([1 / (1 + math.exp(-arg)) for arg in args])


def sigmoid_derivative(args):
    return args * (1 - args)


class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.weights = [np.random.rand(input_size, hidden_size),
                        np.random.rand(hidden_size, output_size)]

    def calculate(self, inp):
        input_out = sigmoid(np.dot(inp, self.weights[0]))
        hidden_out = sigmoid(np.dot(input_out, self.weights[1]))
        return [inp, input_out, hidden_out]

    def calculate_deltas(self, error, derivatives):
        deltas = [error * derivatives[-1]]
        for i in range(len(self.weights) - 1, 0, -1):
            deltas.append(deltas[-1] * derivatives[i] * self.weights[i])
            deltas.reverse()
        return deltas

    def refresh_weights(self, inputs, error, rate):
        delta = error * sigmoid_derivative(inputs[-1])
        for i in range(len(self.weights) - 1, -1, -1):
            old_delta = delta
            delta = sigmoid_derivative(inputs[i]) * np.dot(self.weights[i], delta)
            self.weights[i] -= rate * old_delta * inputs[i].reshape(-1, 1)

    def predict(self, inp):
        return self.calculate(inp)[-1]

    def fit(self, inputs, outputs, rate, iterations):
        for i in range(iterations):
            for inp, out in zip(inputs, outputs):
                input_out = sigmoid(np.dot(inp, self.weights[0]))
                hidden_out = sigmoid(np.dot(input_out, self.weights[1]))
                layers_outputs = [inp, input_out, hidden_out]

                error = layers_outputs[-1] - out
                delta = error * sigmoid_derivative(layers_outputs[-1])
                for i in range(len(self.weights) - 1, -1, -1):
                    old_delta = delta
                    delta = sigmoid_derivative(layers_outputs[i]) * np.dot(self.weights[i], delta)
                    self.weights[i] -= rate * old_delta * layers_outputs[i].reshape(-1, 1)

    # def fit(self, inputs, outputs, rate, iterations):
    #     for i in range(iterations):
    #         for inp, out in zip(inputs, outputs):
    #             layers_outputs = self.calculate(inp)
    #             error = layers_outputs[-1] - out
    #             self.refresh_weights(layers_outputs, error, rate)


if __name__ == '__main__':
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    n = NeuralNetwork(2, 1, 2)
    start_time = time.time()
    # cProfile.run("n.fit(x, y, 0.1, 100000)")
    n.fit(x, y, 0.1, 100000)
    print("--- %s seconds ---\n" % (time.time() - start_time))
    for input in x:
        print(list(input), n.predict(input))

