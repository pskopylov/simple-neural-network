from layer import *


class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_size, weights=None):
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input = []
        self.errors = self.init_errors()
        self.layers = self.init_layers(weights)

    def init_layers(self, weights):
        if weights is None:
            return [Layer(self.input_size, self.hidden_size),
                    Layer(self.hidden_size, self.output_size)]
        else:
            return [Layer(self.input_size, self.hidden_size, weights[0]),
                    Layer(self.hidden_size, self.output_size, weights[1])]

    def init_errors(self):
        return ErrorLayer(self.output_size)

    def feed_forward(self, input):
        self.input = input
        layer_inputs = [self.input]
        for i in range(len(self.layers)):
            layer_inputs.append(self.layers[i].feed_forward(np.append(layer_inputs[i], 1)))
        return layer_inputs[-1]

    def predict(self, input):
        return self.feed_forward(input)

    def calculate(self, inputs, outputs):
        actual_outputs = self.feed_forward(inputs)
        self.calculate_error(actual_outputs, outputs)

    def calculate_error(self, output, expected_result):
        self.errors.feed_forward(output, expected_result)

    def calculate_deltas(self):
        self.errors.calculate_deltas()
        next_neurons = self.errors.neurons
        for i in range(len(self.layers) - 1, -1, -1):
            self.layers[i].calculate_deltas(next_neurons)
            next_neurons = self.layers[i].neurons

    def back_propagation(self, rate):
        self.calculate_deltas()
        input_values = self.input
        for layer in self.layers:
            layer.refresh_weights(input_values, rate)
            input_values = np.array([neuron.act for neuron in layer.neurons] + [1])

    def get_weights(self):
        return [layer.get_weights() for layer in self.layers]

    def set_wights(self, weights):
        for i in range(len(weights)):
            self.layers[i].set_weights(weights[i])

    def get_error(self, output):
        neurons = self.layers[-1].neurons
        return np.average([(neurons[i].act - output[i])**2 for i in range(len(neurons))])

    def rate_correcting(self, rate, bonus, g_before, g_after):
        if g_before * g_after > 0:
            rate += bonus
        else:
            rate *= (1 - bonus)
        return rate

    def rate_correcting_2(self, rate, decay, it):
        return rate * (1 / (1 + decay * it))

    def rate_correcting_3(self, rate, old_err, err, bonus, threshold):
        res = err / old_err
        if res > 1:
            rate *= (1 - bonus)
        elif res > threshold:
            rate += bonus
        return rate

    def slowdown_rate(self, rate, kf):
        return rate * kf

    def gradient(self):
        return np.average([self.layers[-1].neurons[i].der * self.errors.neurons[i].der
                           for i in range(len(self.layers[-1].neurons))])

    def fit(self, x_train, y_train, x_test, y_test, iterations=10000, rate=0.1, bonus=None, epsilon=0.0005, threshold=0.99995):
        rates = [rate]
        train_errors = []
        test_errors = []
        dynamic_rate = bonus is not None
        changeable_rate = dynamic_rate
        slowdown = False
        prev_err = 1
        for i in range(iterations):
            train_error = 0
            test_error = 0
            for x, y in zip(x_train, y_train):
                self.calculate(x, y)
                train_error += self.get_error(y)
                self.back_propagation(rate)
            for x, y in zip(x_test, y_test):
                self.calculate(x, y)
                test_error += self.get_error(y)
            if test_error / len(y_test) < epsilon and i > iterations / 5:
                print("Epochs = {}".format(i))
                break
            if changeable_rate:
                err = train_error
                rate = self.rate_correcting_3(rate, prev_err, err, bonus, threshold)
                prev_err = err
                rates.append(rate)
            if i == iterations * 0.75:
                slowdown = True
                changeable_rate = False
            if slowdown and dynamic_rate:
                rate = self.slowdown_rate(rate, 1 - bonus / 10)
                rates.append(rate)
            train_errors.append(train_error)
            test_errors.append(test_error)
        return train_errors, test_errors, rates
