import matplotlib.pyplot as plt
import numpy as np

class View:
    def __init__(self):
        self.iter = 0

    def get_i(self):
        i = self.iter
        self.iter += 1
        return i

    def current_error(self, network, inputs, results):
        errors = []
        print("input | actual | expected")
        for inp, out in zip(inputs, results):
            print(list(inp), network.predict(inp), list(out))
            errors.append(network.get_error(out))
        print("Current average error = {}".format(np.average(errors)))

    def error_sum_per_epoch(self, train_errors, test_errors, label1="Train", label2="Test"):
        plt.figure(self.get_i())
        plt.plot(train_errors, label=label1)
        plt.plot(test_errors, label=label2)
        plt.xlabel("epoch")
        plt.ylabel("error")
        plt.legend()

    def func(self, x, y, x1, y1):
        plt.figure(self.get_i())
        plt.plot(x, y)
        plt.plot(x1, y1)
        plt.xlabel("epoch")
        plt.ylabel("error")
        plt.legend()

    def func_graphics(self, x, y, x_train, y_train, x_test, y_test, x_network, y_network, label=None):
        if label is not None:
            plt.title(label)
        plt.figure(self.get_i())
        plt.plot(x, y, 'g', label="1/log2(x)", zorder=1)
        plt.plot(x_network, y_network, 'm', label="1/log2(x) from network", zorder=2)
        plt.scatter(x_train, y_train, label="Train", zorder=3)
        plt.scatter(x_test, y_test, label="Test", zorder=3)
        plt.legend()

    def rates(self, rates):
        plt.figure(self.get_i())
        plt.plot(rates, label="Rates")
        plt.legend()

    def show(self):
        plt.show()

