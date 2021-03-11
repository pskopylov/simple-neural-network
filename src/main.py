from network import NeuralNetwork
from view import View
import numpy as np
import time
from threading import Thread
import multiprocessing as mp
from sklearn.model_selection import train_test_split


def add_noise(args):
    return [[e] for e in np.random.normal(0, 0.02, len(args))] + args


def func(args):
    return 1 / np.log2(args)


def func_sin(args):
    return np.abs(np.sin(args))


def main():
    x = np.array([[x] for x in np.arange(2, 30)])
    y = func(x)
    noised_y = add_noise(y)
    x_train, x_test, y_train, y_test = train_test_split(x, noised_y, test_size=0.3, random_state=0)
    network = NeuralNetwork(input_size=x_train.shape[1], output_size=y_train.shape[1], hidden_size=2)

    start_time = time.time()
    train_errors, test_errors, rates = network.fit(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                                   rate=0.5, bonus=0.005, iterations=10000)
    print("--- %s seconds ---\n" % (time.time() - start_time))

    x_network = x
    y_network = [network.predict(xi) for xi in x]
    print("Train:")
    view = View()
    view.current_error(network, x_train, y_train)
    print("\nTest:")
    view.current_error(network, x_test, y_test)
    view.error_sum_per_epoch(train_errors, test_errors)
    view.error_sum_per_epoch(train_errors, test_errors)
    view.func_graphics(x, y, x_train, y_train, x_test, y_test, x_network, y_network)
    view.rates(rates)
    view.show()


def compare():
    rate = 0.1
    it = 10000
    x = np.array([[x] for x in np.arange(2, 30)])
    y = func(x)
    noised_y = add_noise(y)
    x_train, x_test, y_train, y_test = train_test_split(x, noised_y, test_size=0.3, random_state=0)
    network = NeuralNetwork(input_size=x_train.shape[1], output_size=y_train.shape[1], hidden_size=2)
    network_r = NeuralNetwork(input_size=x_train.shape[1], output_size=y_train.shape[1], hidden_size=2,
                              weights=network.get_weights())
    start_time = time.time()
    tr_e, ts_e, rt = network.fit(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, rate=rate, iterations=it)
    tr_e_r, ts_e_r, rt_r = network_r.fit(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, rate=rate, iterations=it, bonus=0.005)
    print("--- %s seconds ---\n" % (time.time() - start_time))

    x_network = x
    y_network = [network.predict(xi) for xi in x]
    y_network_r = [network_r.predict(xi) for xi in x]
    view = View()
    view.func_graphics(x, y, x_train, y_train, x_test, y_test, x_network, y_network, "Learning with static rate")
    view.func_graphics(x, y, x_train, y_train, x_test, y_test, x_network, y_network_r, "Learning with dynamic rate")
    view.error_sum_per_epoch(tr_e, tr_e_r, "static rate", "dynamic rate")
    view.func(x_network, y_network, x_network, y_network_r)
    view.rates(rt_r)
    view.show()


if __name__ == '__main__':
    compare()
    # main()
