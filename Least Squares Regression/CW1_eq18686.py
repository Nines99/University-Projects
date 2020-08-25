from __future__ import print_function  # to avoid issues between Python 2 and 3 printing

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

# By default we set figures to be 6"x4" on a 110 dots per inch (DPI) screen
# (adjust DPI if you have a high res screen!)

plt.rc('figure', figsize=(6, 4), dpi=110)
plt.rc('font', size=10)


def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()


# Calculates the squared error between two variables
def square_error(y, y_hat):
    return np.sum((y - y_hat) ** 2)


# Returns squared error, model parameters and predicted y values generated from an order 3 polynomial function.
def poly3(xs, ys):
    ones = np.ones(xs.shape)
    x2 = np.multiply(xs, xs)
    x3 = np.multiply(x2, xs)
    poly_x = np.column_stack((ones, xs, x2, x3))
    A = np.linalg.inv(poly_x.T.dot(poly_x)).dot(poly_x.T).dot(ys)
    a0 = A[0]
    a1 = A[1]
    a2 = A[2]
    a3 = A[3]
    predicted_ys = a0 + a1 * xs + a2 * x2 + a3 * x3
    error = square_error(ys, predicted_ys)
    return [error, a0, a1, a2, a3, predicted_ys]


# Returns squared error, model parameters and predicted y values generated from a liner function.
def linear(xs, ys):
    ones = np.ones(xs.shape)
    linear_xs = np.column_stack((ones, xs))
    A = np.linalg.inv(linear_xs.T.dot(linear_xs)).dot(linear_xs.T).dot(ys)
    a = A[0]
    b = A[1]
    linear_ys = a + b * xs
    error = square_error(ys, linear_ys)
    return [error, a, b]


# Returns squared error, model parameters generated from a sin function.
def sin(xs, ys):
    X = np.column_stack((np.ones(xs.shape), np.sin(xs)))
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(ys)
    a = A[0]
    b = A[1]
    predicted_ys = a + b * np.sin(xs)
    error = square_error(ys, predicted_ys)
    return [error, a, b]


# Return a list containing all returned values from executing all functions and
# a list of all the errors generated from all the different function
def lse(x_i, y_i):
    lin = linear(x_i, y_i)
    cubic = poly3(x_i, y_i)
    sins = sin(x_i, y_i)
    return np.array([[lin[0], cubic[0], sins[0]], lin, cubic, sins])


# Splits data points into segments of 20, then calculates which function would generate the smallest error, then plots a
# graph corresponding to most accurate for each segment and then prinitng sum error from plotted points and real points.
def regression(xs, ys):
    no_seg = int(xs.size / 20)
    xs_segment = np.split(xs, no_seg)
    ys_segment = np.split(ys, no_seg)
    sum_error = 0
    fig, ax = plt.subplots()
    ax.scatter(xs, ys, s=50)
    for x, y in zip(xs_segment, ys_segment):
        x_i = np.array(x)
        y_i = np.array(y)
        errors = np.array(lse(x_i, y_i)[0])
        best_fit = cross_validation(x_i, y_i)
        sum_error += errors[best_fit]
        plot(x_i, y_i, best_fit, ax)
    print(sum_error)
    ax.legend()
    plt.show()


# Same as regression function but doesn't plot and only prints summed error.
def regression_no_plot(xs, ys):
    no_seg = int(xs.size / 20)
    xs_segment = np.split(xs, no_seg)
    ys_segment = np.split(ys, no_seg)
    sum_error = 0
    for x, y in zip(xs_segment, ys_segment):
        x_i = np.array(x)
        y_i = np.array(y)
        errors = np.array(lse(x_i, y_i)[0])
        best_fit = cross_validation(x_i, y_i)
        sum_error += errors[best_fit]
    print(sum_error)


# Cross Validation function to train model so that it can accurately predict which function to plot with
def cross_validation(xs, ys):
    total_errors = np.zeros(3)

    # We split data in half into "training" and "validation" sets. We do this 40 times
    for i in range(0, 40):
        # Pair x and y values so they don't get mixed up when shuffling values in sets.
        result = np.array(list(zip(xs, ys)))
        np.random.shuffle(result)
        shuffled = np.split(result, 2)
        train = np.array(list(zip(*shuffled[0])))
        test = np.array(list(zip(*shuffled[1])))
        train_xs, train_ys = train[0], train[1]
        test_xs, test_ys = test[0], test[1]
        # Calculate different parameters using training set
        array = lse(train_xs, train_ys)
        linear_a, linear_b = array[1][1], array[1][2]
        linear_error = square_error(test_ys, linear_a + linear_b * test_xs)
        cubic_a0, cubic_a1, cubic_a2, cubic_a3 = array[2][1], array[2][2], array[2][3], array[2][4]
        x2 = np.multiply(test_xs, test_xs)
        x3 = np.multiply(x2, test_xs)
        cubic_test_ys = cubic_a0 + cubic_a1 * test_xs + cubic_a2 * x2 + cubic_a3 * x3
        cubic_error = square_error(test_ys, cubic_test_ys)
        sin_a, sin_b = array[3][1], array[3][2]
        sin_error = square_error(test_ys, sin_a + sin_b * np.sin(test_xs))
        # We calculate sum of error values every time we execute cross validation.
        total_errors += np.array([linear_error, cubic_error, sin_error])
    # Return the function with smallest cross validation error.
    return np.argmin(total_errors)


# Plots for corresponding function
def plot(xs, ys, n, ax):
    # Linear plot
    if n == 0:
        var = linear(xs, ys)
        a = var[1]
        b = var[2]
        linear_ys = a + b * xs
        ax.plot(xs, linear_ys, 'r-', c='r', lw=2, label="Linear")

    # Polynomial or order 3 plot
    if n == 1:
        var = poly3(xs, ys)
        a0 = var[1]
        a1 = var[2]
        a2 = var[3]
        a3 = var[4]
        x2 = np.multiply(xs, xs)
        x3 = np.multiply(x2, xs)
        predicted_ys = a0 + a1 * xs + a2 * x2 + a3 * x3
        ax.plot(xs, predicted_ys, c='g', lw=2, label="Poly3")

    # Sin plot
    if n == 2:
        var = sin(xs, ys)
        a = var[1]
        b = var[2]
        predicted_ys = a + b * np.sin(xs)
        ax.plot(xs, predicted_ys, c='b', lw=2, label="Sin")


# Main Function
def main(argv):
    # If only filename is entered in command line then sum squared error is printed
    if len(argv) == 2:
        x_values, y_values = load_points_from_file(str(argv[1]))
        regression_no_plot(x_values, y_values)
    # If filename and "--plot" is entered on command line then error is returned and line is plotted
    elif len(argv) == 3:
        if argv[2] == "--plot":
            x_values, y_values = load_points_from_file(str(argv[1]))
            regression(x_values, y_values)
    # Error message returned if invalid input
    else:
        print("Error: Input filename and/or '--plot'")


main(sys.argv)

