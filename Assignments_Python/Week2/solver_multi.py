#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import normalEqn
import gradientDescentMulti
import featureNormalize
import matplotlib.pyplot as plt
#Import necessary matplotlib tools for 3d plots
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
## Machine Learning Online Class
#  Exercise 1: Linear regression with multiple variables
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear regression exercise.
#
#  You will need to complete the following functions in this
#  exericse:
#
#     warmUpExercise.m
#     plotData.m
#     gradientDescent.m
#     computeCost.m
#     gradientDescentMulti.m
#     computeCostMulti.m
#     featureNormalize.m
#     normalEqn.m
#
#  For this part of the exercise, you will need to change some
#  parts of the code below for various experiments (e.g., changing
#  learning rates).
#

## Initialization

if __name__ == '__main__':
    ## =================== Part 1: Feature Normalization ===================
    datafile = 'ex1data2.txt'
    data = np.loadtxt(datafile, delimiter=',')  # Read in comma separated data
    # Form the usual "X" matrix and "y" vector
    X = data[:,:2]
    y = data[:,2]
    m = y.size  # number of training examples
    # Print out some data points
    print('First 10 examples from the dataset: \n')
    for i in xrange(10):
        print "x = [{:.0f} {:.0f}], y = {:.0f}".format(X[i, 0], X[i, 1], y[i])

    raw_input('Program paused. Press enter to continue.\n')

    # Scale features and set them to zero mean
    print('Normalizing Features...')
    X_norm, mu, sigma = featureNormalize.featureNormalize(X)
    # print X_norm
    # print mu
    # print sigma

    ones = np.ones((m, 1), dtype=np.int)

    X = np.column_stack((ones, X))  # Add a column of ones to x

    # ================ Part 2: Gradient Descent ================
    # ====================== YOUR CODE HERE ======================
    # Instructions: We have provided you with the following starter
    # code that runs gradient descent with a particular learning rate (alpha).
    # Your task is to first make sure that your functions -
    # computeCost and gradientDescent already work with this starter code
    # and support multiple variables.
    # After that, try running gradient descent with different values of alpha
    # and see which one gives you the best result. Finally, you should
    # complete the code at the end to predict the price of a 1650 sq-ft, 3 br house.
    # Hint: By using the 'hold on' command, you can plot multiple graphs on the same figure.
    #  Hint: At prediction, make sure you do the same feature normalization.
    print 'Running gradient descent ...\n'

    # Choose some alpha value
    alpha = 0.01
    num_iters = 400

    # Init Theta and Run Gradient Descent
    theta = np.zeros((3, 1), dtype=np.int)

    theta, J_history = gradientDescentMulti.gradientDescentMulti(X, y, theta, alpha, num_iters)

    # Plot the convergence graph
    plt.plot(xrange(J_history.size), J_history, "-b", linewidth=2)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.show()

    # Display gradient descent's result
    print 'Theta computed from gradient descent: '
    print "{:f}, {:f}, {:f}".format(theta[0, 0], theta[1, 0], theta[2, 0])
    print ""

    # Estimate the price of a 1650 sq-ft, 3 br house
    # ====================== YOUR CODE HERE ======================
    # Recall that the first column of X is all-ones. Thus, it does
    # not need to be normalized.

    test = np.array([1650., 3.])
    area_norm = (test[0] - float(mu[:, 0])) / float(sigma[:, 0])
    br_norm = test[1]
    # print area_norm
    # print br_norm
    house_norm = np.array([1, area_norm, br_norm])

    price = np.array(house_norm).dot(theta)

    # ============================================================

    print "Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n ${:,.2f}".format(price[0])

    raw_input('Program paused. Press enter to continue.\n')

    # ================ Part 3: Normal Equations ================

    print 'Solving with normal equations...\n'

    # ================  YOUR CODE HERE ======================
    # Instructions: The following code computes the closed form solution
    # for linear regression using the normal equations.You should complete the
    # code in normalEqn.py
    #
    # After doing so, you should complete this code
    # to predict the price of a 1650 sq-ft, 3 br house.
    #

    ## Load Data
    atafile = 'ex1data2.txt'
    data = np.loadtxt(datafile, delimiter=',')  # Read in comma separated data
    # Form the usual "X" matrix and "y" vector
    X = data[:, :2]
    y = data[:, 2]
    m = y.size  # number of training examples

    # Add intercept term to X
    ones = np.ones((m, 1), dtype=np.int)
    X = np.column_stack((ones, X))  # Add a column of ones to x

    # Calculate the parameters from the normal equation
    theta = normalEqn.normalEqn(X, y)

    # Display normal equation's result
    print 'Theta computed from the normal equations:'
    print "{:f}, {:f}, {:f}".format(theta[0], theta[1], theta[2])
    print ''

    # Estimate the price of a 1650 sq-ft, 3 br house
    # ====================== YOUR CODE HERE ======================
    house_norm = np.array([1, 1650, 3])
    price = np.array(house_norm).dot(theta)

    # ============================================================

    print("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n ${:,.2f}".format(price))

raw_input('Program paused. Press enter to finish.\n')