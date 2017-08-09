#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import warmUpExercise
import plotData
import computeCost
import gradientDescent
import matplotlib.pyplot as plt
#Import necessary matplotlib tools for 3d plots
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm

## Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
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
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#

if __name__ == '__main__':
    ## =================== Part 1: Basic Function ===================
    # Complete warmUpExercise.m
    print 'Running warmUpExercise ... \n'
    print '5x5 Identity Matrix: \n';
    eye = warmUpExercise.warmUp(5)
    print eye

    raw_input('Program paused. Press enter to continue.\n')

    ## ====================== Part 2: Plotting =======================
    print 'Plotting Data ...\n'
    datafile = 'ex1data1.txt'
    data = np.loadtxt(datafile, delimiter=',', usecols=(0, 1))  # Read in comma separated data
    # Form the usual "X" matrix and "y" vector
    X = data[:,0]
    y = data[:,1]
    m = y.size  # number of training examples
    # Plot Data
    # Note: You have to complete the code in plotData.py
    plotData.plotData(X, y)

    raw_input('Program paused. Press enter to continue.\n')

    ## =================== Part 3: Gradient descent ===================
    print 'Running Gradient Descent ...\n'
    ones = np.ones((m, 1), dtype=np.int)

    X = np.column_stack((ones, X)) # Add a column of ones to x
    theta = np.zeros((2,1), dtype=np.int) # initialize fitting parameters

    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01

    # compute and display initial cost
    J = computeCost.computeCost(X, y, theta)
    print J

    # run gradient descent
    theta = gradientDescent.gradientDescent(X, y, theta, alpha, iterations);

    # print theta to screen
    print 'Theta found by gradient descent: '
    print "{:f}, {:f}".format(theta[0, 0], theta[1, 0])

    # Plot the linear fit
    plt.plot(X[:,1], X.dot(theta), '-')
    plt.legend(['Training data', 'Linear regression'])
    plt.hold(False) # don't overlay any more plots on this figure
    plt.show()

    # Predict values for population sizes of 35, 000 and 70, 000
    predict1 = np.array([1, 3.5]).dot(theta)
    print "For population = 35,000, we predict a profit of {:f}".format(float(predict1 * 10000))
    predict2 = np.array([1, 7]).dot(theta)
    print 'For population = 70,000, we predict a profit of {:f}'.format(float(predict2 * 10000))

    ## ============= Part 4: Visualizing J(theta_0, theta_1) =============

    print 'Visualizing J(theta_0, theta_1)...'

    # Grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    # initialize J_vals to a matrix of 0's
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    # Fill out J_vals
    for i in xrange(len(theta0_vals)):
        for j in xrange(len(theta1_vals)):
            t = [[theta0_vals[i]], [theta1_vals[j]]]
            J_vals[i,j] = computeCost.computeCost(X, y, t)

    # Because of the way meshgrids work in the surf command, we need to
    # transpose J_vals before calling surf, or else the axes will be flipped
    J_vals = np.transpose(J_vals)

    # Surface plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals) # necessary for 3D graph
    surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap=cm.coolwarm, rstride=2, cstride=2)
    fig.colorbar(surf)
    plt.xlabel(r'$\theta_0$', fontsize=15)
    plt.ylabel(r'$\theta_1$', fontsize=15)
    plt.title('Cost (Minimization Path Shown in Blue)', fontsize=15)
    plt.show()
    #plt.show(block=False)
    plt.hold(False)

    ## Contour plot
    #fig = plt.figure()
    ## Plot J_vals as 20 contours spaced logarithmically between 0.01 and 100
    cset = plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20), cmap=cm.coolwarm)
    fig.colorbar(cset)
    plt.xlabel(r'$\theta_0$', fontsize=15)
    plt.ylabel(r'$\theta_1$', fontsize=15)
    plt.hold(True)
    plt.plot(theta[0,0], theta[1,0], 'rx', markersize=10, linewidth=2)
    plt.title('Contour Plot', fontsize=15)
    plt.show()
    plt.hold(False)

raw_input('Program paused. Press enter to finish.\n')