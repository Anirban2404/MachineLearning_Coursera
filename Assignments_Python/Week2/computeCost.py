# COMPUTECOST Compute cost for linear regression J = COMPUTECOST(X, y, theta)
# computes the cost of using theta as the parameter for linear regression to fit
# the data points in X and y
import numpy as np

def computeCost(X, y, theta):
    # Initialize some useful values
    m = y.size # number of training examples
    # You need to return the following variables correctly
    J = 0

    # ====================== YOUR CODE HERE ======================
    #  Instructions: Compute the cost of a particular choice of theta
    #  You shoulD set J to the cost.
    h_theta = X.dot(theta)
    summation = sum(np.power((h_theta - np.transpose([y])), 2))
    J = (1.0 / (2 * m)) * summation

    # =============================================================
    return J