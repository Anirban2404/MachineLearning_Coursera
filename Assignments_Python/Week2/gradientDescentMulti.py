import numpy as np
import computeCost as cC

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    # GRADIENTDESCENTMULTI Performs gradient descent to learn theta
    # theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by taking
    # num_iters gradient steps with learning rate alpha
    #
    # Initialize some useful values
    m = y.size  # number of training examples
    J_history = np.zeros((num_iters, 1), dtype=np.float)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Perform a single gradient step on the parameter vector theta.
    # Hint: While debugging, it can be useful to print out the values of the cost
    # function(computeCostMulti) and gradient here.

    for iter in xrange(num_iters):
        h_theta = X.dot(theta)
        theta = theta - alpha * (1.0 / m) * np.transpose(X).dot(h_theta - np.transpose([y]))
        J_history[iter] = cC.computeCost(X, y, theta)

    return theta, J_history