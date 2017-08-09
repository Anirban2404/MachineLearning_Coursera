import matplotlib.pyplot as plt

# PLOTDATA Plots the data points x and y into a new figure
# PLOTDATA(x, y) plots the data points and gives the figure axes labels of
#  population and profit.
# ===================== YOUR CODE HERE ======================
# Instructions: Plot the training data into a figure using the "figure" and "plot" commands.
# Set the axes labels using the "xlabel" and "ylabel" commands.Assume the population and revenue
# data have been passed in as the x and y arguments of this function.

def plotData(x, y):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'rx', markersize=10) # Plot the data
    plt.grid(True)  # Always plot.grid true
    plt.ylabel('Profit in $10,000s') # Set the y?axis label
    plt.xlabel('Population of City in 10,000s') # Set the x?axis label
    plt.hold(True)
    #plt.show()

#==============================================================


