import numpy as np
import matplotlib.pyplot as plt

def plotDecisionBoundary(X1, X2, y, model):
    """
    Plots the decision boundary for a binary classification model along with the training data points.

    Parameters:
        X1 (array-like): Feature values for the first feature.
        X2 (array-like): Feature values for the second feature.
        y (array-like): Target labels.
        model (object): Trained binary classification model with a `predict` method.

    Returns:
        None
    """
    #--- Write your code here ---#
    x_min, x_max = X1.min() - 1, X1.max() + 1
    y_min, y_max = X2.min() - 1, X2.max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    # Flatten the grid so that each point can be fed into the model
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict the function value for the whole grid
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X1, X2, c=y, cmap='coolwarm', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()
    plt.show()
    