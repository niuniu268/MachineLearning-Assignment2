from abc import ABC, abstractmethod
import numpy as np
import itertools

class MachineLearningModel(ABC):
    """
    Abstract base class for machine learning models.
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        pass

def _polynomial_features(self, X):
    """
        Generate polynomial features from the input features.
        Check the slides for hints on how to implement this one. 
        This method is used by the regression models and must work
        for any degree polynomial
        Parameters:
        X (array-like): Features of the data.

        Returns:
        X_poly (array-like): Polynomial features.
    """    
    n_samples, n_features = X.shape
    features = [np.ones(n_samples)]
    for degree in range(1, self.degree + 1):
        for items in itertools.combinations_with_replacement(range(n_features), degree):
            feature = np.prod([X[:, i] for i in items], axis=0)
            features.append(feature)
    return np.vstack(features).T

class RegressionModelNormalEquation(MachineLearningModel):
    """
    Class for regression models using the Normal Equation for polynomial regression.
    """

    def __init__(self, degree):
        """
        Initialize the model with the specified polynomial degree.

        Parameters:
        degree (int): Degree of the polynomial features.
        """
        #--- Write your code here ---#

        self.degree = degree
        self.beta = None


    def _polynomial_features(self, X):

        n_samples, n_features = X.shape
        features = [np.ones(n_samples)]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(range(n_features), degree):
                feature = np.prod([X[:, i] for i in items], axis=0)
                features.append(feature)
        return np.vstack(features).T

    def fit(self, X, y):
        X_poly = self._polynomial_features(X)
        self.beta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        #--- Write your code here ---#
        X_poly = self._polynomial_features(X)
        return X_poly @ self.beta

    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (MSE).
        """
        #--- Write your code here ---#
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        return mse

class RegressionModelGradientDescent(MachineLearningModel):
    """
    Class for regression models using gradient descent optimization.
    """

    def __init__(self, degree, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the model with the specified parameters.

        Parameters:
        degree (int): Degree of the polynomial features.
        learning_rate (float): Learning rate for gradient descent.
        num_iterations (int): Number of iterations for gradient descent.
        """
        #--- Write your code here ---#
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.beta = None
        self.cost_history = []

    def _polynomial_features(self, X):

        n_samples, n_features = X.shape
        features = [np.ones(n_samples)]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(range(n_features), degree):
                feature = np.prod([X[:, i] for i in items], axis=0)
                features.append(feature)
        return np.vstack(features).T

    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        #--- Write your code here ---#
        X_poly = self._polynomial_features(X)
        self.beta = np.zeros(X_poly.shape[1])
        m = len(y)

        for _ in range(self.num_iterations):
            predictions = X_poly @ self.beta
            errors = predictions - y
            gradient = (1 / m) * X_poly.T @ errors
            self.beta -= self.learning_rate * gradient
            cost = (1 / (2 * m)) * np.sum(errors ** 2)
            self.cost_history.append(cost)


    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        #--- Write your code here ---#
        
        X_poly = self._polynomial_features(X)
        return X_poly @ self.beta

    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (MSE).
        """
        #--- Write your code here ---#
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        return mse

class LogisticRegression:
    """
    Logistic Regression model using gradient descent optimization.
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the logistic regression model.

        Parameters:
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for gradient descent.
        """
        #--- Write your code here ---#
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.beta = None
        self.cost_history = []

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        #--- Write your code here ---#
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)
        for i in range(self.num_iterations):
            predictions = self._sigmoid(X @ self.beta)
            gradient = X.T @ (predictions - y) / n_samples
            self.beta -= self.learning_rate * gradient
            cost = self._cost_function(X, y)
            self.cost_history.append(cost)

    def predict(self, X):
        """
        Make predictions using the trained logistic regression model.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted probabilities.
        """
        #--- Write your code here ---#
        probabilities = self._sigmoid(X @ self.beta)
        return (probabilities >= 0.5).astype(int)


    def evaluate(self, X, y):
        """
        Evaluate the logistic regression model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (e.g., accuracy).
        """
        #--- Write your code here ---#
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

    def _sigmoid(self, z):
        """
        Sigmoid function.

        Parameters:
        z (array-like): Input to the sigmoid function.

        Returns:
        result (array-like): Output of the sigmoid function.
        """
        #--- Write your code here ---#
        return 1 / (1 + np.exp(-z))
    
    def _initialize_weights(self, n_features):
        self.beta = np.zeros(n_features)

    def _cost_function(self, X, y):
        """
        Compute the logistic regression cost function.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        #--- Write your code here ---#
        m = len(y)
        epsilon = 1e-15
        predictions = self._sigmoid(X @ self.beta)
        cost = -np.mean(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))
        return cost

    
class NonLinearLogisticRegression:
    """
    Nonlinear Logistic Regression model using gradient descent optimization.
    It works for 2 features (when creating the variable interactions)
    """

    def __init__(self, degree=2, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the nonlinear logistic regression model.

        Parameters:
        degree (int): Degree of polynomial features.
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for gradient descent.
        """
        #--- Write your code here ---#
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.beta = None
        self.cost_history = []



    def fit(self, X, y):
        """
        Train the nonlinear logistic regression model using gradient descent.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        #--- Write your code here ---#
        X_poly = self.mapFeature(X[:, 0], X[:, 1])
        n_samples, n_features = X_poly.shape
        self._initialize_weights(n_features)
        for i in range(self.num_iterations):
            predictions = self._sigmoid(X_poly @ self.beta)
            gradient = X_poly.T @ (predictions - y) / n_samples
            self.beta -= self.learning_rate * gradient
            cost = self._cost_function(X_poly, y)
            self.cost_history.append(cost)

    def predict(self, X):
        """
        Make predictions using the trained nonlinear logistic regression model.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted probabilities.
        """
        #--- Write your code here ---#
        X_poly = self.mapFeature(X[:, 0], X[:, 1])

        probabilities = self._sigmoid(X_poly @ self.beta)
        return (probabilities >= 0.5).astype(int)

    def evaluate(self, X, y):
        """
        Evaluate the nonlinear logistic regression model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        #--- Write your code here ---#
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


    def _sigmoid(self, z):
        """
        Sigmoid function.

        Parameters:
        z (array-like): Input to the sigmoid function.

        Returns:
        result (array-like): Output of the sigmoid function.
        """
        #--- Write your code here ---#
        return 1 / (1 + np.exp(-z))

    def mapFeature(self, X1, X2):
        """
        Map the features to a higher-dimensional space using polynomial features.
        Check the slides to have hints on how to implement this function.
        Parameters:
        X1 (array-like): Feature 1.
        X2 (array-like): Feature 2.
        D (int): Degree of polynomial features.

        Returns:
        X_poly (array-like): Polynomial features.
        """
        #--- Write your code here ---#
        if X1.ndim > 1:
            X1 = X1.flatten()
        if X2.ndim > 1:
            X2 = X2.flatten()
        
        # Start with an array of ones for the bias term.
        output = [np.ones(X1.shape[0])]
        
        # Generate terms for each degree.
        for i in range(1, self.degree + 1):
            for j in range(i + 1):
                term = (X1 ** (i - j)) * (X2 ** j)
                output.append(term)
        
        # Stack features horizontally
        return np.vstack(output).T
    
    def _initialize_weights(self, n_features):
        self.beta = np.zeros(n_features)


    def _cost_function(self, X, y):
        """
        Compute the logistic regression cost function.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        #--- Write your code here ---#
        m = len(y)
        epsilon = 1e-15
        predictions = self._sigmoid(X @ self.beta)
        cost = -np.mean(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))
        return cost

