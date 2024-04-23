import numpy as np
from ROCAnalysis import ROCAnalysis
from sklearn.model_selection import train_test_split

class ForwardSelection:
    """
    A class for performing forward feature selection based on maximizing the F-score of a given model.

    Attributes:
        X (array-like): Feature matrix.
        y (array-like): Target labels.
        model (object): Machine learning model with `fit` and `predict` methods.
        selected_features (list): List of selected feature indices.
        best_cost (float): Best F-score achieved during feature selection.
    """

    def __init__(self, X, y, model):
        """
        Initializes the ForwardSelection object.

        Parameters:
            X (array-like): Feature matrix.
            y (array-like): Target labels.
            model (object): Machine learning model with `fit` and `predict` methods.
        """
        #--- Write your code here ---#
        self.X = X
        self.y = y
        self.model = model
        self.selected_features = []
        self.best_cost = -np.inf


    def create_split(self):
        """
        Creates a train-test split of the data.

        Parameters:
            X (array-like): Feature matrix.
            y (array-like): Target labels.

        Returns:
            X_train (array-like): Features for training.
            X_test (array-like): Features for testing.
            y_train (array-like): Target labels for training.
            y_test (array-like): Target labels for testing.
        """
        #--- Write your code here ---#
        return train_test_split(self.X, self.y, test_size=0.20, random_state=42)


    def train_model_with_features(self, features):
        """
        Trains the model using selected features and evaluates it using ROCAnalysis.

        Parameters:
            features (list): List of feature indices.

        Returns:
            float: F-score obtained by evaluating the model.
        """
        #--- Write your code here ---#
        X_train, X_test, y_train, y_test = self.create_split()
        X_train_subset = X_train[:, features]
        X_test_subset = X_test[:, features]

        self.model.fit(X_train_subset, y_train)
        y_pred = self.model.predict(X_test_subset)
        analysis = ROCAnalysis(y_pred, y_test)
        return analysis.f_score()


    def forward_selection(self):
        """
        Performs forward feature selection based on maximizing the F-score.
        """
        #--- Write your code here ---#
        n_features = self.X.shape[1]
        available_features = list(range(n_features))
        
        while available_features:
            best_feature = None
            best_score = -np.inf
            
            for feature in available_features:
                current_features = self.selected_features + [feature]
                score = self.train_model_with_features(current_features)
                
                if score > best_score:
                    best_score = score
                    best_feature = feature

            if best_feature is not None and best_score > self.best_cost:
                self.best_cost = best_score
                self.selected_features.append(best_feature)
                available_features.remove(best_feature)
            else:
                break  # No improvement
                
    def fit(self):
        """
        Fits the model using the selected features.
        """
        #--- Write your code here ---#
        X_train, _, y_train, _ = self.create_split()
        X_train_subset = X_train[:, self.selected_features]
        self.model.fit(X_train_subset, y_train)


    def predict(self, X_test):
        """
        Predicts the target labels for the given test features.

        Parameters:
            X_test (array-like): Test features.

        Returns:
            array-like: Predicted target labels.
        """
        #--- Write your code here ---#
        print("Shape of X_test before slicing:", X_test.shape)
        X_test_subset = X_test[:, self.selected_features]
        print("Shape of X_test after slicing:", X_test_subset.shape)
        return self.model.predict(X_test_subset)
