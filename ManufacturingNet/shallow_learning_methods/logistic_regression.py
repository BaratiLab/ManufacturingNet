"""LogRegression trains a logistic regression model implemented by
Scikit-Learn on the given dataset. Before training, the user is
prompted for parameter input. After training, model metrics are
displayed, and the user can make new predictions.

View the documentation at https://manufacturingnet.readthedocs.io/.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, roc_curve, \
    roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, \
    train_test_split

class LogRegression:
    """Class framework for logistic regression model."""

    def __init__(self, attributes=None, labels=None):
        """Initializes a LogisticRegression object."""
        self.attributes = attributes
        self.labels = labels

        self.test_size = None
        self.cv = None
        self.graph_results = None
        self.fpr = None
        self.tpr = None
        self.bin = False
        self.gridsearch = False
        self.gs_params = None
        self.gs_result = None

        self.regression = None
        self.classes = None
        self.coefficients = None
        self.intercept = None
        self.n_iter = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.roc_auc = None
        self.confusion_matrix = None
        self.cross_val_scores = None

    # Accessor methods

    def get_attributes(self):
        """Accessor method for attributes."""
        return self.attributes

    def get_labels(self):
        """Accessor method for labels."""
        return self.labels

    def get_classes(self):
        """Accessor method for classes."""
        return self.classes

    def get_regression(self):
        """Accessor method for regression."""
        return self.regression

    def get_coefficents(self):
        """Accessor method for coefficients."""
        return self.coefficients

    def get_n_iter(self):
        """Accessor method for n_iter."""
        return self.n_iter

    def get_accuracy(self):
        """Accessor method for accuracy."""
        return self.accuracy

    def get_roc_auc(self):
        """Accessor method for roc_auc."""
        return self.roc_auc

    def get_confusion_matrix(self):
        """Accessor method for confusion_matrix."""
        return self.confusion_matrix

    def get_cross_val_scores(self):
        """Accessor method for cross_val_scores."""
        return self.cross_val_scores

    # Modifier methods

    def set_attributes(self, new_attributes=None):
        """Modifier method for attributes."""
        self.attributes = new_attributes

    def set_labels(self, new_labels=None):
        """Modifier method for labels."""
        self.labels = new_labels

    # Wrapper for logistic regression model

    def run(self):
        """Performs logistic regression on dataset and updates relevant
        instance data.
        """
        if self._check_inputs():
            # Instantiate LogisticRegression() object using helper method
            self.regression = self._create_model()

            # Split into training and testing set
            dataset_X_train, dataset_X_test, dataset_y_train, dataset_y_test = \
                train_test_split(self.attributes, self.labels,
                                 test_size=self.test_size)

            # Train the model and get resultant coefficients
            # Handle exception if arguments are incorrect
            try:
                self.regression.fit(dataset_X_train, np.ravel(dataset_y_train))
            except Exception as e:
                print("An exception occurred while training the regression",
                      "model. Check your inputs and try again.")
                print("Here is the exception message:")
                print(e)
                self.regression = None
                return

            # Get resultant model instance data
            self.classes = self.regression.classes_
            self.coefficients = self.regression.coef_
            self.intercept = self.regression.intercept_
            self.n_iter = self.regression.n_iter_

            # Make predictions using testing set
            y_prediction = self.regression.predict(dataset_X_test)

            # Metrics
            self.accuracy = accuracy_score(y_prediction, dataset_y_test)
            probas = self.regression.predict_proba(dataset_X_test)

            # If classification is binary, calculate roc_auc
            if probas.shape[1] == 2:
                self.bin = True
                self.roc_auc = \
                    roc_auc_score(self.regression.predict(dataset_X_test),
                                  probas[::, 1])
                self.fpr, self.tpr, _ = roc_curve(dataset_y_test, probas[::, 1])
            # Else, calculate confusion matrix
            else:
                self.confusion_matrix = \
                    confusion_matrix(dataset_y_test, y_prediction)

            self.cross_val_scores = cross_val_score(self.regression,
                                                    self.attributes,
                                                    self.labels, cv=self.cv)

            # Output results
            self._output_results()

    def predict(self, dataset_X=None):
        """Predicts the output of each datapoint in dataset_X using the
        regression model. Returns the predictions.
        """
        # Check that run() has already been called
        if self.regression is None:
            print("The regression model seems to be missing. Have you called",
                  "run() yet?")
            return None

        # Try to make the prediction
        # Handle exception if dataset_X isn't a valid input
        try:
            y_prediction = self.regression.predict(dataset_X)
        except Exception as e:
            print("The model failed to run. Check your inputs and try again.")
            print("Here is the exception message:")
            print(e)
            return None

        print("\nLogRegression Predictions:\n", y_prediction, "\n")
        return y_prediction

    # Helper methods

    def _create_model(self):
        """Runs UI for getting parameters and creating model."""
        print("\n==================================")
        print("= LogRegression Parameter Inputs =")
        print("==================================\n")

        if input("Use default parameters (Y/n)? ").lower() != "n":
            self.test_size = 0.25
            self.cv = None
            print("\n=============================================")
            print("= End of inputs; press any key to continue. =")
            input("=============================================\n")
            return LogisticRegression()

        print("\nIf you are unsure about a parameter, press enter to use its",
              "default value.")
        print("Invalid parameter inputs will be replaced with their default",
              "values.")
        print("If you finish entering parameters early, enter 'q' to skip",
              "ahead.\n")

        # Set defaults
        self.test_size = 0.25
        self.cv = None
        self.graph_results = False
        penalty = "l2"
        dual = False
        tol = 0.0001
        C = 1.0
        fit_intercept = True
        intercept_scaling = 1
        class_weight = None
        random_state = None
        solver = "lbfgs"
        max_iter = 100
        multi_class = "auto"
        verbose = 0
        warm_start = False
        n_jobs = None
        l1_ratio = None

        # Get user parameter input
        while True:
            user_input = input("What fraction of the dataset should be the "
                               + "testing set? Input a decimal: ")

            try:
                self.test_size = float(user_input)
            except Exception:
                if user_input.lower() == "q":
                    break

            user_input = input("\n\nUse GridSearch to find the best "
                               + "hyperparameters (y/N)? ").lower()

            if user_input == "q":
                break

            while user_input == "y":
                print("\n= GridSearch Parameter Inputs =\n")
                print("Note: All parameters are required. Skipping ahead",
                      "will quit GridSearch.")
                print("Press 'q' to skip GridSearch.")
                self.gridsearch = True
                params = {}

                print("\nEnter the classifier penalties to evaluate.")
                print("Options: 1-'l1', 2-'l2', 3-'elasticnet'. Enter 'all' for",
                      "all options.")
                print("Example input: 1,2,3")
                user_input = input().lower()

                if user_input == "q":
                    self.gridsearch = False
                    break
                elif user_input == "all":
                    pen_params = ["l1", "l2", "elasticnet"]
                else:
                    pen_dict = {1: "l1", 2: "l2", 3: "elasticnet"}

                    try:
                        pen_params_int = \
                            list(map(int, list(user_input.split(","))))
                        pen_params = []
                        for each in pen_params_int:
                            pen_params.append(pen_dict.get(each))
                    except Exception:
                        print("\nInput not recognized. Skipping GridSearch...")
                        self.gridsearch = False
                        break

                params["penalty"] = pen_params

                print("\nEnter the solvers to evaluate.")
                print("Options: 1-'newton-cg', 2-'lbfgs', 3-'liblinear',",
                      "4-'sag', 5-'saga'. Enter 'all' for all options.")
                print("Example input: 1,2,3")
                user_input = input().lower()

                if user_input == "q":
                    self.gridsearch = False
                    break
                elif user_input == "all":
                    sol_params = ["newton-cg", "lbfgs", "liblinear", "sag",
                                  "saga"]
                else:
                    sol_dict = {1: "newton-cg", 2: "lbfgs", 3: "liblinear",
                                4: "sag", 5: "saga"}

                    try:
                        sol_params_int = \
                            list(map(int, list(user_input.split(","))))
                        sol_params = []
                        for each in sol_params_int:
                            sol_params.append(sol_dict.get(each))
                    except Exception:
                        print("\nInput not recognized. Skipping GridSearch...")
                        self.gridsearch = False
                        break

                params["solver"] = sol_params
                self.gs_params = params

                print("\n= End of GridSearch inputs. =")
                break

            user_input = \
                input("\n\nEnter the number of folds for cross validation: ")

            try:
                self.cv = int(user_input)
            except Exception:
                if user_input.lower() == "q":
                    break

            user_input = input("\nGraph the ROC curve? Only binary "
                               + "classification is supported (y/N): ").lower()

            if user_input == "y":
                self.graph_results = True
            elif user_input == "q":
                break

            print("\nWhich norm should be used in penalization?")
            user_input = input("Enter 1 for 'l1', 2 for 'l2', 3 for "
                               + "'elasticnet', or 4 for 'none': ").lower()

            if user_input == "q":
                break
            elif user_input == "1":
                penalty = "l1"
            elif user_input == "3":
                penalty = "elasticnet"
            elif user_input == "4":
                penalty = "none"

            user_input = input("\nUse dual formulation (y/N)? ").lower()
            if user_input == "y":
                dual = True
            elif user_input == "q":
                break

            user_input = input("\nEnter a number for the tolerance for "
                               + "stopping criteria: ")

            try:
                tol = float(user_input)
            except Exception:
                if user_input.lower() == "q":
                    break

            user_input = input("\nEnter a positive number for the inverse of "
                               + "regularization strength (C): ")

            try:
                C = float(user_input)
            except Exception:
                if user_input.lower() == "q":
                    break

            user_input = \
                input("\nInclude a y-intercept in the model (Y/n)? ").lower()

            if user_input == "n":
                fit_intercept = False
            elif user_input == "q":
                break

            if fit_intercept:
                user_input = input("\nEnter a number for the intercept "
                                   + "scaling factor: ")
                try:
                    intercept_scaling = float(user_input)
                except Exception:
                    if user_input.lower() == "q":
                        break

            user_input = input("\nAutomatically balance the class weights "
                               + "(y/N)? ").lower()

            if user_input == "y":
                class_weight = "balanced"
            elif user_input == "q":
                break

            print("\nTo set manual weights, call",
                  "get_regression().set_params() to set the class_weight",
                  "parameter.")

            user_input = \
                input("\nInput an integer for the random number seed: ")

            try:
                random_state = int(user_input)
            except Exception:
                if user_input.lower() == "q":
                    break

            print("\nWhich algorithm should be used in the optimization",
                  "problem?")
            user_input = input("Enter 1 for 'newton-cg', 2 for 'lbfgs', 3 for "
                               + "'liblinear', 4 for 'sag', or 5 for "
                               + "'saga': ").lower()

            if user_input == "q":
                break
            elif user_input == "1":
                solver = "newton-cg"
            elif user_input == "3":
                solver = "liblinear"
            elif user_input == "4":
                solver = "sag"
            elif user_input == "5":
                solver = "saga"

            user_input = input("\nEnter the max number of iterations: ")

            try:
                max_iter = int(user_input)
            except Exception:
                if user_input.lower() == "q":
                    break

            print("\nPlease choose a multiclass scheme.")
            user_input = input("Enter 1 for one-vs-rest, 2 for multinomial, "
                               + "or 3 to automatically choose: ").lower()

            if user_input == "q":
                break
            elif user_input == "1":
                multi_class = "ovr"
            elif user_input == "2":
                multi_class = "multinomial"

            user_input = \
                input("\nEnable verbose output during training (y/N)? ").lower()

            if user_input == "y":
                verbose = 1
            elif user_input == "q":
                break

            user_input = input("\nEnable warm start? This will use the "
                               + "previous solution for fitting (y/N): ").lower()

            if user_input == "y":
                warm_start = True
            elif user_input == "q":
                break

            user_input = input("\nEnter the number of jobs for computation: ")

            try:
                n_jobs = int(user_input)
            except Exception:
                if user_input.lower() == "q":
                    break

            if penalty == "elasticnet":
                user_input = input("\nEnter a number for the Elastic-Net "
                                   + "mixing parameter: ")
                try:
                    l1_ratio = float(user_input)
                except Exception:
                    break

            break

        print("\n=============================================")
        print("= End of inputs; press any key to continue. =")
        input("=============================================\n")

        # If GridSearch is enabled
        if self.gridsearch:
            acc_scorer = make_scorer(accuracy_score)
            clf = LogisticRegression()
            dataset_X_train, dataset_X_test, dataset_y_train, dataset_y_test = \
                train_test_split(self.attributes, self.labels,
                                 test_size=self.test_size)

            # Run GridSearch
            grid_obj = GridSearchCV(clf, self.gs_params, scoring=acc_scorer)
            grid_obj = grid_obj.fit(dataset_X_train, dataset_y_train)

            # Set the clf to the best combination of parameters
            clf = grid_obj.best_estimator_
            print("Best GridSearch Parameters:\n", clf)

            # Fit the best algorithm to the data
            clf.fit(dataset_X_train, dataset_y_train)
            predictions = clf.predict(dataset_X_test)
            self.gs_result = accuracy_score(dataset_y_test, predictions)

        return LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C,
                                  fit_intercept=fit_intercept,
                                  intercept_scaling=intercept_scaling,
                                  class_weight=class_weight,
                                  random_state=random_state, solver=solver,
                                  max_iter=max_iter, multi_class=multi_class,
                                  verbose=verbose, warm_start=warm_start,
                                  n_jobs=n_jobs, l1_ratio=l1_ratio)

    def _output_results(self):
        """Outputs model metrics after run() finishes."""
        print("\n=========================")
        print("= LogRegression Results =")
        print("=========================\n")

        print("Classes:\n", self.classes)
        print("\nNumber of Iterations:\n", self.n_iter)
        print("\n{:<20} {:<20}".format("Accuracy:", self.accuracy))

        if self.bin:
            print("\n{:<20} {:<20}".format("ROC AUC:", self.roc_auc))
        else:
            print("\nConfusion Matrix:\n", self.confusion_matrix)

        print("\nCross Validation Scores: ", self.cross_val_scores)

        if self.gridsearch:
            print("\n{:<20} {:<20}".format("GridSearch Score:",
                                           self.gs_result))

        if self.bin and self.graph_results:
            plt.plot(self.fpr, self.tpr)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc=4)
            plt.show()

        print("\n\nCall predict() to make predictions for new data.")

        print("\n===================")
        print("= End of results. =")
        print("===================\n")

    def _check_inputs(self):
        """Verifies if the instance data is ready for use in logistic
        regression model.
        """
        # Check if attributes exists
        if self.attributes is None:
            print("attributes is missing; call set_attributes(new_attributes)",
                  "to fix this! new_attributes should be a populated numpy",
                  "array of your independent variables.")
            return False

        # Check if labels exists
        if self.labels is None:
            print("labels is missing; call set_labels(new_labels) to fix this!",
                  "new_labels should be a populated numpy array of your",
                  "dependent variables.")
            return False

        # Check if attributes and labels have same number of rows (samples)
        if self.attributes.shape[0] != self.labels.shape[0]:
            print("attributes and labels don't have the same number of rows.",
                  "Make sure the number of samples in each dataset matches!")
            return False

        return True
