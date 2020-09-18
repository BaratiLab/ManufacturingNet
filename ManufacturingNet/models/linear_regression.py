"""LinRegression trains a linear regression model implemented by
Scikit-Learn on the given dataset. Before training, the user is
prompted for parameter input. After training, model metrics are
displayed, and the user can make new predictions.

View the documentation at https://manufacturingnet.readthedocs.io/.
"""

from math import sqrt

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split


class LinRegression:
    """Class framework for linear regression model."""

    def __init__(self, attributes=None, labels=None):
        """Initializes a LinearRegression object."""
        self.attributes = attributes
        self.labels = labels

        self.test_size = None
        self.cv = None
        self.graph_results = None

        self.regression = None
        self.coefficients = None
        self.intercept = None
        self.mean_squared_error = None
        self.r2_score = None
        self.r_score = None
        self.cross_val_scores = None

    # Accessor methods

    def get_attributes(self):
        """Accessor method for attributes."""
        return self.attributes

    def get_labels(self):
        """Accessor method for labels."""
        return self.labels

    def get_regression(self):
        """Accessor method for regression."""
        return self.regression

    def get_coefficients(self):
        """Accessor method for coefficients."""
        return self.coefficients

    def get_intercept(self):
        """Accessor method for intercept."""
        return self.intercept

    def get_mean_squared_error(self):
        """Accessor method for mean_squared_error."""
        return self.mean_squared_error

    def get_r2_score(self):
        """Accessor method for r2_score."""
        return self.r2_score

    def get_r_score(self):
        """Accessor method for r_score."""
        return self.r_score

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

    # Wrapper for linear regression model

    def run(self):
        """Performs linear regression on dataset and updates relevant
        instance data.
        """
        if self._check_inputs():
            # Instantiate LinearRegression() object using helper method
            self.regression = self._create_model()

            # Split into training and testing sets
            dataset_X_train, dataset_X_test, dataset_y_train, dataset_y_test = \
                train_test_split(self.attributes, self.labels,
                                 test_size=self.test_size)

            # Train the model and get resultant coefficients
            # Handle exception if arguments aren't correct
            try:
                self.regression.fit(dataset_X_train, dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the regression",
                      "model. Check your inputs and try again.")
                print("Here is the exception message:")
                print(e)
                self.regression = None
                return

            # Get resultant coefficients and intercept of regression line
            self.coefficients = self.regression.coef_
            self.intercept = self.regression.intercept_

            # Make predictions using testing set
            y_prediction = self.regression.predict(dataset_X_test)

            # Metrics
            self.mean_squared_error = mean_squared_error(dataset_y_test,
                                                         y_prediction)
            self.r2_score = self.regression.score(dataset_X_test,
                                                  dataset_y_test)
            if self.r2_score >= 0:
                self.r_score = sqrt(self.r2_score)

            self.cross_val_scores = \
                cross_val_score(self.regression, self.attributes, self.labels,
                                cv=self.cv)

            # Output results
            self._output_results()

            # Plot results, if desired
            if self.graph_results:
                self._graph_results(dataset_X_test, dataset_y_test,
                                    y_prediction)

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

        print("\nLinRegression Predictions:\n", y_prediction, "\n")
        return y_prediction

    # Helper methods

    def _create_model(self):
        """Runs UI for getting parameters and creating model."""
        print("\n==================================")
        print("= LinRegression Parameter Inputs =")
        print("==================================\n")
        print("Default values:",
              "test_size = 0.25",
              "cv = 5",
              "graph_results = False",
              "fit_intercept = True",
              "normalize = False",
              "copy_X = True",
              "n_jobs = None", sep="\n")

        # Set defaults
        self.test_size = 0.25
        self.cv = None
        self.graph_results = False

        while True:
            user_input = input("\nUse default parameters (Y/n)? ").lower()
            if user_input in {"y", ""}:
                print("\n===========================================")
                print("= End of inputs; press enter to continue. =")
                input("===========================================\n")
                return LinearRegression()
            elif user_input == "n":
                break
            else:
                print("Invalid input.")

        print("\nIf you are unsure about a parameter, press enter to use its",
              "default value.")
        print("If you finish entering parameters early, enter 'q' to skip",
              "ahead.")

        # Set more defaults
        fit_intercept = True
        normalize = False
        copy_X = True
        n_jobs = None

        # Get user parameter input
        while True:
            break_early = False
            while True:
                user_input = input("\nWhat fraction of the dataset should be the "
                                   + "testing set (0,1)? ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input <= 0 or user_input >= 1:
                        raise Exception
                    else:
                        self.test_size = user_input
                        break
                except Exception:
                    print("Invalid input.")

            print("test_size =", self.test_size)

            if break_early:
                break

            while True:
                user_input = input("\nEnter the number of folds for cross "
                                   + "validation [2,): ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = int(user_input)
                    if user_input < 2:
                        raise Exception
                    else:
                        self.cv = user_input
                        break
                except Exception:
                    print("Invalid input.")

            print("cv =", self.cv)

            if break_early:
                break

            while self.attributes.shape[1] == 1:
                user_input = input("\nGraph the results (y/N)? ").lower()
                if user_input == "y":
                    self.graph_results = True
                    break
                elif user_input in {"n", ""}:
                    break
                elif user_input == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            if self.attributes.shape[1] == 1:
                print("graph_results =", self.graph_results)

            if break_early:
                break

            while True:
                user_input = input("\nInclude a y-intercept in the model "
                                   + "(Y/n)? ").lower()
                if user_input == "n":
                    fit_intercept = False
                    break
                elif user_input in {"y", ""}:
                    break
                elif user_input == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            print("fit_intercept =", fit_intercept)

            if break_early:
                break

            while True:
                user_input = input("\nNormalize the dataset (y/N)? ").lower()
                if user_input == "y":
                    normalize = True
                    break
                elif user_input in {"n", ""}:
                    break
                elif user_input == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            print("normalize =", normalize)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nCopy the dataset's features (Y/n)? ").lower()
                if user_input == "n":
                    copy_X = False
                    break
                elif user_input in {"y", ""}:
                    break
                elif user_input == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            print("copy_X =", copy_X)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nEnter a positive number of CPU cores to use: ")
                try:
                    if user_input.lower() in {"q", ""}:
                        break

                    user_input = int(user_input)
                    if user_input < 1:
                        raise Exception
                    else:
                        n_jobs = user_input
                        break
                except Exception:
                    print("Invalid input.")

            print("n_jobs =", n_jobs)
            break

        print("\n===========================================")
        print("= End of inputs; press enter to continue. =")
        input("===========================================\n")

        return LinearRegression(fit_intercept=fit_intercept,
                                normalize=normalize, copy_X=copy_X,
                                n_jobs=n_jobs)

    def _output_results(self):
        """Outputs model metrics after run() finishes."""
        print("\n=========================")
        print("= LinRegression Results =")
        print("=========================\n")

        print("Coefficients:\n", self.coefficients)
        print("\n{:<20} {:<20}".format("Intercept:", self.intercept))
        print("\n{:<20} {:<20}".format("Mean Squared Error:",
                                       self.mean_squared_error))
        print("\n{:<20} {:<20}".format("R2 Score:", self.r2_score))
        print("\n{:<20} {:<20}".format("R Score:", str(self.r_score)))
        print("\nCross Validation Scores:\n", self.cross_val_scores)

        print("\n\nCall predict() to make predictions for new data.")

        print("\n===================")
        print("= End of results. =")
        print("===================\n")

    def _graph_results(self, X_test, y_test, y_pred):
        """Graphs results of linear regression with one feature. This
        method only graphs two-dimensional results; thus, only
        univariate regression is supported.
        """
        if self.regression is None:
            print("The model isn't available. Have you called run() yet?")
            return

        plt.scatter(X_test, y_test, color="black")
        plt.plot(X_test, y_pred, color="blue", linewidth=3)
        plt.xticks(())
        plt.yticks(())
        plt.show()

    def _check_inputs(self):
        """Verifies if instance data is ready for use in linear
        regression model.
        """
        # Check if attributes exists
        if self.attributes is None:
            print("attributes is missing; call set_attributes(new_attributes)",
                  "to fix this! new_attributes should be a populated dataset",
                  "of independent variables.")
            return False

        # Check if labels exists
        if self.labels is None:
            print("labels is missing; call set_labels(new_labels) to fix this!",
                  "new_labels should be a populated dataset of dependent",
                  "variables.")
            return False

        # Check if attributes and labels have same number of rows (samples)
        if self.attributes.shape[0] != self.labels.shape[0]:
            print("attributes and labels don't have the same number of rows.",
                  "Make sure the number of samples in each dataset matches!")
            return False

        return True
