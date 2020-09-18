"""AllRegressionModels runs every available regression algorithm on the
given dataset and outputs the coefficient of determination and
execution time of each successful model when run() is called.

View the documentation at https://manufacturingnet.readthedocs.io/.
"""

import io
import time
from contextlib import redirect_stderr, redirect_stdout

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, LinearSVR, NuSVR
from xgboost import XGBRegressor


class AllRegressionModels:
    """Wrapper class around all supported regression models:
    LinearRegression, RandomForest, SVR, NuSVR, LinearSVR, and
    XGBRegressor.
    """

    def __init__(self, attributes=None, labels=None):
        """Initializes an AllRegressionModels object."""
        self.attributes = attributes
        self.labels = labels

        self.test_size = None
        self.verbose = None

        self.linear_regression = None
        self.random_forest = None
        self.SVR = None
        self.nu_SVR = None
        self.linear_SVR = None
        self.XGB_regressor = None

        self._regression_models = {"Model": ["R2 Score", "Time (seconds)"]}
        self._failures = []

    # Accessor methods

    def get_attributes(self):
        """Accessor method for attributes."""
        return self.attributes

    def get_labels(self):
        """Accessor method for labels."""
        return self.labels

    def get_all_regression_models(self):
        """Accessor method that returns a list of all models."""
        return [self.linear_regression, self.random_forest, self.SVR,
                self.nu_SVR, self.linear_SVR, self.XGB_regressor]

    def get_linear_regression(self):
        """Accessor method for linear_regression."""
        return self.linear_regression

    def get_random_forest(self):
        """Accessor method for random_forest."""
        return self.random_forest

    def get_SVR(self):
        """Accessor method for SVR."""
        return self.SVR

    def get_nu_SVR(self):
        """Accessor method for nu_SVR."""
        return self.nu_SVR

    def get_linear_SVR(self):
        """Accessor method for linear_SVR."""
        return self.linear_SVR

    def get_XGB_regressor(self):
        """Accessor method for XGB_regressor."""
        return self.XGB_regressor

    # Modifier methods

    def set_attributes(self, new_attributes=None):
        """Modifier method for attributes."""
        self.attributes = new_attributes

    def set_labels(self, new_labels=None):
        """Modifier method for labels."""
        self.labels = new_labels

    # Regression functionality

    def run(self):
        """Driver method for running all regression models with given
        attributes and labels.
        """
        # Get parameters; create models
        self._create_models()

        # Call helper method for running all regression models
        # Suppress output, if needed
        if not self.verbose:
            suppress_output = io.StringIO()
            with redirect_stderr(suppress_output), \
                    redirect_stdout(suppress_output):
                self._all_regression_models_runner()
        else:
            self._all_regression_models_runner()

        # Print results
        self._print_results()

    # Helper methods

    def _create_models(self):
        """Prompts user for parameter input and instantiates all the
        regressor models.
        """
        print("\n==========================================")
        print("= All Regression Models Parameter Inputs =")
        print("==========================================\n")

        # Get user input for verbose, test size
        while True:
            user_input = input("\nEnable verbose logging (y/N)? ").lower()
            if user_input == "y":
                self.verbose = True
                break
            elif user_input in {"n", ""}:
                self.verbose = False
                break
            else:
                print("Invalid input.")

        print("verbose =", self.verbose)

        while True:
            user_input = input("\nWhat fraction of the dataset should be used "
                               + "for testing (0,1)? ")
            try:
                if user_input == "":
                    self.test_size = 0.25
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

        print("\n===========================================")
        print("= End of inputs; press enter to continue. =")
        input("===========================================\n")

        # Create models
        self.linear_regression = LinearRegression()
        self.random_forest = RandomForestRegressor(verbose=self.verbose)
        self.SVR = SVR(verbose=self.verbose)
        self.nu_SVR = NuSVR(verbose=self.verbose)
        self.linear_SVR = LinearSVR(verbose=self.verbose)
        self.XGB_regressor = XGBRegressor(verbosity=int(self.verbose))

    def _all_regression_models_runner(self):
        """Helper method that runs all models using the given dataset
        and all default parameters.
        """

        # Split dataset
        dataset_X_train, dataset_X_test, dataset_y_train, dataset_y_test = \
            train_test_split(self.attributes, self.labels,
                             test_size=self.test_size)

        # Run and time all models; identify each as success or failure
        try:
            start_time = time.time()
            self.linear_regression.fit(dataset_X_train, dataset_y_train)
            end_time = time.time()

            r2_score = self.linear_regression.score(dataset_X_test,
                                                    dataset_y_test)
            elapsed_time = end_time - start_time

            self._regression_models["LinearRegression"] = \
                [r2_score, elapsed_time]
        except Exception as e:
            print("\nLinearRegression failed. Exception message:")
            print(e, "\n")
            self._failures.append("LinearRegression")

        try:
            start_time = time.time()
            self.random_forest.fit(dataset_X_train, dataset_y_train)
            end_time = time.time()

            r2_score = self.random_forest.score(dataset_X_test, dataset_y_test)
            elapsed_time = end_time - start_time

            self._regression_models["RandomForest"] = [r2_score, elapsed_time]
        except Exception as e:
            print("\nRandomForest failed. Exception message:")
            print(e, "\n")
            self._failures.append("RandomForest")

        try:
            start_time = time.time()
            self.SVR.fit(dataset_X_train, dataset_y_train)
            end_time = time.time()

            r2_score = self.SVR.score(dataset_X_test, dataset_y_test)
            elapsed_time = end_time - start_time

            self._regression_models["SVR"] = [r2_score, elapsed_time]
        except Exception as e:
            print("\nSVR failed. Exception message:")
            print(e, "\n")
            self._failures.append("SVR")

        try:
            start_time = time.time()
            self.nu_SVR.fit(dataset_X_train, dataset_y_train)
            end_time = time.time()

            r2_score = self.nu_SVR.score(dataset_X_test, dataset_y_test)
            elapsed_time = end_time - start_time

            self._regression_models["NuSVR"] = [r2_score, elapsed_time]
        except Exception as e:
            print("\nNuSVR failed. Exception message:")
            print(e, "\n")
            self._failures.append("NuSVR")

        try:
            start_time = time.time()
            self.linear_SVR.fit(dataset_X_train, dataset_y_train)
            end_time = time.time()

            r2_score = self.linear_SVR.score(dataset_X_test, dataset_y_test)
            elapsed_time = end_time - start_time

            self._regression_models["LinearSVR"] = [r2_score, elapsed_time]
        except Exception as e:
            print("\nLinearSVR failed. Exception message:")
            print(e, "\n")
            self._failures.append("LinearSVR")

        try:
            start_time = time.time()
            self.XGB_regressor.fit(dataset_X_train, dataset_y_train)
            end_time = time.time()

            r2_score = self.XGB_regressor.score(dataset_X_test, dataset_y_test)
            elapsed_time = end_time - start_time

            self._regression_models["XGBRegressor"] = [r2_score, elapsed_time]
        except Exception as e:
            print("\nXGBRegressor failed. Exception message:")
            print(e, "\n")
            self._failures.append("XGBRegressor")

    def _print_results(self):
        """Helper method that prints results of
        _all_regression_models_runner() in tabular form.
        """
        # Print models that didn't fail
        print("\n===========")
        print("= Results =")
        print("===========")

        for model, data in self._regression_models.items():
            print("\n{:<20} {:<20} {:<20}".format(model, data[0], data[1]))

        print()

        # Print failures, if any
        if len(self._failures) > 0:
            print("The following models failed to run:\n")

            for entry in self._failures:
                print(entry)

        print()
