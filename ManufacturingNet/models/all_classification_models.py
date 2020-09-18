"""AllClassificationModels runs every available classification
algorithm on the given dataset and outputs the mean accuracy, 5-fold
cross validation score, and execution time of each successful model
when run() is called.

View the documentation at https://manufacturingnet.readthedocs.io/.
"""

import io
import time
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC, LinearSVC, NuSVC
from xgboost import XGBClassifier


class AllClassificationModels:
    """Wrapper class around all supported classification models:
    LogisticRegression, RandomForest, SVC, NuSVC, LinearSVC, and
    XGBClassifier.
    """

    def __init__(self, attributes=None, labels=None):
        """Initializes an AllClassificationModels object."""
        self.attributes = attributes
        self.labels = labels

        self.test_size = None
        self.verbose = None

        self.logistic_regression = None
        self.random_forest = None
        self.SVC = None
        self.nu_SVC = None
        self.linear_SVC = None
        self.XGB_classifier = None

        self._classification_models = {"Model": ["Accuracy", "5-Fold CV Mean",
                                                 "Time (seconds)"]}
        self._failures = []

    # Accessor methods

    def get_attributes(self):
        """Accessor method for attributes."""
        return self.attributes

    def get_labels(self):
        """Accessor method for labels."""
        return self.labels

    def get_all_classification_models(self):
        """Accessor method that returns a list of all models."""
        return [self.logistic_regression, self.random_forest, self.SVC,
                self.nu_SVC, self.linear_SVC, self.XGB_classifier]

    def get_logistic_regression(self):
        """Accessor method for logistic_regression."""
        return self.logistic_regression

    def get_random_forest(self):
        """Accessor method for random_forest."""
        return self.random_forest

    def get_SVC(self):
        """Accessor method for SVC."""
        return self.SVC

    def get_nu_SVC(self):
        """Accessor method for nu_SVC."""
        return self.nu_SVC

    def get_linear_SVC(self):
        """Accessor method for linear_SVC."""
        return self.linear_SVC

    def get_XGB_classifier(self):
        """Accessor method for XGB_classifier."""
        return self.XGB_classifier

    # Modifier methods

    def set_attributes(self, new_attributes=None):
        """Modifier method for attributes."""
        self.attributes = new_attributes

    def set_labels(self, new_labels=None):
        """Modifier method for labels."""
        self.labels = new_labels

    # Classification functionality

    def run(self):
        """Driver method for running all classification models with
        given attributes and labels.
        """
        # Get parameters; create models
        self._create_models()

        # Call helper method for running all classification models
        # Suppress output, if needed
        if not self.verbose:
            suppress_output = io.StringIO()
            with redirect_stderr(suppress_output), \
                    redirect_stdout(suppress_output):
                self._all_classification_models_runner()
        else:
            self._all_classification_models_runner()

        # Print results
        self._print_results()

    # Helper methods

    def _create_models(self):
        """Prompts user for parameter input and instantiates all the
        classifier models.
        """
        print("\n==========================================")
        print("= All Classifier Models Parameter Inputs =")
        print("==========================================")

        # Get user input for verbose, test_size
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
        self.logistic_regression = LogisticRegression(verbose=self.verbose)
        self.random_forest = RandomForestClassifier(verbose=self.verbose)
        self.SVC = SVC(verbose=self.verbose, probability=True)
        self.nu_SVC = NuSVC(verbose=self.verbose, probability=True)
        self.linear_SVC = LinearSVC(verbose=self.verbose)
        self.XGB_classifier = XGBClassifier(verbosity=int(self.verbose))

    def _all_classification_models_runner(self):
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
            self.logistic_regression.fit(dataset_X_train, dataset_y_train)
            end_time = time.time()

            accuracy = self.logistic_regression.score(dataset_X_test,
                                                      dataset_y_test)
            cv_score = np.mean(cross_val_score(self.logistic_regression,
                                               self.attributes, self.labels,
                                               cv=5))
            elapsed_time = end_time - start_time

            self._classification_models["LogisticRegression"] = \
                [accuracy, cv_score, elapsed_time]
        except Exception as e:
            print("\nLogisticRegression failed. Exception message:")
            print(e, "\n")
            self._failures.append("LogisticRegression")

        try:
            start_time = time.time()
            self.random_forest.fit(dataset_X_train, dataset_y_train)
            end_time = time.time()

            accuracy = self.random_forest.score(dataset_X_test, dataset_y_test)
            cv_score = np.mean(cross_val_score(self.random_forest,
                                               self.attributes, self.labels,
                                               cv=5))
            elapsed_time = end_time - start_time

            self._classification_models["RandomForest"] = \
                [accuracy, cv_score, elapsed_time]
        except Exception as e:
            print("\nRandomForest failed. Exception message:")
            print(e, "\n")
            self._failures.append("RandomForest")

        try:
            start_time = time.time()
            self.SVC.fit(dataset_X_train, dataset_y_train)
            end_time = time.time()

            accuracy = self.SVC.score(dataset_X_test, dataset_y_test)
            cv_score = np.mean(cross_val_score(self.SVC, self.attributes,
                                               self.labels, cv=5))
            elapsed_time = end_time - start_time

            self._classification_models["SVC"] = \
                [accuracy, cv_score, elapsed_time]
        except Exception as e:
            print("\nSVC failed. Exception message:")
            print(e, "\n")
            self._failures.append("SVC")

        try:
            start_time = time.time()
            self.nu_SVC.fit(dataset_X_train, dataset_y_train)
            end_time = time.time()

            accuracy = self.nu_SVC.score(dataset_X_test, dataset_y_test)
            cv_score = np.mean(cross_val_score(self.nu_SVC, self.attributes,
                                               self.labels, cv=5))
            elapsed_time = end_time - start_time

            self._classification_models["NuSVC"] = \
                [accuracy, cv_score, elapsed_time]
        except Exception as e:
            print("\nNuSVC failed. Exception message:")
            print(e, "\n")
            self._failures.append("NuSVC")

        try:
            start_time = time.time()
            self.linear_SVC.fit(dataset_X_train, dataset_y_train)
            end_time = time.time()

            accuracy = self.linear_SVC.score(dataset_X_test, dataset_y_test)
            cv_score = np.mean(cross_val_score(self.linear_SVC, self.attributes,
                                               self.labels, cv=5))
            elapsed_time = end_time - start_time

            self._classification_models["LinearSVC"] = \
                [accuracy, cv_score, elapsed_time]
        except Exception as e:
            print("\nLinearSVc failed. Exception message:")
            print(e, "\n")
            self._failures.append("LinearSVC")

        try:
            start_time = time.time()
            self.XGB_classifier.fit(dataset_X_train, dataset_y_train)
            end_time = time.time()

            accuracy = self.XGB_classifier.score(
                dataset_X_test, dataset_y_test)
            cv_score = np.mean(cross_val_score(self.XGB_classifier,
                                               self.attributes, self.labels,
                                               cv=5))
            elapsed_time = end_time - start_time

            self._classification_models["XGBClassifier"] = \
                [accuracy, cv_score, elapsed_time]
        except Exception as e:
            print("\nXGBClassifier failed. Exception message:")
            print(e, "\n")
            self._failures.append("XGBClassifier")

    def _print_results(self):
        """Helper method that prints results of
        _all_classification_models_runner() in tabular form.
        """
        # Print models that didn't fail
        print("\n===========")
        print("= Results =")
        print("===========")

        for model, data in self._classification_models.items():
            print("\n{:<20} {:<20} {:<20} {:<20}".format(model, data[0], data[1],
                                                         data[2]))

        print()

        # Print failures, if any
        if len(self._failures) > 0:
            print("The following models failed to run:\n")

            for entry in self._failures:
                print(entry)

        print()
