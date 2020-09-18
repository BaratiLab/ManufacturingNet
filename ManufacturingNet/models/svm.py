"""SVM can train SVC, NuSVC, LinearSVC, SVR, NuSVR, and LinearSVR
models implemented by Scikit-Learn on the given dataset. Before
training, the user is prompted for parameter input. After training,
model metrics are displayed, and the user can make new predictions.
Classification and regression are both supported.

View the documentation at https://manufacturingnet.readthedocs.io/.
"""

from math import sqrt

import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, make_scorer,
                             mean_squared_error, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, NuSVC, NuSVR


class SVM:
    """Class model for support vector machine (SVM) models."""

    def __init__(self, attributes=None, labels=None):
        """Initializes a SVM object."""
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

        self.classifier_SVC = None
        self.accuracy_SVC = None
        self.roc_auc_SVC = None
        self.confusion_matrix_SVC = None
        self.cross_val_scores_SVC = None
        self.classifier_nu_SVC = None
        self.accuracy_nu_SVC = None
        self.roc_auc_nu_SVC = None
        self.confusion_matrix_nu_SVC = None
        self.cross_val_scores_nu_SVC = None
        self.classifier_linear_SVC = None
        self.accuracy_linear_SVC = None
        self.cross_val_scores_linear_SVC = None

        self.regressor_SVR = None
        self.mean_squared_error_SVR = None
        self.r2_score_SVR = None
        self.r_score_SVR = None
        self.cross_val_scores_SVR = None
        self.regressor_nu_SVR = None
        self.mean_squared_error_nu_SVR = None
        self.r2_score_nu_SVR = None
        self.r_score_nu_SVR = None
        self.cross_val_scores_nu_SVR = None
        self.regressor_linear_SVR = None
        self.mean_squared_error_linear_SVR = None
        self.r2_score_linear_SVR = None
        self.r_score_linear_SVR = None
        self.cross_val_scores_linear_SVR = None

        # References to training and testing subsets of dataset
        # For re-use purposes
        self.dataset_X_train = None
        self.dataset_y_train = None
        self.dataset_X_test = None
        self.dataset_y_test = None

    # Accessor Methods

    def get_attributes(self):
        """Accessor method for attributes."""
        return self.attributes

    def get_labels(self):
        """Accessor method for labels."""
        return self.labels

    def get_classifier_SVC(self):
        """Accessor method for classifier_SVC."""
        return self.classifier_SVC

    def get_accuracy_SVC(self):
        """Accessor method for accuracy_SVC."""
        return self.accuracy_SVC

    def get_roc_auc_SVC(self):
        """Accessor method for roc_auc_SVC."""
        return self.roc_auc_SVC

    def get_confusion_matrix_SVC(self):
        """Accessor method for confusion_matrix_SVC."""
        return self.confusion_matrix_SVC

    def get_cross_val_scores_SVC(self):
        """Accessor method for cross_val_scores_SVC."""
        return self.cross_val_scores_SVC

    def get_classifier_nu_SVC(self):
        """Accessor method for classifier_nu_SVC."""
        return self.classifier_nu_SVC

    def get_accuracy_nu_SVC(self):
        """Accessor method for accuracy_nu_SVC."""
        return self.accuracy_nu_SVC

    def get_roc_auc_nu_SVC(self):
        """Accessor method for roc_auc_nu_SVC."""
        return self.roc_auc_nu_SVC

    def get_confusion_matrix_nu_SVC(self):
        """Accessor method for confusion_matrix_nu_SVC."""
        return self.confusion_matrix_nu_SVC

    def get_cross_val_scores_nu_SVC(self):
        """Accessor method for cross_val_scores_nu_SVC."""
        return self.cross_val_scores_nu_SVC

    def get_classifier_linear_SVC(self):
        """Accessor method for classifier_linear_SVC."""
        return self.classifier_linear_SVC

    def get_accuracy_linear_SVC(self):
        """Accessor method for accuracy_linear_SVC."""
        return self.accuracy_linear_SVC

    def get_cross_val_scores_linear_SVC(self):
        """Accessor method for cross_val_scores_linear_SVC."""
        return self.cross_val_scores_linear_SVC

    def get_regressor_SVR(self):
        """Accessor method for regressor_SVR."""
        return self.regressor_SVR

    def get_mean_squared_error_SVR(self):
        """Accessor method for mean_squared_error_SVR."""
        return self.mean_squared_error_SVR

    def get_r2_score_SVR(self):
        """Accessor method for r2_score_SVR."""
        return self.r2_score_SVR

    def get_r_score_SVR(self):
        """Accessor method for r_score_SVR."""
        return self.r_score_SVR

    def get_cross_val_scores_SVR(self):
        """Accessor method for cross_val_scores_SVR."""
        return self.cross_val_scores_SVR

    def get_regressor_nu_SVR(self):
        """Accessor method for regressor_nu_SVR."""
        return self.regressor_nu_SVR

    def get_mean_squared_error_nu_SVR(self):
        """Accessor method for mean_squared_error_nu_SVR."""
        return self.mean_squared_error_nu_SVR

    def get_r2_score_nu_SVR(self):
        """Accessor method for r2_score_nu_SVR."""
        return self.r2_score_nu_SVR

    def get_r_score_nu_SVR(self):
        """Accessor method for r_score_nu_SVR."""
        return self.r_score_nu_SVR

    def get_cross_val_scores_nu_SVR(self):
        """Accessor method for cross_val_scores_nu_SVR."""
        return self.cross_val_scores_nu_SVR

    def get_regressor_linear_SVR(self):
        """Accessor method for regressor_linear_SVR."""
        return self.regressor_linear_SVR

    def get_mean_squared_error_linear_SVR(self):
        """Accessor method for mean_squared_error_linear_SVR."""
        return self.mean_squared_error_linear_SVR

    def get_r2_score_linear_SVR(self):
        """Accessor method for r2_score_linear_SVR."""
        return self.r2_score_linear_SVR

    def get_r_score_linear_SVR(self):
        """Accessor method for r_score_linear_SVR."""
        return self.r_score_linear_SVR

    def get_cross_val_scores_linear_SVR(self):
        """Accessor method for cross_val_scores_linear_SVR."""
        return self.cross_val_scores_linear_SVR

    # Modifier Methods

    def set_attributes(self, new_attributes=None):
        """Modifier method for attributes."""
        self.attributes = new_attributes

    def set_labels(self, new_labels=None):
        """Modifier method for labels."""
        self.labels = new_labels

    # Wrappers for SVM classification classes

    def run_SVC(self):
        """Runs SVC model."""
        if self._check_inputs():
            # Initialize classifier
            self.classifier_SVC = self._create_SVC_model(is_nu=False)

            # Split data, if needed
            # If testing/training sets are still None, call _split_data()
            if self.dataset_X_test is None:
                self._split_data()

            # Train classifier
            # Handle exception if arguments are incorrect
            try:
                self.classifier_SVC.fit(self.dataset_X_train,
                                        self.dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the SVC model.",
                      "Check your arguments and try again.")
                print("Here is the exception message:")
                print(e)
                self.classifier_SVC = None
                return

            # Metrics
            self.accuracy_SVC = self.classifier_SVC.score(self.dataset_X_test,
                                                          self.dataset_y_test)
            y_prediction = self.classifier_SVC.predict(self.dataset_X_test)
            probas = self.classifier_SVC.predict_proba(self.dataset_X_test)

            # If classification is binary, calculate roc_auc
            if probas.shape[1] == 2:
                self.bin = True
                self.roc_auc_SVC = roc_auc_score(y_prediction, probas[::, 1])
                self.fpr, self.tpr, _ = roc_curve(self.dataset_y_test,
                                                  probas[::, 1])
            # Else, calculate confusion matrix
            else:
                self.confusion_matrix_SVC = confusion_matrix(self.dataset_y_test,
                                                             y_prediction)

            self.cross_val_scores_SVC = \
                cross_val_score(self.classifier_SVC, self.attributes,
                                self.labels, cv=self.cv)

            # Output results
            self._output_classifier_results(model="SVC")

    def predict_SVC(self, dataset_X=None):
        """Classifies each datapoint in dataset_X using the SVC model.
        Returns the predicted classifications.
        """
        # Check that run_SVC() has already been called
        if self.classifier_SVC is None:
            print("The SVC model seems to be missing. Have you called",
                  "run_SVC() yet?")
            return None

        # Try to make the prediction
        # Handle exception if dataset_X isn't a valid input
        try:
            y_prediction = self.classifier_SVC.predict(dataset_X)
        except Exception as e:
            print("The SVC model failed to run.",
                  "Check your inputs and try again.")
            print("Here is the exception message:")
            print(e)
            return None

        print("\nSVC Predictions:\n", y_prediction, "\n")
        return y_prediction

    def run_nu_SVC(self):
        """Runs NuSVC model."""
        if self._check_inputs():
            # Initialize classifier
            self.classifier_nu_SVC = self._create_SVC_model(is_nu=True)

            # Split data, if needed
            # If testing/training sets are still None, call _split_data()
            if self.dataset_X_test is None:
                self._split_data()

            # Train classifier
            # Handle exception if arguments are incorrect
            try:
                self.classifier_nu_SVC.fit(self.dataset_X_train,
                                           self.dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the NuSVC model.",
                      "Check your arguments and try again.")
                print("Here is the exception message:")
                print(e)
                self.classifier_nu_SVC = None
                return

            # Metrics
            self.accuracy_nu_SVC =\
                self.classifier_nu_SVC.score(self.dataset_X_test,
                                             self.dataset_y_test)
            y_prediction = self.classifier_nu_SVC.predict(self.dataset_X_test)
            probas = self.classifier_nu_SVC.predict_proba(self.dataset_X_test)

            # If classification is binary, calculate roc_auc
            if probas.shape[1] == 2:
                self.bin = True
                self.roc_auc_nu_SVC = roc_auc_score(
                    y_prediction, probas[::, 1])
                self.fpr, self.tpr, _ = \
                    roc_curve(self.dataset_y_test, probas[::, 1])
            # Else, calculate confusion matrix
            else:
                self.confusion_matrix_nu_SVC = \
                    confusion_matrix(self.dataset_y_test, y_prediction)

            self.cross_val_scores_nu_SVC = \
                cross_val_score(self.classifier_nu_SVC, self.attributes,
                                self.labels, cv=self.cv)

            # Output results
            self._output_classifier_results(model="NuSVC")

    def predict_nu_SVC(self, dataset_X=None):
        """Classifies each datapoint in dataset_X using the NuSVC model.
        Returns the predicted classifications.
        """
        # Check that run_nu_SVC() has already been called
        if self.classifier_nu_SVC is None:
            print("The NuSVC model seems to be missing.",
                  "Have you called run_nu_SVC() yet?")
            return None

        # Try to make the prediction
        # Handle exception if dataset_X isn't a valid input
        try:
            y_prediction = self.classifier_nu_SVC.predict(dataset_X)
        except Exception as e:
            print("The NuSVC model failed to run.",
                  "Check your inputs and try again.")
            print("Here is the exception message:")
            print(e)
            return None

        print("\nNuSVC Predictions:\n", y_prediction, "\n")
        return y_prediction

    def run_linear_SVC(self):
        """Runs LinearSVC model."""
        if self._check_inputs():
            # Initialize classifier
            self.classifier_linear_SVC = self._create_linear_SVC_model()

            # Split data, if needed
            # If testing/training sets are still None, call _split_data()
            if self.dataset_X_test is None:
                self._split_data()

            # Train classifier
            # Handle exception if arguments are incorrect
            try:
                self.classifier_linear_SVC.fit(self.dataset_X_train,
                                               self.dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the LinearSVC",
                      "model. Check your arguments and try again.")
                print("Here is the exception message:")
                print(e)
                self.classifier_linear_SVC = None
                return

            # Metrics
            self.accuracy_linear_SVC = \
                self.classifier_linear_SVC.score(self.dataset_X_test,
                                                 self.dataset_y_test)
            self.cross_val_scores_linear_SVC =\
                cross_val_score(self.classifier_linear_SVC, self.attributes,
                                self.labels, cv=self.cv)

            # Output results
            self._output_classifier_results(model="LinearSVC")

    def predict_linear_SVC(self, dataset_X=None):
        """Classifies each datapoint in dataset_X using the LinearSVC
        model. Returns the predicted classifications.
        """
        # Check that run_linear_SVC() has already been called
        if self.classifier_linear_SVC is None:
            print("The LinearSVC model seems to be missing.",
                  "Have you called run_linear_SVC() yet?")
            return None

        # Try to make the prediction
        # Handle exception if dataset_X isn't a valid input
        try:
            y_prediction = self.classifier_linear_SVC.predict(dataset_X)
        except Exception as e:
            print("The LinearSVC model failed to run.",
                  "Check your inputs and try again.")
            print("Here is the exception message:")
            print(e)
            return None

        print("\nLinearSVC Predictions:\n", y_prediction, "\n")
        return y_prediction

    # Wrappers for SVM regression classes

    def run_SVR(self):
        """Runs SVR model."""
        if self._check_inputs():
            # Initialize regression model
            self.regressor_SVR = self._create_SVR_model(is_nu=False)

            # Split data, if needed
            # If testing/training sets are still None, call _split_data()
            if self.dataset_X_test is None:
                self._split_data()

            # Train regression model
            # Handle exception if arguments are incorrect and/or if labels isn't
            # quantitative data
            try:
                self.regressor_SVR.fit(self.dataset_X_train,
                                       self.dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the SVR model.",
                      "Check you arguments and try again.")
                print("Does labels only contain quantitative data?")
                print("Here is the exception message:")
                print(e)
                self.regressor_SVR = None
                return

            # Evaluate metrics of model
            y_prediction = self.regressor_SVR.predict(self.dataset_X_test)
            self.mean_squared_error_SVR = \
                mean_squared_error(self.dataset_y_test, y_prediction)
            self.r2_score_SVR = self.regressor_SVR.score(self.dataset_X_test,
                                                         self.dataset_y_test)
            if self.r2_score_SVR >= 0:
                self.r_score_SVR = sqrt(self.r2_score_SVR)

            self.cross_val_scores_SVR = \
                cross_val_score(self.regressor_SVR, self.attributes,
                                self.labels, cv=self.cv)

            # Output results
            self._output_regressor_results(model="SVR")

    def predict_SVR(self, dataset_X=None):
        """Predicts the output of each datapoint in dataset_X using the
        SVR model. Returns the predictions.
        """
        # Check that run_SVR() has already been called
        if self.regressor_SVR is None:
            print("The SVR model seems to be missing.",
                  "Have you called run_SVR() yet?")
            return None

        # Try to make the prediction
        # Handle exception if dataset_X isn't a valid input
        try:
            y_prediction = self.regressor_SVR.predict(dataset_X)
        except Exception as e:
            print("The SVR model failed to run.",
                  "Check your inputs and try again.")
            print("Here is the exception message:")
            print(e)
            return None

        print("\nSVR Predictions:\n", y_prediction, "\n")
        return y_prediction

    def run_nu_SVR(self):
        """Runs NuSVR model."""
        if self._check_inputs():
            # Initialize regression model
            self.regressor_nu_SVR = self._create_SVR_model(is_nu=True)

            # Split data, if needed
            # If testing/training sets are still None, call _split_data()
            if self.dataset_X_test is None:
                self._split_data()

            # Train regression model
            # Handle exception if arguments are incorrect and/or if labels isn't
            # quantitative data
            try:
                self.regressor_nu_SVR.fit(self.dataset_X_train,
                                          self.dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the NuSVR model.",
                      "Check you arguments and try again.")
                print("Does labels only contain quantitative data?")
                print("Here is the exception message:")
                print(e)
                self.regressor_nu_SVR = None
                return

            # Metrics
            y_prediction = self.regressor_nu_SVR.predict(self.dataset_X_test)
            self.mean_squared_error_nu_SVR = \
                mean_squared_error(self.dataset_y_test, y_prediction)
            self.r2_score_nu_SVR = \
                self.regressor_nu_SVR.score(self.dataset_X_test,
                                            self.dataset_y_test)
            if self.r2_score_nu_SVR >= 0:
                self.r_score_nu_SVR = sqrt(self.r2_score_nu_SVR)

            self.cross_val_scores_nu_SVR = \
                cross_val_score(self.regressor_nu_SVR, self.attributes,
                                self.labels, cv=self.cv)

            # Output results
            self._output_regressor_results(model="NuSVR")

    def predict_nu_SVR(self, dataset_X=None):
        """Predicts the output of each datapoint in dataset_X using the
        NuSVR model. Returns the predictions.
        """
        # Check that run_nu_SVR() has already been called
        if self.regressor_nu_SVR is None:
            print("The NuSVR model seems to be missing.",
                  "Have you called run_nu_SVR() yet?")
            return None

        # Try to make the prediction; handle exception if dataset_X isn't a valid input
        try:
            y_prediction = self.regressor_nu_SVR.predict(dataset_X)
        except Exception as e:
            print("The NuSVR model failed to run.",
                  "Check your inputs and try again.")
            print("Here is the exception message:")
            print(e)
            return None

        print("\nNuSVR Predictions:\n", y_prediction, "\n")
        return y_prediction

    def run_linear_SVR(self):
        """Runs LinearSVR model."""
        if self._check_inputs():
            # Initialize regression model
            self.regressor_linear_SVR = self._create_linear_SVR_model()

            # Split data, if needed
            # If testing/training sets are still None, call _split_data()
            if self.dataset_X_test is None:
                self._split_data()

            # Train regression model
            # Handle exception if arguments are incorrect and/or labels isn't
            # quantitative data
            try:
                self.regressor_linear_SVR.fit(self.dataset_X_train,
                                              self.dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the LinearSVR",
                      "model. Check you arguments and try again.")
                print("Does labels only contain quantitative data?")
                print("Here is the exception message:")
                print(e)
                self.regressor_linear_SVR = None
                return

            # Metrics
            y_prediction = self.regressor_linear_SVR.predict(
                self.dataset_X_test)
            self.mean_squared_error_linear_SVR = \
                mean_squared_error(self.dataset_y_test, y_prediction)
            self.r2_score_linear_SVR = \
                self.regressor_linear_SVR.score(self.dataset_X_test,
                                                self.dataset_y_test)
            if self.r2_score_linear_SVR >= 0:
                self.r_score_linear_SVR = sqrt(self.r2_score_linear_SVR)

            self.cross_val_scores_linear_SVR = \
                cross_val_score(self.regressor_linear_SVR, self.attributes,
                                self.labels, cv=self.cv)

            # Output results
            self._output_regressor_results(model="LinearSVR")

    def predict_linear_SVR(self, dataset_X=None):
        """Predicts the output of each datapoint in dataset_X using the
        LinearSVR model. Returns the predictions.
        """
        # Check that run_linear_SVR() has already been called
        if self.regressor_linear_SVR is None:
            print("The LinearSVR model seems to be missing.",
                  "Have you called run_linear_SVR() yet?")
            return None

        # Try to make the prediction
        # Handle exception if dataset_X isn't a valid input
        try:
            y_prediction = self.regressor_linear_SVR.predict(dataset_X)
        except Exception as e:
            print("The LinearSVR model failed to run.",
                  "Check your inputs and try again.")
            print("Here is the exception message:")
            print(e)
            return None

        print("\nLinearSVR Predictions:\n", y_prediction, "\n")
        return y_prediction

    # Helper methods

    def _create_SVC_model(self, is_nu):
        """Runs UI for getting parameters and creating SVC or NuSVC
        model.
        """
        if is_nu:
            print("\n==========================")
            print("= NuSVC Parameter Inputs =")
            print("==========================\n")
        else:
            print("\n========================")
            print("= SVC Parameter Inputs =")
            print("========================\n")

        print("Default values:", "test_size = 0.25", "cv = 5",
              "graph_results = False", sep="\n")
        if is_nu:
            print("nu = 0.5")
        else:
            print("C = 1.0")

        print("kernel = 'rbf'",
              "degree = 3",
              "gamma = 'scale'",
              "coef0 = 0.0",
              "shrinking = True",
              "tol = 0.001",
              "cache_size = 200",
              "class_weight = None",
              "max_iter = -1",
              "decision_function_shape = 'ovr'",
              "break_ties = False",
              "random_state = None",
              "verbose = False", sep="\n")

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
                if is_nu:
                    return NuSVC(probability=True)
                return SVC(probability=True)
            elif user_input == "n":
                break
            else:
                print("Invalid input.")

        print("\nIf you are unsure about a parameter, press enter to use its",
              "default value.")
        print("If you finish entering parameters early, enter 'q' to skip",
              "ahead.\n")

        # Set more defaults
        if is_nu:
            nu = 0.5
        else:
            C = 1.0

        kernel = "rbf"
        degree = 3
        gamma = "scale"
        coef0 = 0.0
        shrinking = True
        tol = 0.001
        cache_size = 200
        class_weight = None
        max_iter = -1
        decision_function_shape = "ovr"
        break_ties = False
        random_state = None
        verbose = False

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

                    self.test_size = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("test_size =", self.test_size)

            if break_early:
                break

            while True:
                user_input = input("\nUse GridSearch to find the best "
                                   + "hyperparameters (y/N)? ").lower()
                if user_input == "q":
                    break_early = True
                    break
                elif user_input in {"n", "y", ""}:
                    break
                else:
                    print("Invalid input.")

            if break_early:
                break

            while user_input == "y":
                print("\n= GridSearch Parameter Inputs =\n")
                print("Enter 'q' to skip GridSearch.")
                self.gridsearch = True
                params = {}

                while True:
                    print("\nEnter the kernels to try out.")
                    print("Options: 1-'linear', 2-'poly', 3-'rbf', 4-'sigmoid'.",
                          "Enter 'all' for all options.")
                    print("Example input: 1,2,3")
                    user_input = input().lower()

                    if user_input == "q":
                        self.gridsearch = False
                        break_early = True
                        break
                    elif user_input == "all":
                        kern_params = ["linear", "poly", "rbf", "sigmoid"]
                        break
                    else:
                        kern_dict = {1: "linear", 2: "poly", 3: "rbf",
                                     4: "sigmoid"}
                        try:
                            kern_params_int = \
                                list(map(int, list(user_input.split(","))))
                            if len(kern_params_int) > len(kern_dict):
                                raise Exception

                            kern_params = []
                            for each in kern_params_int:
                                if not kern_dict.get(each):
                                    raise Exception

                                kern_params.append(kern_dict.get(each))
                            break
                        except Exception:
                            print("Invalid input.")

                if break_early:
                    break

                params["kernel"] = kern_params
                print("kernels:", kern_params)

                while True:
                    print("\nEnter the list of kernel coefficients/gamma values",
                          "to try out.")
                    print("Example input: 0.001,0.0001")
                    user_input = input().lower()

                    if user_input == "q":
                        self.gridsearch = False
                        break_early = True
                        break

                    try:
                        gamma_params = \
                            list(map(float, list(user_input.split(","))))
                        if len(gamma_params) == 0:
                            raise Exception

                        for num in gamma_params:
                            if num <= 0:
                                raise Exception
                        break
                    except Exception:
                        print("Invalid input.")

                if break_early:
                    break

                params["gamma"] = gamma_params
                print("gammas:", gamma_params)

                while not is_nu:
                    print("\nEnter the list of regularization parameters to",
                          "try out.")
                    print("Example input: 1,10,100")
                    user_input = input().lower()

                    if user_input == "q":
                        self.gridsearch = False
                        break_early = True
                        break

                    try:
                        gamma_params = \
                            list(map(int, list(user_input.split(","))))
                        if len(gamma_params) == 0:
                            raise Exception

                        for num in gamma_params:
                            if num <= 0:
                                raise Exception

                        params["C"] = gamma_params
                        print("C values:", gamma_params)
                        break
                    except Exception:
                        print("Invalid input.")

                if break_early:
                    break

                print("\n= End of GridSearch inputs. =")
                self.gs_params = params
                best_params = self._run_gridsearch_classifier(is_nu)
                kernel = best_params["kernel"]
                gamma = best_params["gamma"]
                if not is_nu:
                    C = best_params["C"]
                break

            break_early = False

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

                    self.cv = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("cv =", self.cv)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nGraph the ROC curve? Only binary classification "
                          + "is supported (y/N): ").lower()
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

            print("graph_results =", self.graph_results)

            if break_early:
                break

            while is_nu:
                user_input = input("\nEnter a decimal for nu (0,1]: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input <= 0 or user_input > 1:
                        raise Exception

                    nu = user_input
                    break
                except Exception:
                    print("Invalid input.")

            if is_nu:
                print("nu =", nu)

            while not is_nu and not self.gridsearch:
                user_input = \
                    input("\nEnter a positive regularization parameter C: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input <= 0:
                        raise Exception

                    C = user_input
                    break
                except Exception:
                    print("Invalid input.")

            if not is_nu and not self.gridsearch:
                print("C =", C)

            if break_early:
                break

            while not self.gridsearch:
                print("\nWhich kernel type should be used?")
                user_input = \
                    input("Enter 1 for 'linear', 2 for 'poly', 3 for 'rbf', 4 "
                          + "for 'sigmoid', or 5 for 'precomputed': ")
                if user_input == "1":
                    kernel = "linear"
                    break
                elif user_input == "2":
                    kernel = "poly"
                    break
                elif user_input == "4":
                    kernel = "sigmoid"
                    break
                elif user_input == "5":
                    kernel = "recomputed"
                    break
                elif user_input in {"3", ""}:
                    break
                elif user_input.lower() == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            if not self.gridsearch:
                print("kernel =", kernel)

            if break_early:
                break

            while kernel == "poly":
                user_input = \
                    input("\nEnter the degree of the kernel function [0,): ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = int(user_input)
                    if user_input < 0:
                        raise Exception

                    degree = user_input
                    break
                except Exception:
                    print("Invalid input.")

            if kernel == "poly":
                print("degree =", degree)

            if break_early:
                break

            while kernel in {"rbf", "poly", "sigmoid"} and not self.gridsearch:
                print("\nSet the kernel coefficient.")
                user_input = input("Enter 1 for 'scale', or 2 for 'auto': ")
                if user_input == "2":
                    gamma = "auto"
                    break
                elif user_input in {"1", ""}:
                    break
                elif user_input.lower() == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            if kernel in {"rbf", "poly", "sigmoid"} and not self.gridsearch:
                print("gamma =", gamma)

            if break_early:
                break

            while kernel in {"poly", "sigmoid"}:
                user_input = input("\nEnter coef0, the independent term in the "
                                   + "kernel function [0,): ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input < 0:
                        raise Exception

                    coef0 = user_input
                    break
                except Exception:
                    print("Invalid input.")

            if kernel in {"poly", "sigmoid"}:
                print("coef0 =", coef0)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nUse the shrinking heuristic (Y/n)? ").lower()
                if user_input == "n":
                    shrinking = False
                    break
                elif user_input in {"y", ""}:
                    break
                elif user_input == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            print("shrinking =", shrinking)

            if break_early:
                break

            while True:
                user_input = input("\nEnter a positive number for the "
                                   + "tolerance for stopping criterion: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input <= 0:
                        raise Exception

                    tol = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("tol =", tol)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nEnter a positive kernel cache size in MB: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input <= 0:
                        raise Exception

                    cache_size = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("cache_size =", cache_size)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nAutomatically adjust class weights (y/N)? ").lower()
                if user_input == "y":
                    class_weight = "balanced"
                    break
                elif user_input in {"n", ""}:
                    break
                elif user_input == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            print("class_weight =", class_weight)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nEnter a positive maximum number of iterations, or "
                          + "press enter for no limit: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = int(user_input)
                    if user_input <= 0:
                        raise Exception

                    max_iter = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("max_iter =", max_iter)

            if break_early:
                break

            while True:
                print("\nSet the decision function.")
                user_input = \
                    input("Enter 1 for one-vs-rest, or 2 for one-vs-one: ")
                if user_input == "2":
                    decision_function_shape = "ovo"
                    break
                elif user_input in {"1", ""}:
                    break
                elif user_input.lower() == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            print("decision_function_shape =", decision_function_shape)

            if break_early:
                break

            while True:
                user_input = input("\nEnable tie-breaking (y/N)? ").lower()
                if user_input == "y":
                    break_ties = True
                    break
                elif user_input in {"n", ""}:
                    break
                elif user_input == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            print("break_ties =", break_ties)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nEnter a seed for the random number generator: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    random_state = int(user_input)
                    break
                except Exception:
                    print("Invalid input.")

            print("random_state =", random_state)

            if break_early:
                break

            while True:
                user_input = input("\nEnable verbose logging (y/N)? ").lower()
                if user_input == "y":
                    verbose = True
                    break
                elif user_input in {"n", "q", ""}:
                    break
                else:
                    print("Invalid input.")

            print("verbose =", verbose)
            break

        print("\n===========================================")
        print("= End of inputs; press enter to continue. =")
        input("===========================================\n")

        if is_nu:
            return NuSVC(nu=nu, kernel=kernel, degree=degree, gamma=gamma,
                         coef0=coef0, shrinking=shrinking,
                         probability=True, tol=tol, cache_size=cache_size,
                         class_weight=class_weight, verbose=verbose,
                         max_iter=max_iter,
                         decision_function_shape=decision_function_shape,
                         break_ties=break_ties, random_state=random_state)

        return SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
                   shrinking=shrinking, probability=True, tol=tol,
                   cache_size=cache_size, class_weight=class_weight,
                   verbose=verbose, max_iter=max_iter,
                   decision_function_shape=decision_function_shape,
                   break_ties=break_ties, random_state=random_state)

    def _run_gridsearch_classifier(self, is_nu):
        """Runs GridSearch with the parameters given in run_SVC() or
        run_nu_SVC(). Returns the best parameters."""
        if is_nu:
            clf = NuSVC()
        else:
            clf = SVC()

        acc_scorer = make_scorer(accuracy_score)

        if self.dataset_X_test is None:
            self._split_data()

        # Run GridSearch
        grid_obj = GridSearchCV(clf, self.gs_params, scoring=acc_scorer)
        grid_obj = grid_obj.fit(self.dataset_X_train, self.dataset_y_train)

        # Set the clf to the best combination of parameters
        clf = grid_obj.best_estimator_

        # Fit the best algorithm to the data
        clf.fit(self.dataset_X_train, self.dataset_y_train)
        predictions = clf.predict(self.dataset_X_test)
        self.gs_result = accuracy_score(self.dataset_y_test, predictions)

        # Return the best parameters
        print("\nBest GridSearch Parameters:\n", grid_obj.best_params_, "\n")
        return grid_obj.best_params_

    def _create_linear_SVC_model(self):
        """Runs UI for getting parameters and creating LinearSVC model."""
        print("\n==============================")
        print("= LinearSVC Parameter Inputs =")
        print("==============================\n")
        print("Default values:",
              "test_size = 0.25",
              "cv = 5",
              "penalty = 'l2'",
              "loss = 'squared_hinge'",
              "dual = True",
              "tol = 0.0001",
              "C = 1.0",
              "multi_class = 'ovr'",
              "fit_intercept = True",
              "intercept_scaling = 1",
              "class_weight = None",
              "random_state = None",
              "max_iter = 1000",
              "verbose = False", sep="\n")

        # Set defaults
        self.test_size = 0.25
        self.cv = None

        while True:
            user_input = input("\nUse default parameters (Y/n)? ").lower()
            if user_input in {"y", ""}:
                print("\n===========================================")
                print("= End of inputs; press enter to continue. =")
                input("===========================================\n")
                return LinearSVC()
            elif user_input == "n":
                break
            else:
                print("Invalid input.")

        print("\nIf you are unsure about a parameter, press enter to use its",
              "default value.")
        print("If you finish entering parameters early, enter 'q' to skip",
              "ahead.\n")

        # Set more defaults
        penalty = "l2"
        loss = "squared_hinge"
        dual = True
        tol = 0.0001
        C = 1.0
        multi_class = "ovr"
        fit_intercept = True
        intercept_scaling = 1
        class_weight = None
        random_state = None
        max_iter = 1000
        verbose = 0

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

                    self.cv = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("cv =", self.cv)

            if break_early:
                break

            while True:
                user_input = input("\nCalculate a y-intercept (Y/n)? ").lower()
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

            while fit_intercept:
                user_input = \
                    input("\nEnter a number for the intercept scaling factor: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    intercept_scaling = float(user_input)
                except Exception:
                    print("Invalid input.")

            if fit_intercept:
                print("intercept_scaling =", intercept_scaling)

            if break_early:
                break

            while True:
                print("\nSet the norm used in penalization.")
                user_input = input("Enter 1 for 'l1', or 2 for 'l2': ")
                if user_input == "1":
                    penalty = "l1"
                    break
                elif user_input in {"2", ""}:
                    break
                elif user_input.lower() == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            print("penalty =", penalty)

            if break_early:
                break

            while True:
                print("\nChoose a loss function.")
                user_input = \
                    input("Enter 1 for 'hinge', or 2 for 'squared_hinge': ")
                if user_input == "1":
                    loss = "hinge"
                    break
                elif user_input in {"2", ""}:
                    break
                elif user_input.lower() == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            print("loss =", loss)

            if break_early:
                break

            while True:
                print("\nShould the algorithm solve the duel or primal",
                      "optimization problem?")
                user_input = input("Enter 1 for dual, or 2 for primal: ")
                if user_input == "2":
                    dual = False
                    break
                elif user_input in {"1", ""}:
                    break
                elif user_input.lower() == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            print("dual =", dual)

            if break_early:
                break

            while True:
                user_input = input("\nEnter a positive tolerance for stopping "
                                   + "criterion: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input <= 0:
                        raise Exception

                    tol = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("tol =", tol)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nEnter a positive regularization parameter C: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input <= 0:
                        raise Exception

                    C = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("C =", C)

            if break_early:
                break

            while True:
                print("\nSet the multi-class strategy if there are more than",
                      "two classes.")
                user_input = input("Enter 1 for 'one-vs-rest', or 2 for "
                                   + "'Crammer-Singer': ")
                if user_input == "2":
                    multi_class = "crammer_singer"
                    break
                elif user_input in {"1", ""}:
                    break
                elif user_input.lower() == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            print("multi_class =", multi_class)

            if break_early:
                break

            while True:
                user_input = input("\nAutomatically adjust class weights "
                                   + "(y/N)? ").lower()
                if user_input == "y":
                    class_weight = "balanced"
                    break
                elif user_input in {"n", ""}:
                    break
                elif user_input == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            print("class_weight =", class_weight)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nEnter a positive maximum number of iterations: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = int(user_input)
                    if user_input <= 0:
                        raise Exception

                    max_iter = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("max_iter =", max_iter)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nEnter a seed for the random number generator: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    random_state = int(user_input)
                    break
                except Exception:
                    print("Invalid input.")

            print("random_state =", random_state)

            if break_early:
                break

            while True:
                user_input = input("\nEnable verbose logging (y/N)? ").lower()
                if user_input == "y":
                    verbose = True
                    break
                elif user_input in {"n", "q", ""}:
                    break
                else:
                    print("Invalid input.")

            print("verbose =", verbose)
            break

        print("\n===========================================")
        print("= End of inputs; press enter to continue. =")
        input("===========================================\n")

        return LinearSVC(penalty=penalty, loss=loss, dual=dual, tol=tol, C=C,
                         multi_class=multi_class, fit_intercept=fit_intercept,
                         intercept_scaling=intercept_scaling,
                         class_weight=class_weight, verbose=verbose,
                         random_state=random_state, max_iter=max_iter)

    def _create_SVR_model(self, is_nu):
        """Runs UI for getting parameters and creates SVR or NuSVR model."""
        if is_nu:
            print("\n==========================")
            print("= NuSVR Parameter Inputs =")
            print("==========================\n")
        else:
            print("\n========================")
            print("= SVR Parameter Inputs =")
            print("========================\n")

        print("Default values:", "test_size = 0.25", "cv = 5", sep="\n")
        if is_nu:
            print("nu = 0.5")
        else:
            print("epsilon = 0.1")

        print("kernel = 'rbf'",
              "degree = 3",
              "gamma = 'scale'",
              "coef0 = 0.0",
              "tol = 0.001",
              "C = 1.0",
              "shrinking = True",
              "cache_size = 200",
              "verbose = False",
              "max_iter = -1", sep="\n")

        # Set defaults
        self.test_size = 0.25
        self.cv = None

        while True:
            user_input = input("\nUse default parameters (Y/n)? ").lower()
            if user_input in {"y", ""}:
                print("\n===========================================")
                print("= End of inputs; press enter to continue. =")
                input("===========================================\n")
                if is_nu:
                    return NuSVR()
                return SVR()
            elif user_input == "n":
                break
            else:
                print("Invalid input.")

        print("\nIf you are unsure about a parameter, press enter to use its",
              "default value.")
        print("If you finish entering parameters early, enter 'q' to skip",
              "ahead.\n")

        # Set more defaults
        if is_nu:
            nu = 0.5
        else:
            epsilon = 0.1

        kernel = "rbf"
        degree = 3
        gamma = "scale"
        coef0 = 0.0
        tol = 0.001
        C = 1.0
        shrinking = True
        cache_size = 200
        verbose = False
        max_iter = -1

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

                    self.test_size = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("test_size =", self.test_size)

            if break_early:
                break

            while True:
                user_input = input("\nUse GridSearch to find the best "
                                   + "hyperparameters (y/N)? ").lower()
                if user_input == "q":
                    break_early = True
                    break
                elif user_input in {"n", "y", ""}:
                    break
                else:
                    print("Invalid input.")

            if break_early:
                break

            while user_input == "y":
                print("\n= GridSearch Parameter Inputs =\n")
                print("Enter 'q' to skip GridSearch.")
                self.gridsearch = True
                params = {}

                while True:
                    print("\nEnter the kernels to try out.")
                    print("Options: 1-'linear', 2-'poly', 3-'rbf', 4-'sigmoid'.",
                          "Enter 'all' for all options.")
                    print("Example input: 1,2,3")
                    user_input = input().lower()

                    if user_input == "q":
                        self.gridsearch = False
                        break_early = True
                        break
                    elif user_input == "all":
                        kern_params = ["linear", "poly", "rbf", "sigmoid"]
                        break
                    else:
                        kern_dict = {1: "linear", 2: "poly", 3: "rbf",
                                     4: "sigmoid"}
                        try:
                            kern_params_int = \
                                list(map(int, list(user_input.split(","))))
                            if len(kern_params_int) > len(kern_dict):
                                raise Exception

                            kern_params = []
                            for each in kern_params_int:
                                if not kern_dict.get(each):
                                    raise Exception

                                kern_params.append(kern_dict.get(each))
                            break
                        except Exception:
                            print("Invalid input.")

                if break_early:
                    break

                params["kernel"] = kern_params
                print("kernels:", kern_params)

                while is_nu:
                    print("\nEnter a list of decimals for nu.")
                    print("Example input: 0.1,0.2,0.3")
                    user_input = input().lower()

                    if user_input == "q":
                        self.gridsearch = False
                        break_early = True
                        break

                    try:
                        nu_params = \
                            list(map(float, list(user_input.split(","))))
                        if len(nu_params) == 0:
                            raise Exception

                        for num in nu_params:
                            if num <= 0:
                                raise Exception
                        params["nu"] = nu_params
                        print("nu values:", nu_params)
                        break
                    except Exception:
                        print("Invalid input.")

                while not is_nu:
                    print("\nEnter epsilons (the ranges from an actual value",
                          "where penalties aren't applied) to try out.")
                    print("Example input: 0.1,0.2,0.3")
                    user_input = input().lower()

                    if user_input == "q":
                        self.gridsearch = False
                        break_early = True
                        break

                    try:
                        eps_params = \
                            list(map(float, list(user_input.split(","))))
                        if len(eps_params) == 0:
                            raise Exception

                        for num in eps_params:
                            if num <= 0:
                                raise Exception
                        params["epsilon"] = eps_params
                        print("epsilon values:", eps_params)
                        break
                    except Exception:
                        print("Invalid input.")

                if break_early:
                    break

                while not is_nu:
                    print("\nEnter the list of regularization parameters to",
                          "try out.")
                    print("Example input: 1,10,100")
                    user_input = input().lower()

                    if user_input == "q":
                        self.gridsearch = False
                        break_early = True
                        break

                    try:
                        gamma_params = \
                            list(map(int, list(user_input.split(","))))
                        if len(gamma_params) == 0:
                            raise Exception

                        for num in gamma_params:
                            if num <= 0:
                                raise Exception

                        params["C"] = gamma_params
                        print("C values:", gamma_params)
                        break
                    except Exception:
                        print("Invalid input.")

                if break_early:
                    break

                print("\n= End of GridSearch inputs. =")
                self.gs_params = params
                best_params = self._run_gridsearch_regressor(is_nu)
                kernel = best_params["kernel"]
                if is_nu:
                    nu = best_params["nu"]
                else:
                    C = best_params["C"]
                    epsilon = best_params["epsilon"]
                break

            break_early = False

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

                    self.cv = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("cv =", self.cv)

            if break_early:
                break

            while is_nu and not self.gridsearch:
                user_input = input("\nEnter a decimal for nu (0,1]: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input <= 0 or user_input > 1:
                        raise Exception

                    nu = user_input
                    break
                except Exception:
                    print("Invalid input.")

            if is_nu and not self.gridsearch:
                print("nu =", nu)

            while not is_nu and not self.gridsearch:
                user_input = \
                    input("\nEnter a positive epsilon, the range from an actual "
                          + "value where penalties aren't applied: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input <= 0:
                        raise Exception

                    epsilon = user_input
                    break
                except Exception:
                    print("Invalid input.")

            if not is_nu and not self.gridsearch:
                print("epsilon =", epsilon)

            if break_early:
                break

            while not self.gridsearch:
                print("\nWhich kernel type should be used?")
                user_input = \
                    input("Enter 1 for 'linear', 2 for 'poly', 3 for 'rbf', 4 "
                          + "for 'sigmoid', or 5 for 'precomputed': ")
                if user_input == "1":
                    kernel = "linear"
                    break
                elif user_input == "2":
                    kernel = "poly"
                    break
                elif user_input == "4":
                    kernel = "sigmoid"
                    break
                elif user_input == "5":
                    kernel = "recomputed"
                    break
                elif user_input in {"3", ""}:
                    break
                elif user_input.lower() == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            if not self.gridsearch:
                print("kernel =", kernel)

            if break_early:
                break

            while kernel == "poly":
                user_input = \
                    input("\nEnter the degree of the kernel function [0,): ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = int(user_input)
                    if user_input < 0:
                        raise Exception

                    degree = user_input
                    break
                except Exception:
                    print("Invalid input.")

            if kernel == "poly":
                print("degree =", degree)

            if break_early:
                break

            while kernel in {"rbf", "poly", "sigmoid"}:
                print("\nSet the kernel coefficient.")
                user_input = input("Enter 1 for 'scale', or 2 for 'auto': ")
                if user_input == "2":
                    gamma = "auto"
                    break
                elif user_input in {"1", ""}:
                    break
                elif user_input.lower() == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            if kernel in {"rbf", "poly", "sigmoid"}:
                print("gamma =", gamma)

            if break_early:
                break

            while not self.gridsearch or is_nu:
                user_input = \
                    input("\nEnter a positive regularization parameter C: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input <= 0:
                        raise Exception

                    C = user_input
                    break
                except Exception:
                    print("Invalid input.")

            if not self.gridsearch or is_nu:
                print("C =", C)

            if break_early:
                break

            while kernel in {"poly", "sigmoid"}:
                user_input = input("\nEnter coef0, the independent term in the "
                                   + "kernel function [0,): ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input < 0:
                        raise Exception

                    coef0 = user_input
                    break
                except Exception:
                    print("Invalid input.")

            if kernel in {"poly", "sigmoid"}:
                print("coef0 =", coef0)

            if break_early:
                break

            while True:
                user_input = input("\nEnter a positive tolerance for stopping "
                                   + "criterion: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input <= 0:
                        raise Exception

                    tol = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("tol =", tol)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nUse the shrinking heuristic (Y/n)? ").lower()
                if user_input == "n":
                    shrinking = False
                    break
                elif user_input in {"y", ""}:
                    break
                elif user_input == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            print("shrinking =", shrinking)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nEnter a positive kernel cache size in MB: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input <= 0:
                        raise Exception

                    cache_size = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("cache_size =", cache_size)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nEnter a positive maximum number of iterations, or "
                          + "press enter for no limit: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = int(user_input)
                    if user_input <= 0:
                        raise Exception

                    max_iter = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("max_iter =", max_iter)

            if break_early:
                break

            while True:
                user_input = input("\nEnable verbose logging (y/N)? ").lower()
                if user_input == "y":
                    verbose = True
                    break
                elif user_input in {"n", "q", ""}:
                    break
                else:
                    print("Invalid input.")

            print("verbose =", verbose)
            break

        print("\n===========================================")
        print("= End of inputs; press enter to continue. =")
        input("===========================================\n")

        if is_nu:
            return NuSVR(nu=nu, C=C, kernel=kernel, degree=degree, gamma=gamma,
                         coef0=coef0, shrinking=shrinking, tol=tol,
                         cache_size=cache_size, verbose=verbose,
                         max_iter=max_iter)

        return SVR(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
                   tol=tol, C=C, epsilon=epsilon, shrinking=shrinking,
                   cache_size=cache_size, verbose=verbose, max_iter=max_iter)

    def _run_gridsearch_regressor(self, is_nu):
        """Runs GridSearch with the parameters given in run_SVR() or
        run_nu_SVR(). Returns the best parameters."""
        if is_nu:
            clf = NuSVR()
        else:
            clf = SVR()

        if self.dataset_X_test is None:
            self._split_data()

        # Run GridSearch
        grid_obj = GridSearchCV(clf, self.gs_params, scoring="r2")
        grid_obj = grid_obj.fit(self.dataset_X_train, self.dataset_y_train)

        # Set the clf to the best combination of parameters
        clf = grid_obj.best_estimator_

        # Fit the best algorithm to the data
        clf.fit(self.dataset_X_train, self.dataset_y_train)
        self.gs_result = clf.score(self.dataset_X_test, self.dataset_y_test)

        # Return the best parameters
        print("\nBest GridSearch Parameters:\n", grid_obj.best_params_, "\n")
        return grid_obj.best_params_

    def _create_linear_SVR_model(self):
        """Runs UI for getting parameters and creates LinearSVR model."""
        print("\n==============================")
        print("= LinearSVR Parameter Inputs =")
        print("==============================\n")
        print("Default values:",
              "test_size = 0.25",
              "cv = 5",
              "epsilon = 0.0",
              "tol = 0.0001",
              "C = 1.0",
              "loss = 'epsilon_insensitive'",
              "fit_intercept = True",
              "intercept_scaling = 1.0",
              "dual = True",
              "random_state = None",
              "max_iter = 1000",
              "verbose = False", sep="\n")

        # Set defaults
        self.test_size = 0.25
        self.cv = None

        while True:
            user_input = input("\nUse default parameters (Y/n)? ").lower()
            if user_input in {"y", ""}:
                print("\n===========================================")
                print("= End of inputs; press enter to continue. =")
                input("===========================================\n")
                return LinearSVR()
            elif user_input == "n":
                break
            else:
                print("Invalid input.")

        print("\nIf you are unsure about a parameter, press enter to use its",
              "default value.")
        print("If you finish entering parameters early, enter 'q' to skip",
              "ahead.\n")

        # Set more defaults
        epsilon = 0.0
        tol = 0.0001
        C = 1.0
        loss = "epsilon_insensitive"
        fit_intercept = True
        intercept_scaling = 1.0
        dual = True
        random_state = None
        max_iter = 1000
        verbose = 0

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

                    self.cv = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("cv =", self.cv)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nEnter a positive epsilon, the range from an actual "
                          + "value where penalties aren't applied: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input <= 0:
                        raise Exception

                    epsilon = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("epsilon =", epsilon)

            if break_early:
                break

            while True:
                user_input = input("\nEnter a positive tolerance for stopping "
                                   + "criterion: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input <= 0:
                        raise Exception

                    tol = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("tol =", tol)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nEnter a positive regularization parameter C: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input <= 0:
                        raise Exception

                    C = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("C =", C)

            if break_early:
                break

            while True:
                print("\nChoose a loss function.")
                user_input = \
                    input("\nEnter 1 for 'epsilon_insensitive', or 2 for "
                          + "'squared_epsilon_insensitive': ")
                if user_input == "2":
                    loss = "squared_epsilon_insensitive"
                    break
                elif user_input in {"1", ""}:
                    break
                elif user_input.lower() == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            print("loss =", loss)

            if break_early:
                break

            while True:
                user_input = input("\nCalculate a y-intercept (Y/n)? ").lower()
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

            while fit_intercept:
                user_input = \
                    input("\nEnter a number for the intercept scaling factor: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    intercept_scaling = float(user_input)
                except Exception:
                    print("Invalid input.")

            if fit_intercept:
                print("intercept_scaling =", intercept_scaling)

            if break_early:
                break

            while True:
                print("\nShould the algorithm solve the duel or primal",
                      "optimization problem?")
                user_input = input("Enter 1 for dual, or 2 for primal: ")
                if user_input == "2":
                    dual = False
                    break
                elif user_input in {"1", ""}:
                    break
                elif user_input.lower() == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            print("dual =", dual)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nEnter a positive maximum number of iterations: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = int(user_input)
                    if user_input <= 0:
                        raise Exception

                    max_iter = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("max_iter =", max_iter)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nEnter a seed for the random number generator: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    random_state = int(user_input)
                    break
                except Exception:
                    print("Invalid input.")

            print("random_state =", random_state)

            if break_early:
                break

            while True:
                user_input = input("\nEnable verbose logging (y/N)? ").lower()
                if user_input == "y":
                    verbose = True
                    break
                elif user_input in {"n", "q", ""}:
                    break
                else:
                    print("Invalid input.")

            print("verbose =", verbose)
            break

        print("\n===========================================")
        print("= End of inputs; press enter to continue. =")
        input("===========================================\n")

        return LinearSVR(epsilon=epsilon, tol=tol, C=C, loss=loss,
                         fit_intercept=fit_intercept,
                         intercept_scaling=intercept_scaling, dual=dual,
                         verbose=verbose, random_state=random_state,
                         max_iter=max_iter)

    def _output_classifier_results(self, model):
        """Outputs model metrics after a classifier model finishes running."""
        if model == "SVC":
            print("\n===============")
            print("= SVC Results =")
            print("===============\n")

            print("{:<20} {:<20}".format("Accuracy:", self.accuracy_SVC))

            if self.bin:
                print("\n{:<20} {:<20}".format("ROC AUC:", self.roc_auc_SVC))
            else:
                print("\nConfusion Matrix:\n", self.confusion_matrix_SVC)

            print("\nCross Validation Scores:", self.cross_val_scores_SVC)

            if self.gridsearch:
                print("\n{:<20} {:<20}".format("GridSearch Score:",
                                               self.gs_result))

            if self.bin and self.graph_results:
                plt.plot(self.fpr, self.tpr, label="data 1")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend(loc=4)
                plt.show()

            print("\n\nCall predict_SVC() to make predictions for new data.")
        elif model == "NuSVC":
            print("\n=================")
            print("= NuSVC Results =")
            print("=================\n")

            print("{:<20} {:<20}".format("Accuracy:", self.accuracy_nu_SVC))
            if self.bin:
                print("\n{:<20} {:<20}".format(
                    "ROC AUC:", self.roc_auc_nu_SVC))
            else:
                print("\nConfusion Matrix:\n", self.confusion_matrix_nu_SVC)

            print("\nCross Validation Scores:", self.cross_val_scores_nu_SVC)
            if self.gridsearch:
                print("\n{:<20} {:<20}".format("GridSearch Score:",
                                               self.gs_result))

            if self.bin and self.graph_results:
                plt.plot(self.fpr, self.tpr, label="data 1")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend(loc=4)
                plt.show()

            print("\n\nCall predict_nu_SVC() to make predictions for new data.")
        else:
            print("\n=====================")
            print("= LinearSVC Results =")
            print("=====================\n")

            print("{:<20} {:<20}".format(
                "Accuracy:", self.accuracy_linear_SVC))
            print("\nCross Validation Scores:",
                  self.cross_val_scores_linear_SVC)

            print("\n\nCall predict_linear_SVC() to make predictions for",
                  "new data.")

        print("\n===================")
        print("= End of results. =")
        print("===================\n")

    def _output_regressor_results(self, model):
        """Outputs model metrics after a regressor model finishes running."""
        if model == "SVR":
            print("\n===============")
            print("= SVR Results =")
            print("===============\n")

            print("{:<20} {:<20}".format("Mean Squared Error:",
                                         self.mean_squared_error_SVR))
            print("\n{:<20} {:<20}".format("R2 Score:", self.r2_score_SVR))
            print("\n{:<20} {:<20}".format("R Score:", str(self.r_score_SVR)))
            print("\nCross Validation Scores", self.cross_val_scores_SVR)

            if self.gridsearch:
                print("\n{:<20} {:<20}".format("GridSearch Score:",
                                               self.gs_result))

            print("\n\nCall predict_SVR() to make predictions for new data.")
        elif model == "NuSVR":
            print("\n=================")
            print("= NuSVR Results =")
            print("=================\n")

            print("{:<20} {:<20}".format("Mean Squared Error:",
                                         self.mean_squared_error_nu_SVR))
            print("\n{:<20} {:<20}".format("R2 Score:", self.r2_score_nu_SVR))
            print("\n{:<20} {:<20}".format(
                "R Score:", str(self.r_score_nu_SVR)))
            print("\nCross Validation Scores:", self.cross_val_scores_nu_SVR)

            if self.gridsearch:
                print("\n{:<20} {:<20}".format("GridSearch Score:",
                                               self.gs_result))

            print("\n\nCall predict_nu_SVR() to make predictions for new data.")
        else:
            print("\n=====================")
            print("= LinearSVR Results =")
            print("=====================\n")

            print("{:<20} {:<20}".format("Mean Squared Error:",
                                         self.mean_squared_error_linear_SVR))
            print("\n{:<20} {:<20}".format("R2 Score:",
                                           self.r2_score_linear_SVR))
            print("\n{:<20} {:<20}".format("R Score:",
                                           str(self.r_score_linear_SVR)))
            print("\nCross Validation Scores:\n",
                  self.cross_val_scores_linear_SVR)

            print("\n\nCall predict_linear_SVR() to make predictions for new data.")

        print("\n===================")
        print("= End of results. =")
        print("===================\n")

    def _split_data(self):
        """Helper method for splitting attributes and labels into
        training and testing sets.

        This method runs under the assumption that all relevant instance
        data has been checked for correctness.
        """
        self.dataset_X_train, self.dataset_X_test, self.dataset_y_train, \
            self.dataset_y_test = train_test_split(self.attributes, self.labels,
                                                   test_size=self.test_size)

    def _check_inputs(self):
        """Verifies if instance data is ready for use in SVM model."""
        # Check if attributes exists
        if self.attributes is None:
            print("attributes is missing; call set_attributes(new_attributes)",
                  "to fix this! new_attributes should be a populated dataset",
                  "of independent variables.")
            return False

        # Check if labels exists
        if self.labels is None:
            print("labels is missing; call set_labels(new_labels) to fix this!",
                  "new_labels should be a populated dataset of classes.")
            return False

        # Check if attributes and labels have same number of rows (samples)
        if self.attributes.shape[0] != self.labels.shape[0]:
            print("attributes and labels don't have the same number of rows.",
                  "Make sure the number of samples in each dataset matches!")
            return False

        return True
