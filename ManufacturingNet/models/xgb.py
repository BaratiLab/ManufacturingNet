"""XGBoost trains a XGBoost model on the given dataset. Before
training, the user is prompted for parameter input. After training,
model metrics are displayed, and the user can make new predictions.
Classification and regression are both supported.

View the documentation at https://manufacturingnet.readthedocs.io/.
"""

from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, make_scorer,
                             mean_squared_error, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from xgboost import XGBClassifier, XGBRegressor


class XGBoost:
    """Class framework for XGBoost's classification and regression
    functionality.
    """

    def __init__(self, attributes=None, labels=None):
        """Initializes an XGBoost object."""
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

        self.regressor = None
        self.mean_squared_error = None
        self.r2_score = None
        self.r_score = None
        self.cross_val_scores_regressor = None
        self.feature_importances_regressor = None

        self.classifier = None
        self.accuracy = None
        self.confusion_matrix = None
        self.roc_auc = None
        self.classes = None
        self.cross_val_scores_classifier = None
        self.feature_importances_classifier = None

    # Accessor methods

    def get_attributes(self):
        """Accessor method for attributes."""
        return self.attributes

    def get_labels(self):
        """Accessor method for labels."""
        return self.labels

    def get_regressor(self):
        """Accessor method for regressor."""
        return self.regressor

    def get_classifier(self):
        """Accessor method for classifier."""
        return self.classifier

    def get_classes(self):
        """Accessor method for classes."""
        return self.classes

    def get_accuracy(self):
        """Accessor method for accuracy."""
        return self.accuracy

    def get_cross_val_scores_classifier(self):
        """Accessor method for cross_val_scores_classifier."""
        return self.cross_val_scores_classifier

    def get_feature_importances_classifier(self):
        """Accessor method for feature_importances_classifier."""
        return self.feature_importances_classifier

    def get_cross_val_scores_regressor(self):
        """Accessor method for cross_val_scores_regressor."""
        return self.cross_val_scores_regressor

    def get_feature_importances_regressor(self):
        """Accessor method for feature_importances_regressor."""
        return self.feature_importances_regressor

    def get_mean_squared_error(self):
        """Accessor method for mean_squared_error."""
        return self.mean_squared_error

    def get_r2_score(self):
        """Accessor method for r2_score."""
        return self.r2_score

    def get_r_score(self):
        """Accessor method for r_score."""
        return self.r_score

    def get_confusion_matrix(self):
        """Accessor method for confusion_matrix."""
        return self.confusion_matrix

    def get_roc_auc(self):
        """Accessor method for roc-auc."""
        return self.roc_auc

    # Modifier methods

    def set_attributes(self, new_attributes=None):
        """Modifier method for attributes."""
        self.attributes = new_attributes

    def set_labels(self, new_labels=None):
        """Modifier method for labels."""
        self.labels = new_labels

    # Wrapper for regression functionality

    def run_regressor(self):
        """Runs XGBRegressor model."""
        if self._check_inputs():
            # Initialize regressor
            self.regressor = self._create_model(classifier=False)

            # Split dataset into testing and training data
            dataset_X_train, dataset_X_test, dataset_y_train, dataset_y_test = \
                train_test_split(self.attributes, self.labels,
                                 test_size=self.test_size)

            # Train the model and get resultant coefficients
            # Handle exception if arguments aren't correct
            try:
                self.regressor.fit(dataset_X_train, dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the regression",
                      "model. Check your inputs and try again.")
                print("Does labels only contain quantitative data?")
                print("Here is the exception message:")
                print(e)
                self.regressor = None
                return

            # Make predictions using testing set
            y_prediction = self.regressor.predict(dataset_X_test)

            # Metrics
            self.mean_squared_error = mean_squared_error(dataset_y_test,
                                                         y_prediction)
            self.r2_score = self.regressor.score(
                dataset_X_test, dataset_y_test)
            if self.r2_score >= 0:
                self.r_score = sqrt(self.r2_score)

            self.cross_val_scores_regressor = \
                cross_val_score(self.regressor, self.attributes, self.labels,
                                cv=self.cv)
            try:
                self.feature_importances_regressor = \
                    self.regressor.feature_importances_
            except Exception:
                self.feature_importances_regressor = \
                    "Not supported for selected booster"

            # Output results
            self._output_regressor_results()

    def predict_regressor(self, dataset_X=None):
        """Predicts the output of each datapoint in dataset_X using the
        regressor model. Returns the predictions.
        """

        # Check that run_regressor() has already been called
        if self.regressor is None:
            print("The regressor model seems to be missing.",
                  "Have you called run_regressor() yet?")
            return None

        # Try to make the prediction
        # Handle exception if dataset_X isn't a valid input
        try:
            y_prediction = self.regressor.predict(dataset_X)
        except Exception as e:
            print("The model failed to run. Check your inputs and try again.")
            print("Here is the exception message:")
            print(e)
            return None

        print("\nXGBRegressor Predictions:\n", y_prediction, "\n")
        return y_prediction

    # Wrapper for classification functionality

    def run_classifier(self):
        """Runs XGBClassifier model."""
        if self._check_inputs():
            # Initialize classifier
            self.classifier = self._create_model(classifier=True)

            # Split dataset into testing and training data
            dataset_X_train, dataset_X_test, dataset_y_train, dataset_y_test = \
                train_test_split(self.attributes, self.labels,
                                 test_size=self.test_size)

            # Train the model and get resultant coefficients
            # Handle exception if arguments aren't correct
            try:
                self.classifier.fit(dataset_X_train, np.ravel(dataset_y_train))
            except Exception as e:
                print("An exception occurred while training the",
                      "classification model. Check your inputs and try again.")
                print("Here is the exception message:")
                print(e)
                self.classifier = None
                return

            # Metrics
            y_prediction = self.classifier.predict(dataset_X_test)
            probas = self.classifier.predict_proba(dataset_X_test)
            self.classes = self.classifier.classes_
            self.accuracy = accuracy_score(dataset_y_test, y_prediction)

            # If classification is binary, calculate roc_auc
            if probas.shape[1] == 2:
                self.bin = True
                self.roc_auc = roc_auc_score(y_prediction, probas[::, 1])
                self.fpr, self.tpr, _ = roc_curve(
                    dataset_y_test, probas[::, 1])
            # Else, calculate confusion_matrix
            else:
                self.confusion_matrix = \
                    confusion_matrix(dataset_y_test, y_prediction)

            self.cross_val_scores_classifier = \
                cross_val_score(self.classifier, self.attributes, self.labels,
                                cv=self.cv)
            try:
                self.feature_importances_classifier = \
                    self.classifier.feature_importances_
            except Exception:
                self.feature_importances_classifier = \
                    "Not supported for selected booster"

            # Output results
            self._output_classifier_results()

    def predict_classifier(self, dataset_X=None):
        """Classifies each datapoint in dataset_X using the classifier model.
        Returns the predicted classifications.
        """
        # Check that run_classifier() has already been called
        if self.classifier is None:
            print("The classifier model seems to be missing.",
                  "Have you called run_classifier() yet?")
            return None

        # Try to make the prediction
        # Handle exception if dataset_X isn't a valid input
        try:
            y_prediction = self.classifier.predict(dataset_X)
        except Exception as e:
            print("The model failed to run. Check your inputs and try again.")
            print("Here is the exception message:")
            print(e)
            return None

        print("\nXGBClassifier Predictions:\n", y_prediction, "\n")
        return y_prediction

    # Helper methods

    def _create_model(self, classifier):
        """Runs UI for getting parameters and creating classifier or
        regression model.
        """
        if classifier:
            print("\n==================================")
            print("= XGBClassifier Parameter Inputs =")
            print("==================================\n")
        else:
            print("\n=================================")
            print("= XGBRegressor Parameter Inputs =")
            print("=================================\n")

        print("Default values:", "test_size = 0.25", "cv = 5", sep="\n")
        if classifier:
            print("graph_results = False",
                  "objective = 'binary:logistic'", sep="\n")
        else:
            print("objective = 'reg:squarederror'")

        print("n_estimators = 100",
              "max_depth = 3",
              "learning_rate = 0.1",
              "booster = 'gbtree'",
              "n_jobs = 1",
              "nthread = None",
              "gamma = 0",
              "min_child_weight = 1",
              "max_delta_step = 0",
              "subsample = 1",
              "colsample_bytree = 1",
              "colsample_bylevel = 1",
              "reg_alpha = 0",
              "reg_lambda = 1",
              "scale_pos_weight = 1",
              "base_score = 0.5",
              "random_state = 42",
              "missing = None",
              "verbosity = False", sep="\n")

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
                if classifier:
                    return XGBClassifier()
                return XGBRegressor()
            elif user_input == "n":
                break
            else:
                print("Invalid input.")

        print("\nIf you are unsure about a parameter, press enter to use its",
              "default value.")
        print("If you finish entering parameters early, enter 'q' to skip",
              "ahead.\n")

        # Set more defaults; same parameters for classification and regression
        if classifier:
            objective = "binary:logistic"
        else:
            objective = "reg:squarederror"

        n_estimators = 100
        max_depth = 3
        learning_rate = 0.1
        booster = "gbtree"
        n_jobs = 1
        nthread = None
        gamma = 0
        min_child_weight = 1
        max_delta_step = 0
        subsample = 1
        colsample_bytree = 1
        colsample_bylevel = 1
        reg_alpha = 0
        reg_lambda = 1
        scale_pos_weight = 1
        base_score = 0.5
        random_state = 42
        missing = None
        verbosity = 0

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
                    print("\nEnter the types of boosters.")
                    print("Options: 1-'gbtree', 2-'gblinear' or 3-'dart'. Enter",
                          "'all' for all options.")
                    print("Example input: 1,2,3")
                    user_input = input().lower()

                    if user_input == "q":
                        self.gridsearch = False
                        break_early = True
                        break
                    elif user_input == "all":
                        boost_params = ["gbtree", "gblinear", "dart"]
                        break
                    else:
                        boost_dict = {1: "gbtree", 2: "gblinear", 3: "dart"}
                        try:
                            boost_params_int = \
                                list(map(int, list(user_input.split(","))))
                            if len(boost_params_int) > len(boost_dict):
                                raise Exception

                            boost_params = []
                            for each in boost_params_int:
                                if not boost_dict.get(each):
                                    raise Exception

                                boost_params.append(boost_dict.get(each))
                            break
                        except Exception:
                            print("Invalid input.")

                if break_early:
                    break

                params["booster"] = boost_params
                print("boosters:", boost_params)

                while True:
                    print("\nEnter a list of learning rates to try out.")
                    print("Example input: 0.1,0.01,0.001")
                    user_input = input().lower()

                    if user_input == "q":
                        self.gridsearch = False
                        break_early = True
                        break

                    try:
                        lr_params = \
                            list(map(float, list(user_input.split(","))))
                        if len(lr_params) == 0:
                            raise Exception

                        for num in lr_params:
                            if num <= 0:
                                raise Exception
                        break
                    except Exception:
                        print("Invalid input.")

                if break_early:
                    break

                params["learning_rate"] = lr_params
                print("learning_rates:", lr_params)

                while True:
                    print("\nEnter a list of gamma values/minimum loss",
                          "reductions to try out.")
                    print("Example input: 0.5,1,1.5")
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

                while True:
                    print("\nEnter a list of number of trees to try out.")
                    print("Example input: 1,2,3")
                    user_input = input().lower()

                    if user_input == "q":
                        self.gridsearch = False
                        break_early = True
                        break

                    try:
                        ntrees_params = \
                            list(map(int, list(user_input.split(","))))
                        if len(ntrees_params) == 0:
                            raise Exception

                        for num in ntrees_params:
                            if num <= 0:
                                raise Exception
                        break
                    except Exception:
                        print("Invalid input.")

                if break_early:
                    break

                params["n_estimators"] = ntrees_params
                print("n_estimators:", ntrees_params)

                while True:
                    print("\nEnter a list of max tree depths to try out.")
                    print("Example input: 1,2,3")
                    user_input = input().lower()

                    if user_input == "q":
                        self.gridsearch = False
                        break_early = True
                        break

                    try:
                        mdepth_params = \
                            list(map(int, list(user_input.split(","))))
                        if len(mdepth_params) == 0:
                            raise Exception

                        for num in mdepth_params:
                            if num <= 0:
                                raise Exception
                        break
                    except Exception:
                        print("Invalid input.")

                if break_early:
                    break

                params["max_depth"] = mdepth_params
                print("max_depths:", mdepth_params)

                print("\n= End of GridSearch inputs. =\n")
                self.gs_params = params
                best_params = self._run_gridsearch(classifier)
                booster = best_params["booster"]
                gamma = best_params["gamma"]
                learning_rate = best_params["learning_rate"]
                max_depth = best_params["max_depth"]
                n_estimators = best_params["n_estimators"]
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

            while classifier:
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

            if classifier:
                print("graph_results =", self.graph_results)

            if break_early:
                break

            while not self.gridsearch:
                user_input = input("\nEnter the number of gradient-boosted "
                                   + "trees to use: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = int(user_input)
                    if user_input <= 0:
                        raise Exception

                    n_estimators = user_input
                    break
                except Exception:
                    print("Invalid input.")

            if not self.gridsearch:
                print("n_estimators =", n_estimators)

            if break_early:
                break

            while not self.gridsearch:
                user_input = input("\nEnter a positive maximum tree depth: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = int(user_input)
                    if user_input <= 0:
                        raise Exception

                    max_depth = user_input
                    break
                except Exception:
                    print("Invalid input.")

            if not self.gridsearch:
                print("max_depth =", max_depth)

            if break_early:
                break

            while not self.gridsearch:
                user_input = input("\nEnter a positive learning rate: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input <= 0:
                        raise Exception

                    learning_rate = user_input
                    break
                except Exception:
                    print("Invalid input.")

            if not self.gridsearch:
                print("learning_rate =", learning_rate)

            if break_early:
                break

            while not self.gridsearch:
                print("\nWhich booster should be used?")
                user_input = input("Enter 1 for 'gbtree', 2 for 'gblinear', or "
                                   + "3 for 'dart': ").lower()
                if user_input == "2":
                    booster = "gblinear"
                    break
                elif user_input == "3":
                    booster = "dart"
                    break
                elif user_input in {"1", ""}:
                    break
                elif user_input == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            if not self.gridsearch:
                print("booster =", booster)

            if break_early:
                break

            while True:
                print("\nEnter a positive number of CPU cores to use.")
                user_input = input("Enter -1 to use all cores: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = int(user_input)
                    if user_input <= 0 and user_input != -1:
                        raise Exception

                    n_jobs = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("n_jobs =", n_jobs)

            if break_early:
                break

            while not self.gridsearch:
                user_input = \
                    input("\nEnter gamma, the minimum loss reduction "
                          + "needed to further partition a leaf node [0,): ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input < 0:
                        raise Exception

                    gamma = user_input
                    break
                except Exception:
                    print("Invalid input.")

            if not self.gridsearch:
                print("gamma =", gamma)

            if break_early:
                break

            while True:
                user_input = input("\nEnter min_child_weight [0,): ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input < 0:
                        raise Exception

                    min_child_weight = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("min_child_weight =", min_child_weight)

            if break_early:
                break

            while True:
                user_input = input("\nEnter max_delta_step [0,): ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = int(user_input)
                    if user_input < 0:
                        raise Exception

                    max_delta_step = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("max_delta_step =", max_delta_step)

            if break_early:
                break

            while True:
                user_input = input("\nEnter the subsample ratio (0,1]: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input <= 0 or user_input > 1:
                        raise Exception

                    subsample = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("subsample =", subsample)

            if break_early:
                break

            while True:
                user_input = input("\nEnter colsample_bytree, the subsample "
                                   + "column ratio for all trees (0,1]: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input <= 0 or user_input > 1:
                        raise Exception

                    colsample_bytree = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("colsample_bytree =", colsample_bytree)

            if break_early:
                break

            while True:
                user_input = input("\nEnter colsample_bylevel, the subsample "
                                   + "column ratio for all levels (0,1]: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input <= 0 or user_input > 1:
                        raise Exception

                    colsample_bylevel = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("colsample_bylevel =", colsample_bylevel)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nEnter alpha, the L1 regularization term [0,): ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input < 0:
                        raise Exception

                    reg_alpha = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("reg_alpha =", reg_alpha)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nEnter lambda, the L2 regularization term [0,): ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input < 0:
                        raise Exception

                    reg_lambda = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("reg_lambda =", reg_lambda)

            if break_early:
                break

            while True:
                user_input = input("\nEnter scale_pos_weight to control class "
                                   + "balancing [0,): ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input < 0:
                        raise Exception

                    scale_pos_weight = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("scale_pos_weight =", scale_pos_weight)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nEnter the initial prediction score (0,1): ")

                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input <= 0 or user_input >= 1:
                        raise Exception

                    base_score = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("base_score =", base_score)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nEnter an integer for the random number seed: ")
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
                user_input = input("\nEnable verbose output during training "
                                   + "(y/N)? ").lower()
                if user_input == "y":
                    verbosity = 1
                    break
                elif user_input in {"n", "q", ""}:
                    break
                else:
                    print("Invalid input.")

            print("verbose =", bool(verbosity))
            break

        print("\n===========================================")
        print("= End of inputs; press enter to continue. =")
        input("===========================================\n")

        if classifier:
            return XGBClassifier(max_depth=max_depth,
                                 learning_rate=learning_rate,
                                 n_estimators=n_estimators, objective=objective,
                                 booster=booster, n_jobs=n_jobs,
                                 nthread=nthread, gamma=gamma,
                                 min_child_weight=min_child_weight,
                                 max_delta_step=max_delta_step,
                                 subsample=subsample,
                                 colsample_bytree=colsample_bytree,
                                 colsample_bylevel=colsample_bylevel,
                                 reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                                 scale_pos_weight=scale_pos_weight,
                                 base_score=base_score,
                                 random_state=random_state, missing=missing,
                                 verbosity=verbosity)

        return XGBRegressor(max_depth=max_depth, learning_rate=learning_rate,
                            n_estimators=n_estimators, objective=objective,
                            booster=booster, n_jobs=n_jobs, nthread=nthread,
                            gamma=gamma, min_child_weight=min_child_weight,
                            max_delta_step=max_delta_step, subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            colsample_bylevel=colsample_bylevel,
                            reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                            scale_pos_weight=scale_pos_weight,
                            base_score=base_score, random_state=random_state,
                            missing=missing, verbosity=verbosity)

    def _output_classifier_results(self):
        """Outputs model metrics after run_classifier() finishes."""
        print("\n=========================")
        print("= XGBClassifier Results =")
        print("=========================\n")

        print("Classes:\n", self.classes)
        print("\n{:<20} {:<20}".format("Accuracy:", self.accuracy))

        if self.bin:
            print("\n{:<20} {:<20}".format("ROC AUC:", self.roc_auc))
        else:
            print("\nConfusion Matrix:\n", self.confusion_matrix)

        print("\nCross Validation Scores:", self.cross_val_scores_classifier)
        print("\nFeature Importances:", self.feature_importances_classifier)

        if self.gridsearch:
            print("\n{:<20} {:<20}".format("GridSearch Score:",
                                           self.gs_result))

        if self.bin and self.graph_results:
            plt.plot(self.fpr, self.tpr, label="data 1")
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc=4)
            plt.show()

        print("\n\nCall predict_classifier() to make predictions for new data.")

        print("\n===================")
        print("= End of results. =")
        print("===================\n")

    def _output_regressor_results(self):
        """Outputs model metrics after run_regressor() finishes."""
        print("\n========================")
        print("= XGBRegressor Results =")
        print("========================\n")

        print("{:<20} {:<20}".format("Mean Squared Error:",
                                     self.mean_squared_error))
        print("\n{:<20} {:<20}".format("R2 Score:", self.r2_score))
        print("\n{:<20} {:<20}".format("R Score:", str(self.r_score)))
        print("\nCross Validation Scores:", self.cross_val_scores_regressor)
        print("\nFeature Importances:", self.feature_importances_regressor)

        if self.gridsearch:
            print("\n{:<20} {:<20}".format("GridSearch Score:",
                                           self.gs_result))

        print("\n\nCall predict_regressor() to make predictions for new data.")

        print("\n===================")
        print("= End of results. =")
        print("===================\n")

    def _run_gridsearch(self, classifier):
        """Runs GridSearch with the parameters given in run_classifier()
        or run_regressor(). Returns the best parameters."""
        dataset_X_train, dataset_X_test, dataset_y_train, dataset_y_test = \
            train_test_split(self.attributes, self.labels,
                             test_size=self.test_size)
        if classifier:
            acc_scorer = make_scorer(accuracy_score)
            clf = XGBClassifier()

            # Run GridSearch
            grid_obj = GridSearchCV(clf, self.gs_params, scoring=acc_scorer)
            grid_obj = grid_obj.fit(dataset_X_train, dataset_y_train)

            # Set the clf to the best combination of parameters
            clf = grid_obj.best_estimator_

            # Fit the best algorithm to the data
            clf.fit(dataset_X_train, dataset_y_train)
            predictions = clf.predict(dataset_X_test)
            self.gs_result = accuracy_score(dataset_y_test, predictions)
        else:
            clf = XGBRegressor()

            # Run GridSearch
            grid_obj = GridSearchCV(clf, self.gs_params, scoring="r2")
            grid_obj = grid_obj.fit(dataset_X_train, dataset_y_train)

            # Set the clf to the best combination of parameters
            clf = grid_obj.best_estimator_

            # Fit the best algorithm to the data
            clf.fit(dataset_X_train, dataset_y_train)
            predictions = clf.predict(dataset_X_test)
            self.gs_result = clf.score(dataset_X_test, dataset_y_test)

        # Return the best parameters
        print("\nBest GridSearch Parameters:\n", grid_obj.best_params_, "\n")
        return grid_obj.best_params_

    def _check_inputs(self):
        """Verifies if the instance data is ready for use in XGBoost model."""
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
