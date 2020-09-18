"""RandomForest trains a random forest model implemented by
Scikit-Learn on the given dataset. Before training, the user is
prompted for parameter input. After training, model metrics are
displayed, and the user can make new predictions. Classification and
regression are both supported.

View the documentation at https://manufacturingnet.readthedocs.io/.
"""

from math import sqrt

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix, make_scorer,
                             mean_squared_error, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)


class RandomForest:
    """Class framework for random forest classification and regression
    models.
    """

    def __init__(self, attributes=None, labels=None):
        """Initializes a RandomForest object."""
        self.attributes = attributes
        self.labels = labels

        self.test_size = None
        self.cv = None
        self.graph_results = None
        self.fpr = None
        self.tpr = None
        self.bin = None
        self.gridsearch = False
        self.gs_params = None
        self.gs_result = None

        self.classifier = None
        self.accuracy = None
        self.roc_auc = None
        self.confusion_matrix = None
        self.cross_val_scores_classifier = None
        self.feature_importances_classifier = None

        self.regressor = None
        self.r2_score = None
        self.r_score = None
        self.mean_squared_error = None
        self.cross_val_scores_regressor = None
        self.feature_importances_regressor = None

    # Accessor methods

    def get_attributes(self):
        """Accessor method for attributes."""
        return self.attributes

    def get_labels(self):
        """Accessor method for labels."""
        return self.labels

    def get_classifier(self):
        """Accessor method for classifier."""
        return self.classifier

    def get_accuracy(self):
        """Accessor method for accuracy."""
        return self.accuracy

    def get_roc_auc(self):
        """Accessor method for roc_auc."""
        return self.roc_auc

    def get_confusion_matrix(self):
        """Accessor method for confusion_matrix."""
        return self.confusion_matrix

    def get_cross_val_scores_classifier(self):
        """Accessor method for cross_val_scores_classifier."""
        return self.cross_val_scores_classifier

    def get_feature_importances_classifier(self):
        """Accessor method for feature_importances_classifier."""
        return self.feature_importances_classifier

    def get_regressor(self):
        """Accessor method for regressor."""
        return self.regressor

    def get_r2_score(self):
        """Accessor method for r2_score."""
        return self.r2_score

    def get_r_score(self):
        """Accessor method for r_score."""
        return self.r_score

    def get_mean_squared_error(self):
        """Accessor method for mean_squared_error."""
        return self.mean_squared_error

    def get_cross_val_scores_regressor(self):
        """Accessor method for cross_val_scores_regressor."""
        return self.cross_val_scores_regressor

    def get_feature_importances_regressor(self):
        """Accessor method for feature_importances_regressor."""
        return self.feature_importances_regressor

    # Modifier methods

    def set_attributes(self, new_attributes=None):
        """Modifier method for attributes."""
        self.attributes = new_attributes

    def set_labels(self, new_labels=None):
        """Modifier method for labels."""
        self.labels = new_labels

    # Wrappers for RandomForest classes

    def run_classifier(self):
        """Provides random forest's classifier functionality."""
        if self._check_inputs():
            # Initialize classifier
            self.classifier = self._create_model(classifier=True)

            # Split attributes and labels into training/testing data
            dataset_X_train, dataset_X_test, dataset_y_train, dataset_y_test = \
                train_test_split(self.attributes, self.labels,
                                 test_size=self.test_size)

            # Train classifier
            # Handle exception if arguments are incorrect
            try:
                self.classifier.fit(dataset_X_train, dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the",
                      "classification model. Check your arguments and try",
                      "again.")
                print("Here is the exception message:")
                print(e)
                self.classifier = None
                return

            # Metrics
            self.accuracy = self.classifier.score(dataset_X_test,
                                                  dataset_y_test)

            y_prediction = self.classifier.predict(dataset_X_test)

            probas = self.classifier.predict_proba(dataset_X_test)

            # If classification is binary, calculate roc_auc
            if probas.shape[1] == 2:
                self.bin = True
                self.roc_auc = roc_auc_score(y_prediction, probas[::, 1])
                self.fpr, self.tpr, _ = roc_curve(
                    dataset_y_test, probas[::, 1])
            # Else, calculate confusion matrix
            else:
                self.confusion_matrix = confusion_matrix(dataset_y_test,
                                                         y_prediction)

            self.cross_val_scores_classifier = \
                cross_val_score(self.classifier, self.attributes, self.labels,
                                cv=self.cv)
            self.feature_importances_classifier = \
                self.classifier.feature_importances_

            # Output results
            self._output_classifier_results()

    def run_regressor(self):
        """Provides random forest's regressor functionality."""
        if self._check_inputs():
            # Initialize regressor
            self.regressor = self._create_model(classifier=False)

            # Split attributes and labels into training/testing data
            dataset_X_train, dataset_X_test, dataset_y_train, dataset_y_test =\
                train_test_split(self.attributes, self.labels,
                                 test_size=self.test_size)

            # Train regressor
            # Handle exception if arguments are incorrect and/or labels
            # isn't quantitative
            try:
                self.regressor.fit(dataset_X_train, dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the regressor",
                      "model. Check your arguments and try again.")
                print("Does labels contain only quantitative data?")
                print("Here is the exception message:")
                print(e)
                self.regressor = None
                return

            # Metrics
            self.r2_score = self.regressor.score(
                dataset_X_test, dataset_y_test)
            if self.r2_score >= 0:
                self.r_score = sqrt(self.r2_score)

            self.mean_squared_error = \
                mean_squared_error(dataset_y_test,
                                   self.regressor.predict(dataset_X_test))
            self.cross_val_scores_regressor = \
                cross_val_score(self.regressor, self.attributes, self.labels,
                                cv=self.cv)
            self.feature_importances_regressor = \
                self.regressor.feature_importances_

            # Output results
            self._output_regressor_results()

    def predict_classifier(self, dataset_X=None):
        """Classifies each datapoint in dataset_X using the classifier
        model. Returns the predicted classifications.
        """
        # Check that run_classifier() has already been called
        if self.classifier is None:
            print("The classifier model seems to be missing. Have you called",
                  "run_classifier() yet?")
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

        print("\nRandomForestClassifier Predictions:\n", y_prediction, "\n")
        return y_prediction

    def predict_regressor(self, dataset_X=None):
        """Predicts the output of each datapoint in dataset_X using the
        regressor model. Returns the predictions.
        """
        # Check that run_regressor() has already been called
        if self.regressor is None:
            print("The regressor model seems to be missing. Have you called",
                  "run_regressor() yet?")
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

        print("\nRandomForestRegressor Predictions:\n", y_prediction, "\n")
        return y_prediction

    # Helper methods

    def _create_model(self, classifier):
        """Runs UI for getting parameters and creating classifier or
        regression model.
        """
        if classifier:
            print("\n===========================================")
            print("= RandomForestClassifier Parameter Inputs =")
            print("===========================================\n")
        else:
            print("\n==========================================")
            print("= RandomForestRegressor Parameter Inputs =")
            print("==========================================\n")

        print("Default values:", "test_size = 0.25", "cv = 5", sep="\n")
        if classifier:
            print("graph_results = False", "criterion = 'gini'",
                  "class_weight = None", sep="\n")
        else:
            print("criterion = 'mse'")

        print("n_estimators = 100",
              "max_depth = None",
              "min_samples_split = 2",
              "min_samples_leaf = 1",
              "min_weight_fraction_leaf = 0.0",
              "max_features = 'auto'",
              "max_leaf_nodes = None",
              "min_impurity_decrease = 0.0",
              "bootstrap = True",
              "oob_score = False",
              "n_jobs = None",
              "random_state = None",
              "verbose = 0",
              "warm_start = False",
              "ccp_alpha = 0.0",
              "max_samples = None", sep="\n")

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
                    return RandomForestClassifier()
                return RandomForestRegressor()
            elif user_input == "n":
                break
            else:
                print("Invalid input.")

        print("\nIf you are unsure about a parameter, press enter to use its",
              "default value.")
        print("If you finish entering parameters early, enter 'q' to skip",
              "ahead.\n")

        # Set more defaults
        if classifier:
            criterion = "gini"
            class_weight = None
        else:
            criterion = "mse"

        n_estimators = 100
        max_depth = None
        min_samples_split = 2
        min_samples_leaf = 1
        min_weight_fraction_leaf = 0.0
        max_features = "auto"
        max_leaf_nodes = None
        min_impurity_decrease = 0.0
        bootstrap = True
        oob_score = False
        n_jobs = None
        random_state = None
        verbose = 0
        warm_start = False
        ccp_alpha = 0.0
        max_samples = None

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
                    print("\nEnter the max_features for the best split.")
                    print("Options: 1-auto, 2-sqrt, 3-log2. Enter 'all' for all",
                          "options.")
                    print("Example input: 1,2,3")
                    user_input = input().lower()

                    if user_input == "q":
                        self.gridsearch = False
                        break_early = True
                        break
                    elif user_input == "all":
                        feat_params = ["auto", "sqrt", "log2"]
                        break
                    else:
                        feat_dict = {1: "auto", 2: "sqrt", 3: "log2"}
                        try:
                            feat_params_int = \
                                list(map(int, list(user_input.split(","))))
                            if len(feat_params_int) > len(feat_dict):
                                raise Exception

                            feat_params = []
                            for each in feat_params_int:
                                if not feat_dict.get(each):
                                    raise Exception

                                feat_params.append(feat_dict.get(each))
                            break
                        except Exception:
                            print("Invalid input.")

                if break_early:
                    break

                params["max_features"] = feat_params
                print("max_features:", feat_params)

                while True:
                    print("\nEnter the list of num_estimators to try out.")
                    print("Example input: 1,2,3")
                    user_input = input().lower()

                    if user_input == "q":
                        self.gridsearch = False
                        break_early = True
                        break

                    try:
                        n_est_params = \
                            list(map(int, list(user_input.split(","))))
                        if len(n_est_params) == 0:
                            raise Exception

                        for num in n_est_params:
                            if num <= 0:
                                raise Exception
                        break
                    except Exception:
                        print("Invalid input.")

                if break_early:
                    break

                params["n_estimators"] = n_est_params
                print("n_estimators:", n_est_params)

                while True:
                    print("\nEnter the criterion to be tried for.")

                    if classifier:
                        print("Options: 1-'gini', 2-'entropy'. Enter 'all' for",
                              "all options.")
                        user_input = input().lower()

                        if user_input == "q":
                            self.gridsearch = False
                            break_early = True
                            break
                        elif user_input == "all":
                            crit_params = ["gini", "entropy"]
                            break
                        else:
                            crit_dict = {1: "gini", 2: "entropy"}
                            try:
                                crit_params_int = \
                                    list(map(int, list(user_input.split(","))))
                                if len(crit_params_int) > len(crit_dict):
                                    raise Exception

                                crit_params = []
                                for each in crit_params_int:
                                    if not crit_dict.get(each):
                                        raise Exception

                                    crit_params.append(crit_dict.get(each))
                                break
                            except Exception:
                                print("Invalid input.")
                    else:
                        print("Options: 1-'mse', 2-'mae'. Enter 'all' for all",
                              "options.")
                        user_input = input().lower()

                        if user_input == "q":
                            self.gridsearch = False
                            break_early = True
                            break
                        elif user_input == "all":
                            crit_params = ["mse", "mae"]
                            break
                        else:
                            crit_dict = {1: "mse", 2: "mae"}
                            try:
                                crit_params_int = \
                                    list(map(int, list(user_input.split(","))))
                                if len(crit_params_int) > len(crit_dict):
                                    raise Exception

                                crit_params = []
                                for each in crit_params_int:
                                    if not crit_dict.get(each):
                                        raise Exception

                                    crit_params.append(crit_dict.get(each))
                                break
                            except Exception:
                                print("Invalid input.")

                if break_early:
                    break

                params["criterion"] = crit_params
                print("criterion:", crit_params)

                while True:
                    print("\nEnter the maximum depth of trees to try for.")
                    print("Example input: 1,2,3")
                    user_input = input().lower()

                    if user_input == "q":
                        self.gridsearch = False
                        break_early = True
                        break

                    try:
                        max_dep_params = \
                            list(map(int, list(user_input.split(","))))
                        if len(max_dep_params) == 0:
                            raise Exception

                        for num in max_dep_params:
                            if num <= 0:
                                raise Exception
                        break
                    except Exception:
                        print("Invalid input.")

                if break_early:
                    break

                params["max_depth"] = max_dep_params
                print("max_depths:", max_dep_params)

                print("\n= End of GridSearch inputs. =\n")
                self.gs_params = params
                best_params = self._run_gridsearch(classifier)
                criterion = best_params["criterion"]
                max_depth = best_params["max_depth"]
                max_features = best_params["max_features"]
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
                user_input = \
                    input("\nEnter a positive number of trees for the forest: ")
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
                print("\nWhich criteria should be used for measuring split",
                      "quality?")
                if classifier:
                    user_input = input(
                        "Enter 1 for 'gini' or 2 for 'entropy': ")
                    if user_input == "2":
                        criterion = "entropy"
                        break
                    elif user_input in {"1", ""}:
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break
                    else:
                        print("Invalid input.")
                else:
                    user_input = input("Enter 1 for 'mse' or 2 for 'mae': ")
                    if user_input == "2":
                        criterion = "mae"
                        break
                    elif user_input in {"1", ""}:
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break
                    else:
                        print("Invalid input.")

            if not self.gridsearch:
                print("criterion =", criterion)

            if break_early:
                break

            while classifier:
                user_input = input("\nAutomatically balance the class weights "
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

            if classifier:
                print("class_weight =", class_weight)

            if break_early:
                break

            while not self.gridsearch:
                print("\nEnter a positive maximum tree depth.")
                user_input = input("Press enter for no maximum depth: ")
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

            while True:
                user_input = \
                    input("\nEnter min_samples_split, a positive minimum number "
                          + "of samples required to split an internal node: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    if int(user_input) < 1:
                        if float(user_input) <= 0:
                            raise Exception

                        min_samples_split = float(user_input)
                    else:
                        if int(user_input) <= 0:
                            raise Exception

                        min_samples_split = int(user_input)
                    break
                except Exception:
                    print("Invalid input.")

            print("min_samples_split =", min_samples_split)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nEnter min_samples_leaf, a positive minimum number "
                          + "of samples required to be at a leaf node: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    if int(user_input) < 1:
                        if float(user_input) <= 0:
                            raise Exception

                        min_samples_leaf = float(user_input)
                    else:
                        if int(user_input) <= 0:
                            raise Exception

                        min_samples_leaf = int(user_input)
                    break
                except Exception:
                    print("Invalid input.")

            print("min_samples_leaf =", min_samples_leaf)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nEnter min_weight_fraction_leaf, the minimum "
                          + "weighted fraction of the weight total required to "
                          + "be at a leaf node [0,0.5]: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input < 0 or user_input > 0.5:
                        raise Exception

                    min_weight_fraction_leaf = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("min_weight_fraction_leaf =", min_weight_fraction_leaf)

            if break_early:
                break

            while not self.gridsearch:
                print("\nHow many features should be considered when looking",
                      "for the best split?")
                print("Enter 'auto' to use n_features, 'sqrt' to use",
                      "sqrt(n_features), 'log2' to use log2(n_features) or a",
                      "positive number/fraction: ")
                user_input = input().lower()

                try:
                    if user_input in {"sqrt", "log2", "auto"}:
                        max_features = user_input
                        break
                    elif user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    if int(user_input) == 0:
                        user_input = float(user_input)
                    else:
                        user_input = int(user_input)

                    if user_input <= 0:
                        raise Exception

                    max_features = user_input
                    break
                except Exception:
                    print("Invalid input.")

            if not self.gridsearch:
                print("max_features =", max_features)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nEnter a positive maximum number of leaf nodes: ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = int(user_input)
                    if user_input <= 0:
                        raise Exception

                    max_leaf_nodes = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("max_leaf_nodes =", max_leaf_nodes)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nEnter minimum_impurity_decrease [0,): ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input < 0:
                        raise Exception

                    min_impurity_decrease = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("min_impurity_decrease =", min_impurity_decrease)

            if break_early:
                break

            while True:
                user_input = input("\nUse bootstrap samples when building "
                                   + "trees (Y/n)? ").lower()
                if user_input == "n":
                    bootstrap = False
                    break
                elif user_input in {"y", ""}:
                    break
                elif user_input == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            print("bootstrap =", bootstrap)

            if break_early:
                break

            while True:
                user_input = input("\nUse out-of-bag samples to estimate R2 "
                                   + "scores on unseen data (y/N)? ").lower()
                if user_input == "y":
                    oob_score = True
                    break
                elif user_input in {"n", ""}:
                    break
                elif user_input == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            print("oob_score =", oob_score)

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
                    verbose = 1
                    break
                elif user_input in {"n", ""}:
                    break
                elif user_input == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            print("verbose =", bool(verbose))

            if break_early:
                break

            while True:
                user_input = \
                    input("\nEnable warm start? This will use the previous "
                          + "solution for fitting (y/N): ").lower()
                if user_input == "y":
                    warm_start = True
                    break
                elif user_input in {"n", ""}:
                    break
                elif user_input == "q":
                    break_early = True
                    break
                else:
                    print("Invalid input.")

            print("warm_start =", warm_start)

            if break_early:
                break

            while True:
                user_input = \
                    input("\nEnter ccp_alpha, the complexity parameter for "
                          + "Minimal Cost-Complexity Pruning [0,): ")
                try:
                    if user_input == "":
                        break
                    elif user_input.lower() == "q":
                        break_early = True
                        break

                    user_input = float(user_input)
                    if user_input < 0:
                        raise Exception

                    ccp_alpha = user_input
                    break
                except Exception:
                    print("Invalid input.")

            print("ccp_alpha =", ccp_alpha)

            if break_early:
                break

            while bootstrap:
                user_input = \
                    input("\nEnter a positive number/fraction for the maximum "
                          + "number of samples to train the base estimators: ")
                try:
                    if user_input.lower() in {"q", ""}:
                        break

                    if int(user_input) == 0:
                        user_input = float(user_input)
                    else:
                        user_input = int(user_input)

                    if user_input <= 0:
                        raise Exception

                    max_samples = user_input
                    break
                except Exception:
                    print("Invalid input.")

            if bootstrap:
                print("max_samples =", max_samples)

            break

        print("\n===========================================")
        print("= End of inputs; press enter to continue. =")
        input("===========================================\n")

        if classifier:
            return RandomForestClassifier(n_estimators=n_estimators,
                                          criterion=criterion,
                                          max_depth=max_depth,
                                          min_samples_split=min_samples_split,
                                          min_samples_leaf=min_samples_leaf,
                                          min_weight_fraction_leaf=min_weight_fraction_leaf,
                                          max_features=max_features,
                                          max_leaf_nodes=max_leaf_nodes,
                                          min_impurity_decrease=min_impurity_decrease,
                                          bootstrap=bootstrap, oob_score=oob_score,
                                          n_jobs=n_jobs, random_state=random_state,
                                          verbose=verbose, warm_start=warm_start,
                                          class_weight=class_weight,
                                          ccp_alpha=ccp_alpha,
                                          max_samples=max_samples)

        return RandomForestRegressor(n_estimators=n_estimators,
                                     criterion=criterion, max_depth=max_depth,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     min_weight_fraction_leaf=min_weight_fraction_leaf,
                                     max_features=max_features,
                                     max_leaf_nodes=max_leaf_nodes,
                                     min_impurity_decrease=min_impurity_decrease,
                                     bootstrap=bootstrap, oob_score=oob_score,
                                     n_jobs=n_jobs, random_state=random_state,
                                     verbose=verbose, warm_start=warm_start,
                                     ccp_alpha=ccp_alpha,
                                     max_samples=max_samples)

    def _output_classifier_results(self):
        """Outputs model metrics after run_classifier() finishes."""
        print("\n==================================")
        print("= RandomForestClassifier Results =")
        print("==================================\n")

        print("Classes:\n", self.classifier.classes_)
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
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc=4)
            plt.show()

        print("\n\nCall predict_classifier() to make predictions for new data.")

        print("\n===================")
        print("= End of results. =")
        print("===================\n")

    def _output_regressor_results(self):
        """Outputs model metrics after run_regressor() finishes."""
        print("\n=================================")
        print("= RandomForestRegressor Results =")
        print("=================================\n")

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
            clf = RandomForestClassifier()

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
            clf = RandomForestRegressor()

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
        """Verifies if instance data is ready for use in RandomForest
        models.
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
                  "new_labels should be a populated dataset of classes (for",
                  "classification) or dependent variables (for regression).")
            return False

        # Check if attributes and labels have same number of rows (samples)
        if self.attributes.shape[0] != self.labels.shape[0]:
            print("attributes and labels don't have the same number of rows.",
                  "Make sure the number of samples in each dataset matches!")
            return False

        return True
