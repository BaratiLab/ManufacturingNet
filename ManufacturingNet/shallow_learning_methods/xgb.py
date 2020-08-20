"""XGBoost trains a XGBoost model on the given dataset. Before
training, the user is prompted for parameter input. After training,
model metrics are displayed, and the user can make new predictions.
Classification and regression are both supported.

View the documentation at https://manufacturingnet.readthedocs.io/.
"""

from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, \
    mean_squared_error, roc_auc_score, roc_curve, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split, \
    GridSearchCV
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

        self.classifier = None
        self.accuracy = None
        self.confusion_matrix = None
        self.roc_auc = None
        self.classes = None
        self.cross_val_scores_classifier = None

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

    def get_cross_val_scores_regressor(self):
        """Accessor method for cross_val_scores_regressor."""
        return self.cross_val_scores_regressor

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
            self.r2_score = self.regressor.score(dataset_X_test, dataset_y_test)
            self.r_score = sqrt(self.r2_score)
            self.cross_val_scores_regressor = \
                cross_val_score(self.regressor, self.attributes, self.labels,
                                cv=self.cv)

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
                self.fpr, self.tpr, _ = roc_curve(dataset_y_test, probas[::, 1])
            # Else, calculate confusion_matrix
            else:
                self.confusion_matrix = \
                    confusion_matrix(dataset_y_test, y_prediction)

            self.cross_val_scores_classifier = \
                cross_val_score(self.classifier, self.attributes, self.labels,
                                cv=self.cv)

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

        if input("Use default parameters (Y/n)? ").lower() != "n":
            self.test_size = 0.25
            self.cv = None
            print("\n=============================================")
            print("= End of inputs; press any key to continue. =")
            input("=============================================\n")

            if classifier:
                return XGBClassifier()

            return XGBRegressor()

        print("\nIf you are unsure about a parameter, press enter to use its",
              "default value.")
        print("Invalid parameter inputs will be replaced with their default",
              "values.")
        print("If you finish entering parameters early, enter 'q' to skip",
              "ahead.\n")

        # Set defaults; same parameters for classification and regression
        self.test_size = 0.25
        self.cv = None
        self.graph_results = False
        n_estimators = 100
        max_depth = 3
        learning_rate = 0.1

        if classifier:
            objective = "binary:logistic"
        else:
            objective = "reg:squarederror"

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

                print("\nEnter the types of boosters.")
                print("Options: 1-'gbtree', 2-'gblinear' or 3-'dart'. Enter",
                      "'all' for all options.")
                print("Example input: 1,2,3")
                user_input = input().lower()

                if user_input == "q":
                    self.gridsearch = False
                    break
                elif user_input == "all":
                    boost_params = ["gbtree", "gblinear", "dart"]
                else:
                    boost_dict = {1: "gbtree", 2: "gblinear", 3: "dart"}

                    try:
                        boost_params_int = \
                            list(map(int, list(user_input.split(","))))
                        boost_params = []
                        for each in boost_params_int:
                            boost_params.append(boost_dict.get(each))
                    except Exception:
                        print("\nInput not recognized. Skipping GridSearch...")
                        self.gridsearch = False
                        break

                params["booster"] = boost_params

                print("\nEnter a list of learning rates to try out.")
                print("Example input: 0.1,0.01,0.001")
                user_input = input().lower()

                if user_input == "q":
                    self.gridsearch = False
                    break

                try:
                    lr_params = list(map(float, list(user_input.split(","))))
                except Exception:
                    print("\nInput not recognized. Skipping GridSearch...")
                    self.gridsearch = False
                    break

                params["learning_rate"] = lr_params

                print("\nEnter a list of gamma values/minimum loss reductions",
                      "to try out.")
                print("Example input: 0.5,1,1.5")
                user_input = input().lower()

                if user_input == "q":
                    self.gridsearch = False
                    break

                try:
                    gamma_params = \
                        list(map(float, list(user_input.split(","))))
                except Exception:
                    print("\nInput not recognized. Skipping GridSearch...")
                    self.gridsearch = False
                    break

                params["gamma"] = gamma_params

                print("\nEnter a list of number of trees to try out.")
                print("Example input: 1,2,3")
                user_input = input().lower()

                if user_input == "q":
                    self.gridsearch = False
                    break

                try:
                    ntrees_params = list(map(int, list(user_input.split(","))))
                except Exception:
                    print("\nInput not recognized. Skipping GridSearch...")
                    self.gridsearch = False
                    break

                params["n_estimators"] = ntrees_params

                print("\nEnter a list of max tree depths to try out.")
                print("Example input: 1,2,3")
                user_input = input().lower()

                if user_input == "q":
                    self.gridsearch = False
                    break

                try:
                    mdepth_params = list(map(int, list(user_input.split(","))))
                except Exception:
                    print("\nInput not recognized. Skipping GridSearch...")
                    self.gridsearch = False
                    break

                params["max_depth"] = mdepth_params

                self.gs_params = params
                print("\n= End of GridSearch inputs. =")
                break

            user_input = \
                input("\n\nInput the number of folds for cross validation: ")

            try:
                self.cv = int(user_input)
            except Exception:
                if user_input.lower() == "q":
                    break

            if classifier:
                user_input = input("\nGraph the ROC curve (y/N)? ").lower()

                if user_input == "y":
                    self.graph_results = True
                elif user_input == "q":
                    break

            user_input = \
                input("\nEnter the number of gradient-boosted trees to use: ")

            try:
                n_estimators = int(user_input)
            except Exception:
                if user_input.lower() == "q":
                    break

            user_input = input("\nEnter the maximum depth of each tree: ")

            try:
                max_depth = int(user_input)
            except Exception:
                if user_input.lower() == "q":
                    break

            user_input = input("\nEnter the learning rate as a decimal: ")

            try:
                learning_rate = float(user_input)
            except Exception:
                if user_input.lower() == "q":
                    break

            print("\nWhich booster should be used?")
            user_input = input("Enter 1 for 'gbtree', 2 for 'gblinear', or "
                               + "3 for 'dart': ").lower()

            if user_input == "q":
                break
            elif user_input == "2":
                booster = "gblinear"
            elif user_input == "3":
                booster = "dart"

            user_input = \
                input("\nEnter the number of parallel threads to use: ")

            try:
                n_jobs = int(user_input)
            except Exception:
                if user_input.lower() == "q":
                    break

            user_input = input("\nEnter gamma, the minimum loss reduction "
                               + "needed to further partition a leaf node: ")

            try:
                gamma = int(user_input)
            except Exception:
                if user_input.lower() == "q":
                    break

            user_input = input("\nEnter the minimum child weight: ")

            try:
                min_child_weight = int(user_input)
            except Exception:
                if user_input.lower() == "q":
                    break

            user_input = input("\nEnter the maximum delta step: ")

            try:
                max_delta_step = int(user_input)
            except Exception:
                if user_input.lower() == "q":
                    break

            user_input = input("\nEnter the subsample ratio: ")

            try:
                subsample = float(user_input)
            except Exception:
                if user_input.lower() == "q":
                    break

            user_input = \
                input("\nEnter the subsample column ratio for all trees: ")

            try:
                colsample_bytree = float(user_input)
            except Exception:
                if user_input.lower() == "q":
                    break

            user_input = \
                input("\nEnter the subsample column ratio for all levels: ")

            try:
                colsample_bylevel = float(user_input)
            except Exception:
                if user_input.lower() == "q":
                    break

            user_input = input("\nEnter alpha, the L1 regularization term: ")

            try:
                reg_alpha = float(user_input)
            except Exception:
                if user_input.lower() == "q":
                    break

            user_input = input("\nEnter lambda, the L2 regularization term: ")

            try:
                reg_lambda = float(user_input)
            except Exception:
                if user_input.lower() == "q":
                    break

            user_input = \
                input("\nEnter scale_pos_weight to control class balancing: ")

            try:
                scale_pos_weight = float(user_input)
            except Exception:
                if user_input.lower() == "q":
                    break

            user_input = input("\nEnter the initial prediction score: ")

            try:
                base_score = float(user_input)
            except Exception:
                if user_input.lower() == "q":
                    break

            user_input = \
                input("\nEnter a seed for the random number generator: ")

            try:
                random_state = int(user_input)
            except Exception:
                if user_input.lower() == "q":
                    break

            user_input = input("\nEnable verbose logging (y/N)? ").lower()

            if user_input == "y":
                verbosity = 1

            break

        print("\n=============================================")
        print("= End of inputs; press any key to continue. =")
        input("=============================================\n")

        if classifier:
            # If GridSearch is enabled
            if self.gridsearch:
                acc_scorer = make_scorer(accuracy_score)
                clf = XGBClassifier()
                dataset_X_train, dataset_X_test, dataset_y_train, \
                    dataset_y_test = train_test_split(self.attributes,
                                                      self.labels,
                                                      test_size=self.test_size)

                # Run GridSearch
                grid_obj = GridSearchCV(clf, self.gs_params, scoring=acc_scorer)
                grid_obj = grid_obj.fit(dataset_X_train, dataset_y_train)

                # Set the clf to the best combination of parameters
                clf = grid_obj.best_estimator_

                print("Best GridSearch Parameters:\n", clf.get_xgb_params())

                # Fit the best algorithm to the data
                clf.fit(dataset_X_train, dataset_y_train)
                predictions = clf.predict(dataset_X_test)
                self.gs_result = accuracy_score(dataset_y_test, predictions)

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

        # If GridSearch is enabled
        if self.gridsearch:
            clf = XGBRegressor()
            dataset_X_train, dataset_X_test, dataset_y_train, dataset_y_test = \
                train_test_split(self.attributes, self.labels,
                                 test_size=self.test_size)

            # Run GridSearch
            grid_obj = GridSearchCV(clf, self.gs_params, scoring="r2")
            grid_obj = grid_obj.fit(dataset_X_train, dataset_y_train)

            # Set the clf to the best combination of parameters
            clf = grid_obj.best_estimator_

            print("Best GridSearch Parameters:\n", grid_obj.best_params_)

            # Fit the best algorithm to the data
            clf.fit(dataset_X_train, dataset_y_train)
            predictions = clf.predict(dataset_X_test)
            self.gs_result = clf.score(dataset_X_test, dataset_y_test)

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
        print("\n{:<20} {:<20}".format("R Score:", self.r_score))
        print("\nCross Validation Scores:", self.cross_val_scores_regressor)

        if self.gridsearch:
            print("\n{:<20} {:<20}".format("GridSearch Score:",
                                           self.gs_result))

        print("\n\nCall predict_regressor() to make predictions for new data.")

        print("\n===================")
        print("= End of results. =")
        print("===================\n")

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
