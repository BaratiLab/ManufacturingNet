from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from math import sqrt

import numpy as np

import xgboost as xgb



class XGBoost: 
    
    """
    XGBoost Documentation

    """


    def __init__(self, attributes=None, labels=None, test_size=0.25):
        

        """
        Initializes an XGBoost object.

        The following parameters are needed to create a logistic regression model:

            - attributes: a numpy array of the desired independent variables
            - labels: a numpy array of the desired dependent variables
            - test_size: the proportion of the dataset to be used for testing the model (defaults to 0.25);
            the proportion of the dataset to be used for training will be the complement of test_size

        """

        self.attributes = attributes
        self.labels = labels
        self.test_size = test_size

        self.xgb_model = None

        self.precision_scores = []
        self.precision = None
        self.recall_scores = []
        self.recall = None

        self.mean_squared_error = None
        self.r2_score = None
        self.r_score = None
        

        self.accuracy = None
        self.confusion_matrix = None
        self.roc_auc = None

        self.classes_ = None

        self.dataset_X_train = None
        self.dataset_X_test = None
        self.dataset_y_train = None
        self.dataset_y_test = None


    # Accessor methods

    def get_attributes(self):
        """
        Accessor method for attributes.

        If a MLP object is constructed without specifying attributes, attributes will be None.
        logistic_regression() cannot be called until attributes is a populated numpy array; call
        set_attributes(new_attributes) to fix this.
        """
        return self.attributes

    def get_labels(self):
        """
        Accessor method for labels.

        If a MLP object is constructed without specifying labels, labels will be None.
        logistic_regression() cannot be called until labels is a populated numpy array; call set_labels(new_labels)
        to fix this.
        """
        return self.labels


    
    def get_classes(self):
        """
        Accessor method for classes.

        Will return None if linear_regression() hasn't been called, yet.
        """
        return self.classes_


    def get_accuracy(self):
        """
        Accessor method for accuracy.

        Will return None if linear_regression() hasn't been called, yet.
        """
        return self.accuracy

    def get_mean_squared_error(self):
        """
        Accessor method for mean squared error of linear regression model.
        Will return None if linear_regression() hasn't been called, yet.
        """
        return self.mean_squared_error

    def get_r2_score(self):
        """
        Accessor method for coefficient of determination of linear regression model.
        Will return None if linear_regression() hasn't been called, yet.
        """
        return self.r2_score

    def get_r_score(self):
        """
        Accessor method for correlation coefficient of linear regression model.
        Will return None if linear_regression() hasn't been called, yet.
        """
        return self.r_score

    def get_confusion_matrix(self):
        """
        Accessor method for confusion matrix of the XGBoost classification model.
        Will return None if XGBClassifier() hasn't been called, yet.

        """
        return self.confusion_matrix

    def get_roc_auc(self):
        """
        Accessor method for roc-auc score.

        Will return None if XGBClassifier() hasn't been called, yet.
        """

        return self.roc_auc



    def get_precision(self, label):
        """
        Accessor method for precision.

        Will return None if linear_regression() hasn't been called, yet.
        """
        self.precision = self.precision_scores.get(label)
        return self.precision

    
    def get_recall(self, label):
        """
        Accessor method for precision.

        Will return None if linear_regression() hasn't been called, yet.
        """
        self.recall = self.recall_scores.get(label)
        return self.recall


    #Modifier methods

    def set_attributes(self, new_attributes = None):
        """
        Modifier method for attributes.

        Input should be a populated numpy array of the desired independent variables.
        """
        self.attributes = new_attributes

    def set_labels(self, new_labels = None):
        """
        Modifier method for labels.

        Input should be a populated numpy array of the desired dependent variables.
        """
        self.labels = new_labels

    def set_test_size(self, new_test_size = None):
        """
        Modifier method for train_test_split.

        Input should be a list of strings, where each string is the name of a dependent variable.
        """
        self.test_size  = new_test_size



    def XGBRegressor(self, base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.1, \
            max_delta_step=0, max_depth=3, min_child_weight=1, missing=None, n_estimators=100, n_jobs=1, nthread=None, objective='reg:squarederror', \
                random_state=42, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None, subsample=1):


        if self._check_inputs():

            self.xgb_model = xgb.XGBRegressor(base_score=base_score, booster=booster, colsample_bylevel=colsample_bylevel, colsample_bytree=colsample_bytree, gamma=gamma, learning_rate=learning_rate, \
                max_delta_step=max_delta_step, max_depth=max_depth, min_child_weight=min_child_weight, missing=missing, n_estimators=n_estimators, n_jobs=n_jobs, nthread=nthread, objective=objective, \
                    random_state=random_state, reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight, seed=seed, subsample=subsample)

            if self.dataset_X_test is None:
                self._split_data()

            # Train the model and get resultant coefficients; handle exception if arguments aren't correct
            try:
                self.xgb_model.fit(self.dataset_X_train, self.dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the model. Check your inputs and try again.")
                print("Here is the exception message:")
                print(e)
                self.regression = None
                return


            # Make predictions using testing set
            y_prediction = self.xgb_model.predict(self.dataset_X_test)

            # Get mean squared error, coefficient of determination, and correlation coefficient
            self.mean_squared_error = mean_squared_error(self.dataset_y_test, y_prediction)
            self.r2_score = r2_score(self.dataset_y_test, y_prediction)
            self.r_score = sqrt(self.r2_score)



    def XGBClassifier(self, max_depth=3, learning_rate=0.1, n_estimators=100, objective='binary:logistic', booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None):

        if self._check_inputs():

            self.xgb_model = xgb.XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, \
                objective=objective, booster=booster, n_jobs=n_jobs, nthread=nthread, gamma=gamma, min_child_weight=min_child_weight, \
                    max_delta_step=max_delta_step, subsample=subsample, colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel, reg_alpha=reg_alpha, \
                        reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight, base_score=base_score, random_state=random_state, seed=seed, missing=missing)


            if self.dataset_X_test is None:
                self._split_data()

            # Train the model and get resultant coefficients; handle exception if arguments aren't correct
            try:
                self.xgb_model.fit(self.dataset_X_train, np.ravel(self.dataset_y_train))
            except Exception as e:
                print("An exception occurred while training the model. Check your inputs and try again.")
                print("Here is the exception message:")
                print(e)
                self.regression = None
                return


            # Make predictions using testing set
            y_prediction = self.xgb_model.predict(self.dataset_X_test)
            y_pred_probas = self.xgb_model.predict_proba(self.dataset_X_test)[::, 1]

            self.classes_ = self.xgb_model.classes_

            self.accuracy = accuracy_score(self.dataset_y_test, y_prediction)
            self.confusion_matrix = confusion_matrix(self.dataset_y_test, y_prediction)
            self.roc_auc = roc_auc_score(y_prediction, y_pred_probas)

            self.precision_scores = { each : precision_score(self.dataset_y_test, y_prediction, pos_label=each) \
                                                                    for each in self.classes_}
            self.recall_scores = { each : recall_score(self.dataset_y_test, y_prediction, pos_label=each) \
                                                                    for each in self.classes_}



    # Helper methods

    def _split_data(self):
        """
        Helper method for splitting attributes and labels into training and testing sets.
        This method runs under the assumption that all relevant instance data has been checked for correctness.
        """

        self.dataset_X_train, self.dataset_X_test, self.dataset_y_train, self.dataset_y_test =\
            train_test_split(self.attributes, self.labels, test_size=self.test_size)


    # Helper method for checking inputs

    def _check_inputs(self):
        """
        Verifies if the instance data is ready for use in logistic regression model.
        """

        # Check if attributes exists
        if self.attributes is None or type(self.attributes) is not np.ndarray:
            print("attributes is missing; call set_attributes(new_attributes) to fix this! new_attributes should be a",
            "populated numpy array of your independent variables.")
            return False

        # Check if labels exists
        if self.labels is None or type(self.labels) is not np.ndarray:
            print("labels is missing; call set_labels(new_labels) to fix this! new_labels should be a populated numpy",
            "array of your dependent variables.")
            return False

        # Check if attributes and labels have same number of rows (samples)
        if self.attributes.shape[0] != self.labels.shape[0]:
            print("attributes and labels don't have the same number of rows. Make sure the number of samples in each",
                  "dataset matches!")
            return False

        # Check if test_size is a float or None
        if self.test_size is not None and not isinstance(self.test_size, (int, float)):
            print("test_size must be None or a number; call set_test_size(new_test_size) to fix this!")
            return False

        return True


