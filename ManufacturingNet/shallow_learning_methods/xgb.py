from math import sqrt
from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
import numpy as np

class XGBoost: 
    """
    Wrapper class around XGBoost's classification and regression functionality. Per XGBoost's documentation:

    XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.
    It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree
    boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way.
    """

    def __init__(self, attributes=None, labels=None, test_size=0.25):
        """
        Initializes an XGBoost object.

        The following parameters are needed to create a XGBoost model:

            - attributes: a numpy array of the desired independent variables
            - labels: a numpy array of the desired dependent variables
            - test_size: the proportion of the dataset to be used for testing the model (defaults to 0.25);
            the proportion of the dataset to be used for training will be the complement of test_size
        
        After successfully running run_regressor(), the following instance data will be available:

            - regressor: a reference to the XGBRegressor model
            - mean_squared_error: the mean squared error for the regression model
            - r2_score: the coefficient of determination for the regression model
            - r_score: the correlation coefficient for the regression model
        
        After successfully running run_classifier(), the following instance data will be available:

            - classifier: a reference to the XGBClassifier model
            - precision_scores: a list of the precision for each label
            - recall_scores: a list of the recall scores for each label
            - accuracy: the mean accuracy of the classification model
            - confusion_matrix: a 2x2 matrix of true negatives, false negatives, true positives, and false positives
            of the model
            - roc_auc: the area under the ROC curve for the classification model
            - classes_: the classes for each label
        """
        self.attributes = attributes
        self.labels = labels
        self.test_size = test_size

        self.regressor = None
        self.mean_squared_error = None
        self.r2_score = None
        self.r_score = None

        self.classifier = None
        self.precision_scores = []
        self.recall_scores = []
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

        If a XGBoost object is constructed without specifying attributes, attributes will be None.
        run_regressor() cannot be called until attributes is a populated numpy array; call
        set_attributes(new_attributes) to fix this.
        """
        return self.attributes

    def get_labels(self):
        """
        Accessor method for labels.

        If a XGBoost object is constructed without specifying labels, labels will be None.
        run_regressor() cannot be called until labels is a populated numpy array; call set_labels(new_labels)
        to fix this.
        """
        return self.labels

    def get_regressor(self):
        """
        Accessor method for regressor.

        Will return None if run_regressor() hasn't been called, yet.
        """
        return self.regressor

    def get_classifier(self):
        """
        Accessor method for classifier.

        Will return None if run_classifier() hasn't been called, yet.
        """
        return self.classifier

    def get_classes(self):
        """
        Accessor method for classes of the XGBoost classification model.

        Will return None if run_classifier() hasn't been called, yet.
        """
        return self.classes_

    def get_accuracy(self):
        """
        Accessor method for accuracy of the XGBoost classification model.

        Will return None if run_classifier() hasn't been called, yet.
        """
        return self.accuracy

    def get_mean_squared_error(self):
        """
        Accessor method for mean squared error of the XGBoost regression model.

        Will return None if run_regressor() hasn't been called, yet.
        """
        return self.mean_squared_error

    def get_r2_score(self):
        """
        Accessor method for coefficient of determination of the XGBoost regression model.

        Will return None if run_regressor() hasn't been called, yet.
        """
        return self.r2_score

    def get_r_score(self):
        """
        Accessor method for correlation coefficient of the XGBoost regression model.
        
        Will return None if run_regressor() hasn't been called, yet.
        """
        return self.r_score

    def get_confusion_matrix(self):
        """
        Accessor method for confusion matrix of the XGBoost classification model.
        
        Will return None if run_classifier() hasn't been called, yet.
        """
        return self.confusion_matrix

    def get_roc_auc(self):
        """
        Accessor method for roc-auc score.

        Will return None if run_classifier() hasn't been called, yet.
        """
        return self.roc_auc

    def get_precision(self, label=None):
        """
        Accessor method for precision of the given label.

        Will return None if run_classifier() hasn't been called, yet.
        """
        return self.precision_scores.get(label)
    
    def get_precision_scores(self):
        """
        Accessor method for precision_scores.

        Will return None if run_classifier() hasn't been called, yet.
        """
        return self.precision_scores

    def get_recall(self, label=None):
        """
        Accessor method for recall score of the given label.

        Will return None if run_classifier() hasn't been called, yet.
        """
        return self.recall_scores.get(label)

    def get_recall_scores(self):
        """
        Accessor method for recall_scores.

        Will return None if run_classifier() hasn't been called, yet.
        """
        return self.recall_scores

    # Modifier methods

    def set_attributes(self, new_attributes=None):
        """
        Modifier method for attributes.

        Input should be a populated numpy array of the desired independent variables.
        """
        self.attributes = new_attributes

    def set_labels(self, new_labels=None):
        """
        Modifier method for labels.

        Input should be a populated numpy array of the desired dependent variables.
        """
        self.labels = new_labels

    def set_test_size(self, new_test_size=None):
        """
        Modifier method for test_size.

        Input should be a number or None.
        """
        self.test_size = new_test_size
    
    # Wrapper for regression functionality

    def run_regressor(self, base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
                      max_delta_step=0, max_depth=3, min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
                      nthread=None, objective='reg:squarederror', random_state=42, reg_alpha=0, reg_lambda=1,
                      scale_pos_weight=1, seed=None, subsample=1):
        """
        Wrapper for XGBRegressor's functionality.
        Parameters per XGBRegressor's documentation:

            - base_score: The initial prediction score of all instances, global bias. (Default is 0.5)
            
            - booster: Specify which booster to use: gbtree, gblinear or dart. (Default is "gbtree")
            
            - colsample_bylevel: Subsample ratio of columns for each level. (Default is 1)
            
            - colsample_bytree: Subsample ratio of columns when constructing each tree. (Default is 1)
            
            - gamma: Minimum loss reduction required to make a further partition on a leaf node of the tree. (Default is 0)
            
            - learning_rate: Boosting learning rate. (Default is 0.1)
            
            - max_delta_step: Maximum delta step we allow each tree’s weight estimation to be. (Default is 0)
            
            - max_depth: Maximum tree depth for base learners. (Default is 3)
            
            - min_child_weight: Minimum sum of instance weight (hessian) needed in a child. (Default is 1)
            
            - missing: Value in the data which needs to be present as a missing value. (Default is None)
            
            - n_estimators: Number of gradient boosted trees. Equivalent to number of boosting rounds. (Default is 100)
            
            - n_jobs: Number of parallel threads used to run xgboost. (Default is 1)
            
            - nthread: Number of threads to use for loading data when parallelization is applicable. If -1, uses maximum
            threads available on the system. (Default is None)
            
            - objective: Specify the learning task and the corresponding learning objective or a custom objective
            function to be used. (Default is "reg:squarederror")
            
            - random_state: Random number seed. (Default is 42)
            
            - reg_alpha: L1 regularization term on weights. (Default is 0)
            
            - reg_lambda: L2 regularization term on weights. (Default is 1)
            
            - scale_pos_weight: Balancing of positive and negative weights. (Default is 1)
            
            - seed: Used to generate the folds. (Default is None)
            
            - subsample: Subsample ratio of the training instance. (Default is 1)
        """
        if self._check_inputs():
            # Initialize regressor
            self.regressor =\
                XGBRegressor(base_score=base_score, booster=booster, colsample_bylevel=colsample_bylevel,
                             colsample_bytree=colsample_bytree, gamma=gamma, learning_rate=learning_rate,
                             max_delta_step=max_delta_step, max_depth=max_depth, min_child_weight=min_child_weight,
                             missing=missing, n_estimators=n_estimators, n_jobs=n_jobs, nthread=nthread,
                             objective=objective, random_state=random_state, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                             scale_pos_weight=scale_pos_weight, seed=seed, subsample=subsample)

            # If dataset hasn't been split into training/testing sets yet, do so now
            if self.dataset_X_test is None:
                self._split_data()

            # Train the model and get resultant coefficients; handle exception if arguments aren't correct
            try:
                self.regressor.fit(self.dataset_X_train, self.dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the regression model. Check your inputs and try again.")
                print("Does labels only contain quantitative data?")
                print("Here is the exception message:")
                print(e)
                self.regressor = None
                return

            # Make predictions using testing set
            y_prediction = self.regressor.predict(self.dataset_X_test)

            # Get mean squared error, coefficient of determination, and correlation coefficient
            self.mean_squared_error = mean_squared_error(self.dataset_y_test, y_prediction)
            self.r2_score = r2_score(self.dataset_y_test, y_prediction)
            self.r_score = sqrt(self.r2_score)
    
    # Wrapper for classification functionality

    def run_classifier(self, max_depth=3, learning_rate=0.1, n_estimators=100, objective='binary:logistic',
                       booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0,
                       subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1,
                       scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None):
        """
        Wrapper for XGBClassifier's functionality.
        Parameters per XGBClassifier's documentation:

            - max_depth: Maximum tree depth for base learners. (Default is 3)

            - learning_rate: Boosting learning rate. (Default is 0.1)

            - n_estimators: Number of gradient boosted trees. Equivalent to number of boosting rounds. (Default is 100)

            - objective: Specify the learning task and the corresponding learning objective or a custom objective
            function to be used. (Default is "binary:logistic")

            - booster: Specify which booster to use: gbtree, gblinear, or dart. (Default is "gbtree")

            - n_jobs: Number of parallel threads used to run xgboost. (Default is 1)

            - nthread: Number of threads to use for loading data when parallelization is applicable. If -1, uses
            maximum threads available on the system. (Default is None)

            - gamma: Minimum loss reduction required to make a further partition on a leaf node of the tree.
            (Default is 0)

            - min_child_weight: Minimum sum of instance weight(hessian) needed in a child. (Default is 1)

            - max_delta_step: Maximum delta step we allow each tree's weight estimation to be. (Default is 0)

            - subsample: Subsample ratio of the training instance. (Default is 1)

            - colsample_bylevel: Subsample ratio of columns for each level. (Default is 1)
            
            - colsample_bytree: Subsample ratio of columns when constructing each tree. (Default is 1)

            - reg_alpha: L1 regularization term on weights. (Default is 0)
            
            - reg_lambda: L2 regularization term on weights. (Default is 1)

            - scale_pos_weight: Balancing of positive and negative weights. (Default is 1)

            - base_score: The initial prediction score of all instances, global bias. (Default is 0.5)

            - random_state: Random number seed. (Default is 0)

            - seed: Used to generate the folds. (Default is None)

            - missing: Value in the data which needs to be present as a missing value. (Default is None)
        """
        if self._check_inputs():
            # Initialize classifier
            self.classifier =\
                XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
                              objective=objective, booster=booster, n_jobs=n_jobs, nthread=nthread, gamma=gamma,
                              min_child_weight=min_child_weight, max_delta_step=max_delta_step, subsample=subsample,
                              colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel,
                              reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight,
                              base_score=base_score, random_state=random_state, seed=seed, missing=missing)

            # If dataset hasn't been split into training/testing sets yet, do so now
            if self.dataset_X_test is None:
                self._split_data()

            # Train the model and get resultant coefficients; handle exception if arguments aren't correct
            try:
                self.classifier.fit(self.dataset_X_train, np.ravel(self.dataset_y_train))
            except Exception as e:
                print("An exception occurred while training the classification model. Check your inputs and try again.")
                print("Here is the exception message:")
                print(e)
                self.classifier = None
                return

            # Make predictions using testing set
            y_prediction = self.classifier.predict(self.dataset_X_test)
            y_pred_probas = self.classifier.predict_proba(self.dataset_X_test)[::, 1]

            self.classes_ = self.classifier.classes_

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
        Verifies if the instance data is ready for use in XGBoost model.
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