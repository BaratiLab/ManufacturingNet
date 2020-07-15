from math import sqrt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

class RandomForest:
    """
    Wrapper class around scikit-learn's random forest classification and regression functionality.
    Initialized RandomForest objects provide classification and regression functionality via the methods
    random_forest_classifier() and random_forest_regressor(), respectively.
    Per scikit-learn's documentation:

    In random forests, each tree in the ensemble is built from a sample drawn with replacement
    (i.e., a bootstrap sample) from the training set. Furthermore, when splitting each node during the construction of
    a tree, the best split is found either from all input features or a random subset of size max_features. The purpose
    of these two sources of randomness is to decrease the variance of the forest estimator. Indeed, individual decision
    trees typically exhibit high variance and tend to overfit. The injected randomness in forests yield decision trees
    with somewhat decoupled prediction errors. By taking an average of those predictions, some errors can cancel out.
    Random forests achieve a reduced variance by combining diverse trees, sometimes at the cost of a slight increase in
    bias. In practice the variance reduction is often significant hence yielding an overall better model.
    """

    def __init__(self, attributes=None, labels=None, test_size=0.25):
        """
        Initializes a RandomForest object.

        The following parameters are needed to use a RandomForest:

            – attributes: a numpy array of the independent variables
            – labels: a numpy array of the classes (for classification) or dependent variables (for regression)
            – test_size: the proportion of the dataset to be used for testing the model (defaults to 0.25);
            the proportion of the dataset to be used for training will be the complement of test_size

        After successfully running random_forest_classifier(), the following instance data will be available:

            – classifier_RF: the classification model trained using scikit-learn's random forest implementation
            – classifier_accuracy: the mean accuracy of the model on the given test data
            – classifier_roc_auc: the area under the ROC curve for the model

        After successfully running random_forest_regressor(), the following instance data will be available:

            – regressor_RF: the regression model trained using scikit-learn's random forest implementation
            – regressor_r2_score: the coefficient of determination for the regression model
            – regressor_r_score: the correlation coefficient for the regression model
        """
        self.attributes = attributes
        self.labels = labels
        self.test_size = test_size

        self.classifier_RF = None
        self.classifier_accuracy = None
        self.classifier_roc_auc = None
        self.regressor_RF = None
        self.regressor_r2_score = None
        self.regressor_r_score = None
    
    # Accessor methods

    def get_attributes(self):
        """
        Accessor method for attributes.

        If a RandomForest object is initialized without specifying attributes, attributes will be None. No
        classification or regression functionality can be used until attributes is a populated numpy array. Call
        set_attributes(new_attributes) to fix this.
        """
        return self.attributes

    def get_labels(self):
        """
        Accessor method for labels.

        If a RandomForest object is initialized without specifying labels, labels will be None. No classification or
        regression functionality can be used until labels is a populated numpy array. Call set_labels(new_labels) to
        fix this
        """
        return self.labels

    def get_test_size(self):
        """
        Accessor method for test_size.

        Should be a number or None.
        """
        return self.test_size

    def get_classifier_RF(self):
        """
        Accessor method for classifier_RF.

        Will return None if random_forest_classifier() hasn't successfully run, yet.
        """
        return self.classifier_RF

    def get_classifier_accuracy(self):
        """
        Accessor method for classifier_accuracy.

        Will return None if random_forest_classifier() hasn't successfully run, yet.
        """
        return self.classifier_accuracy

    def get_classifier_roc_auc(self):
        """
        Accessor method for classifier_roc_auc.

        Will return None if random_forest_classifier() hasn't successfully run, yet.
        """
        return self.classifier_roc_auc

    def get_regressor_RF(self):
        """
        Accessor method for regressor_RF.

        Will return None if random_forest_regressor() hasn't successfully run, yet.
        """
        return self.regressor_RF

    def get_regressor_r2_score(self):
        """
        Accessor method for regressor_r2_score.

        Will return None if random_forest_regressor() hasn't successfully run, yet.
        """
        return self.regressor_r2_score

    def get_regressor_r_score(self):
        """
        Accessor method for regressor_r_score.

        Will return None if random_forest_regressor() hasn't successfully run, yet.
        """
        return self.regressor_r_score

    # Modifier methods

    def set_attributes(self, new_attributes=None):
        """
        Modifier method for attributes.

        Input should be a populated numpy array. Defaults to None.
        """
        self.attributes = new_attributes

    def set_labels(self, new_labels=None):
        """
        Modifier method for labels.

        Input should be a populated numpy array. Defaults to None.
        """
        self.labels = new_labels

    def set_test_size(self, new_test_size=0.25):
        """
        Modifier method for test_size.

        Input should be a float between 0.0 and 1.0 or None. Defaults to 0.25. The training size will be set to the
        complement of test_size.
        """
        self.test_size = new_test_size

    # Wrappers for RandomForest classes

    def random_forest_classifier(self, n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                 max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                 bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,
                                 warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None):
        """
        Wrapper for scikit-learn's random forest classifier functionality.
        Per scikit-learn's documentation:
            A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples
            of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The
            sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the
            whole dataset is used to build each tree.
        Parameters per scikit-learn's documentation:

            – n_estimators: The number of trees in the forest. (Default is 100)
            
            – criterion: The function to measure the quality of a split. Supported criteria are “gini” for the Gini
            impurity and “entropy” for the information gain. Note: this parameter is tree-specific. (Default is "gini")

            – max_depth: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or
            until all leaves contain less than min_samples_split samples. (Default is None)

            – min_samples_split: The minimum number of samples required to split an internal node. If int, then consider
            min_samples_split as the minimum number. If float, then min_samples_split is a fraction and
            ceil(min_samples_split * n_samples) are the minimum number of samples for each split. (Default is 2)

            – min_samples_leaf: The minimum number of samples required to be at a leaf node. A split point at any depth
            will only be considered if it leaves at least min_samples_leaf training samples in each of the left and
            right branches. This may have the effect of smoothing the model, especially in regression. If int, then
            consider min_samples_leaf as the minimum number. If float, then min_samples_leaf is a fraction and
            ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node. (Default is 1)

            – min_weight_fraction_leaf: The minimum weighted fraction of the sum total of weights (of all the input
            samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
            (Default is 0.0)

            – max_features: The number of features to consider when looking for the best split. If int, then consider
            max_features features at each split. If float, then max_features is a fraction and
            int(max_features * n_features) features are considered at each split. If "auto", then
            max_features=sqrt(n_features). If "sqrt", same behavior as "auto". If "log2", then
            max_features=log2(n_features). If None, then max_features=n_features. (Default is "auto")

            – max_leaf_nodes: Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative
            reduction in impurity. If None then unlimited number of leaf nodes.

            – min_impurity_decrease: A node will be split if this split induces a decrease of the impurity greater than
            or equal to this value. The weighted impurity decrease equation is the following:
                
                N_t / N * (impurity - N_t_R / N_t * right_impurity
                                    - N_t_L / N_t * left_impurity)

            where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the
            number of samples in the left child, and N_t_R is the number of samples in the right child. N, N_t, N_t_R
            and N_t_L all refer to the weighted sum, if sample_weight is passed. (Default is 0.0)

            – min_impurity_split: Threshold for early stopping in tree growth.
            A node will split if its impurity is above the threshold, otherwise it is a leaf. (Default is None)

            Note: min_impurity_split has been deprecated in favor of min_impurity_decrease. Use min_impurity_decrease
            instead.

            – bootstrap: Whether bootstrap samples are used when building trees. If False, the whole dataset is used to
            build each tree. (Default is True)
            
            – oob_score: Whether to use out-of-bag samples to estimate the generalization accuracy. (Default is False)

            – n_jobs: The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized
            over the trees. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
            (Default is None)

            – random_state: Controls both the randomness of the bootstrapping of the samples used when building trees
            (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each
            node (if max_features < n_features). (Default is None)

            – verbose: Controls the verbosity when fitting and predicting. (Default is 0)

            – warm_start: When set to True, reuse the solution of the previous call to fit and add more estimators to
            the ensemble, otherwise, just fit a whole new forest. (Default is False)

            – class_weight: Weights associated with classes in the form {class_label: weight}. If not given, all classes
            are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same
            order as the columns of y. The “balanced” mode uses the values of y to automatically adjust weights
            inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
            The “balanced_subsample” mode is the same as “balanced” except that weights are computed based on the
            bootstrap sample for every tree grown. For multi-output, the weights of each column of y will be multiplied.
            Note that these weights will be multiplied with sample_weight (passed through the fit method) if
            sample_weight is specified. (Default is None)

            – ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest
            cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed.
            (Default is 0.0)

            – max_samples: If bootstrap is True, the number of samples to draw from X to train each base estimator.
            If None, then draw X.shape[0] samples. If int, then draw max_samples samples. If float, then draw
            max_samples * X.shape[0] samples; thus, max_samples should be in the interval (0, 1). (Default is None)
        """
        if self._check_inputs():
            # Initialize classifier
            self.classifier_RF =\
                RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                       min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                       max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                       min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score,
                                       n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start,
                                       class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)
            
            # Split attributes and labels into training/testing data
            dataset_X_train, dataset_X_test, dataset_y_train, dataset_y_test =\
                train_test_split(self.attributes, self.labels, test_size=self.test_size)

            # Train classifier; handle exception if arguments are incorrect
            try:
                self.classifier_RF.fit(dataset_X_train, dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the random forest classification model.",
                      "Check your arguments and try again.")
                print("Here is the exception message:")
                print(e)
                self.classifier_RF = None
                return

            # Evaluate accuracy and ROC AUC of model using testing set and actual classification
            self.classifier_accuracy = self.classifier_RF.score(dataset_X_test, dataset_y_test)
            self.classifier_roc_auc = roc_auc_score(self.classifier_RF.predict(dataset_X_test),
                                                    self.classifier_RF.predict_proba(dataset_X_test)[::, 1])

    def random_forest_regressor(self, n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2,
                                min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True,
                                oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False,
                                ccp_alpha=0.0, max_samples=None):
        """
        Wrapper for scikit-learn's random forest regressor functionality.
        Per scikit-learn's documentation:
            A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples
            of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The
            sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the
            whole dataset is used to build each tree.
        Parameters per scikit-learn's documentation:

            – n_estimators: The number of trees in the forest. (Default is 100)

            – criterion: The function to measure the quality of a split. Supported criteria are “mse” for the mean
            squared error, which is equal to variance reduction as feature selection criterion, and “mae” for the mean
            absolute error. (Default is "mse")

            – max_depth: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or
            until all leaves contain less than min_samples_split samples. (Default is None)

            – min_samples_split: The minimum number of samples required to split an internal node. If int, then consider
            min_samples_split as the minimum number. If float, then min_samples_split is a fraction and
            ceil(min_samples_split * n_samples) are the minimum number of samples for each split. (Default is 2)

            – min_samples_leaf: The minimum number of samples required to be at a leaf node. A split point at any depth
            will only be considered if it leaves at least min_samples_leaf training samples in each of the left and
            right branches. This may have the effect of smoothing the model, especially in regression. If int, then
            consider min_samples_leaf as the minimum number. If float, then min_samples_leaf is a fraction and
            ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node. (Default is 1)

            – min_weight_fraction_leaf: The minimum weighted fraction of the sum total of weights (of all the input
            samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
            (Default is 0.0)

            – max_features: The number of features to consider when looking for the best split. If int, then consider
            max_features features at each split. If float, then max_features is a fraction and
            int(max_features * n_features) features are considered at each split. If "auto", then max_features=n_features.
            If "sqrt", then max_features=sqrt(n_features). If "log2", then max_features=log2(n_features). If None, then
            max_features=n_features. (Default is "auto")

            – max_leaf_nodes: Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative
            reduction in impurity. If None, then unlimited number of leaf nodes. (Default is None)

            – min_impurity_decrease: A node will be split if this split induces a decrease of the impurity greater than
            or equal to this value. The weighted impurity decrease equation is the following:
                
                N_t / N * (impurity - N_t_R / N_t * right_impurity
                                    - N_t_L / N_t * left_impurity)

            where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the
            number of samples in the left child, and N_t_R is the number of samples in the right child. N, N_t, N_t_R
            and N_t_L all refer to the weighted sum, if sample_weight is passed. (Default is 0.0)

            – min_impurity_split: Threshold for early stopping in tree growth.
            A node will split if its impurity is above the threshold, otherwise it is a leaf. (Default is None)

            Note: min_impurity_split has been deprecated in favor of min_impurity_decrease. Use min_impurity_decrease
            instead.

            – bootstrap: Whether bootstrap samples are used when building trees. If False, the whole dataset is used to
            build each tree. (Default is True)
            
            – oob_score: Whether to use out-of-bag samples to estimate the generalization accuracy. (Default is False)

            – n_jobs: The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized
            over the trees. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
            (Default is None)

            – random_state: Controls both the randomness of the bootstrapping of the samples used when building trees
            (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each
            node (if max_features < n_features). (Default is None)

            – verbose: Controls the verbosity when fitting and predicting. (Default is 0)

            – warm_start: When set to True, reuse the solution of the previous call to fit and add more estimators to
            the ensemble, otherwise, just fit a whole new forest. (Default is False)

            – ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest
            cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed.
            (Default is 0.0)

            – max_samples: If bootstrap is True, the number of samples to draw from X to train each base estimator.
            If None, then draw X.shape[0] samples. If int, then draw max_samples samples. If float, then draw
            max_samples * X.shape[0] samples; thus, max_samples should be in the interval (0, 1). (Default is None)
        """
        if self._check_inputs():
            # Initialize regressor
            self.regressor_RF =\
                RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                      min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                      max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                      min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score,
                                      n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start,
                                      ccp_alpha=ccp_alpha, max_samples=max_samples)

            # Split attributes and labels into training/testing data
            dataset_X_train, dataset_X_test, dataset_y_train, dataset_y_test =\
                train_test_split(self.attributes, self.labels, test_size=self.test_size)

            # Train regressor; handle exception if arguments are incorrect and/or labels isn't quantitative
            try:
                self.regressor_RF.fit(dataset_X_train, dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the random forest regressor model.",
                      "Check your arguments and try again.")
                print("Does labels contain only quantitative data?")
                print("Here is the exception message:")
                print(e)
                self.regressor_RF = None
                return
            
            # Evaluate coefficient of determination and correlation coefficient for model using testing data
            self.regressor_r2_score = self.regressor_RF.score(dataset_X_test, dataset_y_test)
            self.regressor_r_score = sqrt(self.regressor_r2_score)

    # Helper methods

    def _check_inputs(self):
        """
        Verifies if instance data is ready for use in RandomForest models.
        """

        # Check if attributes exists
        if self.attributes is None:
            print("attributes is missing; call set_attributes(new_attributes) to fix this! new_attributes should be a",
                  "populated dataset of independent variables.")
            return False

        # Check if labels exists
        if self.labels is None:
            print("labels is missing; call set_labels(new_labels) to fix this! new_labels should be a populated dataset",
                  "of classes (for classification) or dependent variables (for regression).")
            return False

        # Check if attributes and labels have same number of rows (samples)
        if self.attributes.shape[0] != self.labels.shape[0]:
            print("attributes and labels don't have the same number of rows. Make sure the number of samples in each",
                  "dataset matches!")
            return False

        # Check if test_size is a number
        if self.test_size is not None and not isinstance(self.test_size, (int, float)):
            print("test_size must be None or a number; call set_test_size(new_test_size) to fix this!")
            return False

        return True