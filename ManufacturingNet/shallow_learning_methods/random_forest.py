from math import sqrt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

class RandomForest:
    """
    Class framework for random forest classification and regression models.
    Initialized RandomForest objects provide classification and regression functionality via the methods
    run_classifier() and run_regressor(), respectively.
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

    def __init__(self, attributes=None, labels=None):
        """
        Initializes a RandomForest object.

        The following parameters are needed to use a RandomForest:

            – attributes: a numpy array of the independent variables
            – labels: a numpy array of the classes (for classification) or dependent variables (for regression)
            – test_size: the proportion of the dataset to be used for testing the model (defaults to 0.25);
            the proportion of the dataset to be used for training will be the complement of test_size

        After successfully running run_classifier(), the following instance data will be available:

            – classifier: the classification model trained using scikit-learn's random forest implementation
            – accuracy: the mean accuracy of the model on the given test data
            – roc_auc: the area under the ROC curve for the model
            – cross_val_scores_classifier: the cross validation score(s) for the classification model

        After successfully running run_regressor(), the following instance data will be available:

            – regressor: the regression model trained using scikit-learn's random forest implementation
            – r2_score: the coefficient of determination for the regression model
            – r_score: the correlation coefficient for the regression model
            – mean_squared_error: the mean squared error of the regression model
            – cross_val_scores_regressor: the cross validation score(s) for the regression model
        """
        self.attributes = attributes
        self.labels = labels
        self.test_size = None
        self.cv = None

        self.classifier = None
        self.accuracy = None
        self.roc_auc = None
        self.cross_val_scores_classifier = None

        self.regressor = None
        self.r2_score = None
        self.r_score = None
        self.mean_squared_error = None
        self.cross_val_scores_regressor = None
    
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

    def get_classifier(self):
        """
        Accessor method for classifier.

        Will return None if run_classifier() hasn't successfully run, yet.
        """
        return self.classifier

    def get_accuracy(self):
        """
        Accessor method for accuracy.

        Will return None if run_classifier() hasn't successfully run, yet.
        """
        return self.accuracy

    def get_roc_auc(self):
        """
        Accessor method for roc_auc.

        Will return None if run_classifier() hasn't successfully run, yet.
        """
        return self.roc_auc
    
    def get_cross_val_scores_classifier(self):
        """
        Accessor method for cross_val_scores_classifier.

        Will return None if run_classifier() hasn't successfully run, yet.
        """
        return self.cross_val_scores_classifier

    def get_regressor(self):
        """
        Accessor method for regressor.

        Will return None if run_regressor() hasn't successfully run, yet.
        """
        return self.regressor

    def get_r2_score(self):
        """
        Accessor me.

        Will return None if run_regressor() hasn't successfully run, yet.
        """
        return self.r2_score

    def get_r_score(self):
        """
        Accessor method for r_score.

        Will return None if run_regressor() hasn't successfully run, yet.
        """
        return self.r_score

    def get_mean_squared_error(self):
        """
        Accessor method for mean_squared_error.

        Will return None if run_regressor() hasn't successfully run, yet.
        """
        return self.mean_squared_error
    
    def get_cross_val_scores_regressor(self):
        """
        Accessor method for cross_val_scores_regressor.

        Will return None if run_regressor() hasn't successfully run, yet.
        """
        return self.cross_val_scores_regressor

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

    # Wrappers for RandomForest classes

    def run_classifier(self):
        """
        Provides random forest's classifier functionality.
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

            – cv: the number of folds to use for cross validation of model (defaults to None)
        """
        if self._check_inputs():
            # Initialize classifier
            self.classifier = self._create_model(classifier=True)
            
            # Split attributes and labels into training/testing data
            dataset_X_train, dataset_X_test, dataset_y_train, dataset_y_test =\
                train_test_split(self.attributes, self.labels, test_size=self.test_size)

            # Train classifier; handle exception if arguments are incorrect
            try:
                self.classifier.fit(dataset_X_train, dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the random forest classification model.",
                      "Check your arguments and try again.")
                print("Here is the exception message:")
                print(e)
                self.classifier = None
                return

            # Evaluate accuracy and ROC AUC of model using testing set and actual classification
            self.accuracy = self.classifier.score(dataset_X_test, dataset_y_test)
            self.roc_auc = roc_auc_score(self.classifier.predict(dataset_X_test),
                                         self.classifier.predict_proba(dataset_X_test)[::, 1])
            self.cross_val_scores_classifier = cross_val_score(self.classifier, self.attributes, self.labels, cv=self.cv)

            # Output results
            self._output_classifier_results()

    def run_regressor(self):
        """
        Provides random forest's regressor functionality.
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

            – cv: the number of folds to use for cross validation of model (defaults to None)
        """
        if self._check_inputs():
            # Initialize regressor
            self.regressor = self._create_model(classifier=False)

            # Split attributes and labels into training/testing data
            dataset_X_train, dataset_X_test, dataset_y_train, dataset_y_test =\
                train_test_split(self.attributes, self.labels, test_size=self.test_size)

            # Train regressor; handle exception if arguments are incorrect and/or labels isn't quantitative
            try:
                self.regressor.fit(dataset_X_train, dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the random forest regressor model.",
                      "Check your arguments and try again.")
                print("Does labels contain only quantitative data?")
                print("Here is the exception message:")
                print(e)
                self.regressor = None
                return
            
            # Evaluate accuracy measurements for model using testing data = self.regressor.score(dataset_X_test, dataset_y_test)
            self.r2_score = self.regressor.score(dataset_X_test, dataset_y_test)
            self.r_score = sqrt(self.r2_score)
            self.mean_squared_error =\
                mean_squared_error(dataset_y_test, self.regressor.predict(dataset_X_test))
            self.cross_val_scores_regressor = cross_val_score(self.regressor, self.attributes, self.labels, cv=self.cv)

            # Output results
            self._output_regressor_results()

    def predict_classifier(self, dataset_X=None):
        """
        Classifies each datapoint in dataset_X using the classifier model. Returns the predicted classifications.

        predict_classifier() can only run after run_classifier() has successfully trained the classifier model.
        """
        # Check that run_classifier() has already been called
        if self.classifier is None:
            print("The classifier model seems to be missing. Have you called run_classifier() yet?")
            return
        
        # Try to make the prediction; handle exception if dataset_X isn't a valid input
        try:
            y_prediction = self.classifier.predict(dataset_X)
        except Exception as e:
            print("The model failed to run. Check your inputs and try again.")
            print("Here is the exception message:")
            print(e)
            return
        
        print("\nRandomForestClassifier predictions:\n", y_prediction, "\n")
        return y_prediction
    
    def predict_regressor(self, dataset_X=None):
        """
        Predicts the output of each datapoint in dataset_X using the regressor model. Returns the predictions.

        predict_regressor() can only run after run_regressor() has successfully trained the regressor model.
        """
        # Check that run_regressor() has already been called
        if self.regressor is None:
            print("The regressor model seems to be missing. Have you called run_regressor() yet?")
            return
        
        # Try to make the prediction; handle exception if dataset_X isn't a valid input
        try:
            y_prediction = self.regressor.predict(dataset_X)
        except Exception as e:
            print("The model failed to run. Check your inputs and try again.")
            print("Here is the exception message:")
            print(e)
            return
        
        print("\nRandomForestRegressor predictions:\n", y_prediction, "\n")
        return y_prediction
    
    # Helper methods

    def _create_model(self, classifier):
        """
        Runs UI for getting parameters and creating classifier or regression model.
        """
        if classifier:
            print("\n===============================================")
            print("= Parameter inputs for RandomForestClassifier =")
            print("===============================================\n")
        else:
            print("\n==============================================")
            print("= Parameter inputs for RandomForestRegressor =")
            print("==============================================\n")
        
        if input("Use default parameters (Y/n)? ").lower() != "n":
            self.test_size = 0.25
            self.cv = None
            print("\n=======================================================")
            print("= End of parameter inputs; press any key to continue. =")
            input("=======================================================")
            
            if classifier:
                return RandomForestClassifier()
            else:
                return RandomForestRegressor()
        
        print("\nIf you are unsure about a parameter, press enter to use its default value.")
        print("If you finish entering parameters early, enter 'q' to skip ahead.\n")

        # Set defaults
        if classifier:
            criterion = "gini"
            class_weight = None
        else:
            criterion = "mse"
        
        self.test_size = 0.25
        self.cv = None
        n_estimators = 100
        max_depth = None
        min_samples_split = 2
        min_samples_leaf = 1
        min_weight_fraction_leaf = 0.0
        max_features = "auto"
        max_leaf_nodes = None
        min_impurity_decrease = 0.0
        min_impurity_split = None
        bootstrap = True
        oob_score = False
        n_jobs = None
        random_state = None
        verbose = 0
        warm_start = False
        ccp_alpha = 0.0
        max_samples = None

        while True:
            user_input = input("What fraction of the dataset should be the testing set? Input a decimal: ")

            try:
                self.test_size = float(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Input the number of folds for cross validation: ")

            try:
                self.cv = int(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Enter the number of trees in the forest: ")

            try:
                n_estimators = int(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            print("Which criteria should be used for measuring split quality?")
            if classifier:
                user_input = input("Enter 1 for 'gini' or 2 for 'entropy': ")

                if user_input == "q":
                    break
                elif user_input == "2":
                    criterion = "entropy"
            else:
                user_input = input("Enter 1 for 'mse' or 2 for 'mae': ")

                if user_input == "q":
                    break
                elif user_input == "2":
                    criterion = "mae"
            
            user_input = input("Enter the maximum depth of each tree: ")

            try:
                max_depth = int(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Enter the minimum number of samples required to split an internal node: ")

            try:
                if int(user_input) < 1:
                    min_samples_split = float(user_input)
                else:
                    min_samples_split = int(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Enter the minimum number of samples required to be at a leaf node: ")

            try:
                if int(user_input) < 1:
                    min_samples_leaf = float(user_input)
                else:
                    min_samples_leaf = int(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Enter the minimum weighted fraction of the weight total required to be at a leaf node: ")

            try:
                min_weight_fraction_leaf = float(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            print("How many features should be considered when looking for the best split?")
            print("Enter 'auto' to use n_features, 'sqrt' to use sqrt(n_features), 'log2' to use log2(n_features) or a number: ")
            user_input = input().lower()
            
            try:
                max_features = int(user_input)
            except:
                if user_input == "q":
                    break
                elif user_input == "sqrt":
                    max_features = "sqrt"
                elif user_input == "log2":
                    max_features = "log2"
            
            user_input = input("Enter the maximum number of leaf nodes: ")

            try:
                max_leaf_nodes = int(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Enter the minimum impurity decrease: ")

            try:
                min_impurity_decrease = float(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Enter the threshold for early stopping in tree growth: ")

            try:
                min_impurity_split = float(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Use bootstrap samples when building trees (Y/n)? ").lower()

            if user_input == "q":
                break
            elif user_input == "n":
                bootstrap = False
            
            user_input = input("Use out-of-bag samples to estimate R2 scores on unseen data (y/N)? ").lower()

            if user_input == "q":
                break
            elif user_input == "y":
                oob_score = True
            
            user_input = input("Enter the number of jobs to run in parallel: ")

            try:
                n_jobs = int(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Enter an integer for the random number seed: ")

            try:
                random_state = int(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Use verbose logging when fitting and predicting (y/N)? ").lower()

            if user_input == "q":
                break
            elif user_input == "y":
                verbose = 1

            user_input =\
                input("Use warm start? This will reuse the previous solution to fit and add more estimators (y/N): ").lower()

            if user_input == "q":
                break
            elif user_input == "y":
                warm_start = True
            
            user_input = input("Enter the complexity parameter for Minimal Cost-Complexity Pruning: ")

            try:
                if float(user_input) >= 0:
                    ccp_alpha = float(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            if bootstrap:
                user_input = input("Enter the maximum number of samples to train each base estimator: ")

                try:
                    if int(user_input) < 1:
                        max_samples = float(user_input)
                    else:
                        max_samples = int(user_input)
                except:
                    break

            break
        
        print("\n=======================================================")
        print("= End of parameter inputs; press any key to continue. =")
        input("=======================================================\n")

        if classifier:
            return RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                          min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                          min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                          max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                          min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score,
                                          n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start,
                                          class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)
        else:
            return RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                         min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                         max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                         min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score,
                                         n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start,
                                         ccp_alpha=ccp_alpha, max_samples=max_samples)

    def _output_classifier_results(self):
        """
        Outputs model metrics after run_classifier() finishes.
        """
        print("\n==================================")
        print("= RandomForestClassifier Results =")
        print("==================================\n")

        print("Classes:\n", self.classifier.classes_)
        print("\n{:<20} {:<20}".format("Accuracy:", self.accuracy))
        print("\n{:<20} {:<20}".format("ROC AUC:", self.roc_auc))
        print("\nCross Validation Scores:\n", self.cross_val_scores_classifier)
        print("\n\nCall predict_classifier() to make predictions for new data.")

        print("\n===================")
        print("= End of results. =")
        print("===================\n")
    
    def _output_regressor_results(self):
        """
        Outputs model metrics after run_regressor() finishes.
        """
        print("\n=================================")
        print("= RandomForestRegressor Results =")
        print("=================================\n")

        print("{:<20} {:<20}".format("Mean Squared Error:", self.mean_squared_error))
        print("\n{:<20} {:<20}".format("R2 Score:", self.r2_score))
        print("\n{:<20} {:<20}".format("R Score:", self.r_score))
        print("\nCross Validation Scores:\n", self.cross_val_scores_regressor)
        print("\n\nCall predict_regressor() to make predictions for new data.")

        print("\n===================")
        print("= End of results. =")
        print("===================\n")

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

        return True