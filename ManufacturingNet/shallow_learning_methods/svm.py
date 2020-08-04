from math import sqrt
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC, NuSVC, LinearSVC, SVR, NuSVR, LinearSVR

class SVM:
    """
    Class model for support vector machine (SVM) model.
    This class supports binary and multi-class classification and regression.
    Per scikit-learn's documentation:

    Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and
    outliers detection.

    The advantages of support vector machines are:

        – Effective in high dimensional spaces.
        – Still effective in cases where number of dimensions is greater than the number of samples.
        – Uses a subset of training points in the decision function (called support vectors), so it is also memory
        efficient.
        – Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided,
        but it is also possible to specify custom kernels.

    The disadvantages of support vector machines include:

        – If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel
        functions and regularization term is crucial.
        – SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold
        cross-validation.
    """

    def __init__(self, attributes=None, labels=None):
        """
        Initializes a SVM object.

        The following parameters are needed to use a SVM:

            – attributes: a numpy array of the independent variables
            – labels: a numpy array of the classes (for classification) or dependent variables (for regression)
            – test_size: the proportion of the dataset to be used for testing the model (defaults to 0.25);
            the proportion of the dataset to be used for training will be the complement of test_size

        After successfully running one of the classifier methods (run_SVC(), run_nu_SVC(), or run_linear_SVC()), the
        corresponding classifier below will be trained:

            – classifier_SVC: a classifier trained using scikit-learn's SVC implementation
            – accuracy_SVC: the accuracy of the SVC model, based on its predictions for dataset_X_test
            – roc_auc_SVC: the area under the ROC curve for the SVC model
            – cross_val_scores_SVC: the cross validation score(s) for the SVC model
            – classifier_nu_SVC: a classifier trained using scikit-learn's NuSVC implementation
            – accuracy_nu_SVC: the accuracy of the NuSVC model, based on its predictions for dataset_X_test
            – roc_auc_nu_SVC: the area under the ROC curve for the NuSVC model
            – cross_val_scores_nu_SVC: the cross validation score(s) for the NuSVC model
            – classifier_linear_SVC: a classifier trained using scikit-learn's LinearSVC implementation
            – accuracy_linear_SVC: the accuracy of the LinearSVC model, based on its predictions for dataset_X_test
            – cross_val_scores_linear_SVC: the cross validation score(s) for the LinearSVC model

        After successfully running one of the regression methods (SVR(), nu_SVR(), or linear_SVR()), the corresponding
        regression model below will be trained:

            – regressor_SVR: a regression model trained using scikit-learn's SVR implementation
            – r2_score_SVR: the coefficient of determination for the SVR model
            – r_score_SVR: the correlation coefficient for the SVR model
            – cross_val_scores_SVR: the cross validation score(s) for the SVR model
            – regressor_nu_SVR: a regression model trained using scikit-learn's NuSVR implementation
            – r2_score_nu_SVR: the coefficient of determination for the NuSVR model
            – r_score_nu_SVR: the correlation coefficient for the NuSVR model
            – cross_val_scores_nu_SVR: the cross validation score(s) for the NuSVR model
            – regressor_linear_SVR: a regression model trained using scikit-learn's LinearSVR implementation
            – r2_score_linear_SVR: the coefficient of determination for the LinearSVR model
            – r_score_linear_SVR: the correlation coefficient for the LinearSVR model
            – cross_val_scores_linear_SVR: the cross validation score(s) for the LinearSVR model
        """
        self.attributes = attributes
        self.labels = labels
        self.test_size = None
        self.cv = None

        self.classifier_SVC = None
        self.accuracy_SVC = None
        self.roc_auc_SVC = None
        self.cross_val_scores_SVC = None
        self.classifier_nu_SVC = None
        self.accuracy_nu_SVC = None
        self.roc_auc_nu_SVC = None
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

        # References to training and testing subsets of dataset; instance data for re-use purposes
        self.dataset_X_train = None
        self.dataset_y_train = None
        self.dataset_X_test = None
        self.dataset_y_test = None

    # Accessor Methods

    def get_attributes(self):
        """
        Accessor method for attributes.

        If a SVM object is initialized without specifying attributes, attributes will be None. No SVM functionality can
        be used until attributes is a populated numpy array. Call set_attributes(new_attributes) to fix this.
        """
        return self.attributes

    def get_labels(self):
        """
        Accessor method for labels.

        If a SVM object is initialized without specifying labels, labels will be None. No SVM functionality can be used
        until labels is a populated numpy array. Call set_labels(new_labels) to fix this.
        """
        return self.labels

    def get_classifier_SVC(self):
        """
        Accessor method for classifier_SVC.

        Will return None if run_SVC() hasn't successfully run, yet.
        """
        return self.classifier_SVC

    def get_accuracy_SVC(self):
        """
        Accessor method for accuracy_SVC.

        Will return None if run_SVC() hasn't successfully run, yet.
        """
        return self.accuracy_SVC

    def get_roc_auc_SVC(self):
        """
        Accessor method for roc_auc_SVC.

        Will return None if run_SVC() hasn't successfully run, yet.
        """
        return self.roc_auc_SVC
    
    def get_cross_val_scores_SVC(self):
        """
        Accessor method for cross_val_scores_SVC.

        Will return None if run_SVC() hasn't successfully run, yet.
        """
        return self.cross_val_scores_SVC

    def get_classifier_nu_SVC(self):
        """
        Accessor method for classifier_nu_SVC.

        Will return None if run_nu_SVC() hasn't successfully run, yet.
        """
        return self.classifier_nu_SVC

    def get_accuracy_nu_SVC(self):
        """
        Accessor method for accuracy_nu_SVC.

        Will return None if run_nu_SVC() hasn't successfully run, yet.
        """
        return self.accuracy_nu_SVC

    def get_roc_auc_nu_SVC(self):
        """
        Accessor method for roc_auc_nu_SVC.

        Will return None if run_nu_SVC() hasn't successfully run, yet.
        """
        return self.roc_auc_nu_SVC
    
    def get_cross_val_scores_nu_SVC(self):
        """
        Accessor method for cross_val_scores_nu_SVC.

        Will return None if run_nu_SVC() hasn't successfully run, yet.
        """
        return self.cross_val_scores_nu_SVC

    def get_classifier_linear_SVC(self):
        """
        Accessor method for classifier_linear_SVC.

        Will return None if run_linear_SVC() hasn't successfully run, yet.
        """
        return self.classifier_linear_SVC

    def get_accuracy_linear_SVC(self):
        """
        Accessor method for accuracy_linear_SVC.

        Will return None if run_linear_SVC() hasn't successfully run, yet.
        """
        return self.accuracy_linear_SVC
    
    def get_cross_val_scores_linear_SVC(self):
        """
        Accessor method for cross_val_scores_linear_SVC.

        Will return None if run_linear_SVC() hasn't successfully run, yet.
        """
        return self.cross_val_scores_linear_SVC

    def get_regressor_SVR(self):
        """
        Accessor method for regressor_SVR.

        Will return None if run_SVR() hasn't successfully run, yet.
        """
        return self.regressor_SVR
    
    def get_mean_squared_error_SVR(self):
        """
        Accessor method for mean_squared_error_SVR.

        Will return None if run_SVR() hasn't successfully run, yet.
        """
        return self.mean_squared_error_SVR

    def get_r2_score_SVR(self):
        """
        Accessor method for r2_score_SVR.

        Will return None if run_SVR() hasn't successfully run, yet.
        """
        return self.r2_score_SVR

    def get_r_score_SVR(self):
        """
        Accessor method for r_score_SVR.

        Will return None if run_SVR() hasn't successfully run, yet.
        """
        return self.r_score_SVR
    
    def get_cross_val_scores_SVR(self):
        """
        Accessor method for cross_val_scores_SVR.

        Will return None if run_SVR() hasn't successfully run, yet.
        """
        return self.cross_val_scores_SVR

    def get_regressor_nu_SVR(self):
        """
        Accessor method for regressor_nu_SVR.

        Will return None if run_nu_SVR() hasn't successfully run, yet.
        """
        return self.regressor_nu_SVR
    
    def get_mean_squared_error_nu_SVR(self):
        """
        Accessor method for mean_squared_error_nu_SVR.

        Will return None if run_nu_SVR() hasn't successfully run, yet.
        """
        return self.mean_squared_error_nu_SVR

    def get_r2_score_nu_SVR(self):
        """
        Accessor method for r2_score_nu_SVR.

        Will return None if run_nu_SVR() hasn't successfully run, yet.
        """
        return self.r2_score_nu_SVR

    def get_r_score_nu_SVR(self):
        """
        Accessor method for r_score_nu_SVR.

        Will return None if run_nu_SVR() hasn't successfully run, yet.
        """
        return self.r_score_nu_SVR
    
    def get_cross_val_scores_nu_SVR(self):
        """
        Accessor method for cross_val_scores_nu_SVR.

        Will return None if run_nu_SVR() hasn't successfully run, yet.
        """
        return self.cross_val_scores_nu_SVR

    def get_regressor_linear_SVR(self):
        """
        Accessor method for regressor_linear_SVR.

        Will return None if run_linear_SVR() hasn't successfully run, yet.
        """
        return self.regressor_linear_SVR
    
    def get_mean_squared_error_linear_SVR(self):
        """
        Accessor method for mean_squared_error_linear_SVR.

        Will return None if run_linear_SVR() hasn't successfully run, yet.
        """
        return self.mean_squared_error_linear_SVR

    def get_r2_score_linear_SVR(self):
        """
        Accessor method for r2_score_linear_SVR.

        Will return None if run_linear_SVR() hasn't successfully run, yet.
        """
        return self.r2_score_linear_SVR

    def get_r_score_linear_SVR(self):
        """
        Accessor method for r_score_linear_SVR.

        Will return None if run_linear_SVR() hasn't successfully run, yet.
        """
        return self.r_score_linear_SVR
    
    def get_cross_val_scores_linear_SVR(self):
        """
        Accessor method for cross_val_scores_linear_SVR.

        Will return None if run_linear_SVR() hasn't successfully run, yet.
        """
        return self.cross_val_scores_linear_SVR

    # Modifier Methods

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

    # Wrappers for SVM classification classes

    def run_SVC(self):
        """
        Runs SVC model.
        Parameters per scikit-learn's documentation:

            – C: Regularization parameter. The strength of the regularization is inversely proportional to C.
            Must be strictly positive. The penalty is a squared l2 penalty. (Default is 1.0)

            – kernel: Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’,
            ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is
            used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape
            (n_samples, n_samples). (Default is "rbf")
            
            – degree: Degree of the polynomial kernel function ("poly"). Ignored by all other kernels. (Default is 3)
            
            – gamma: Kernel coefficient for "rbf", "poly", and "sigmoid". If gamma="scale", then it uses
            1 / (n_features * training_samples.var()) as value of gamma. IF gamma="auto", it uses 1 / n_features.
            (Default is "scale")
            
            – coef0: Independent term in kernel function. It is only significant in "poly" and "sigmoid". (Default is 0.0)
            
            – shrinking: Whether to use the shrinking heuristic. (Default is True)
            
            – probability: Whether to enable probability estimates. This must be enabled prior to calling fit, will slow
            down that method as it internally uses 5-fold cross-validation, and predict_proba may be inconsistent with
            predict. (Default is False)
            
            – tol: Tolerance for stopping criterion. (Default is 1e-3, or 0.001)
            
            – cache_size: Specify the size of the kernel cache in MB. (Default is 200)
            
            – class_weight: Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are
            supposed to have weight one. The “balanced” mode uses the values of y to automatically adjust weights
            inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
            (Default is None)
            
            – verbose: Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in
            libsvm that, if enabled, may not work properly in a multithreaded context. (Default is False)
            
            – max_iter: Hard limit on iterations within solver, or -1 for no limit. (Default is -1)
            
            – decision_function_shape: Whether to return a one-vs-rest (‘ovr’) decision function of shape
            (n_samples, n_classes) as all other classifiers, or the original one-vs-one (‘ovo’) decision function of
            libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one (‘ovo’) is always
            used as multi-class strategy. The parameter is ignored for binary classification. (Default is "ovr")
            
            – break_ties: If true, decision_function_shape='ovr', and number of classes > 2, predict will break ties
            according to the confidence values of decision_function; otherwise the first class among the tied classes is
            returned. Please note that breaking ties comes at a relatively high computational cost compared to a simple
            predict. (Default is False)
            
            – random_state: Controls the pseudo random number generation for shuffling the data for probability
            estimates. Ignored when probability is False. Pass an int for reproducible output across multiple function
            calls. (Default is None)

            – cv: the number of folds to use for cross validation of model (defaults to None)

        The implementation is based on libsvm. The fit time scales at least quadratically with the number of samples
        and may be impractical beyond tens of thousands of samples.
        """
        if self._check_inputs():
            # Initialize classifier
            self.classifier_SVC = self._create_SVC_model(is_nu=False)

            # Split data, if needed; if testing/training sets are still None, call _split_data()
            if self.dataset_X_test is None:
                self._split_data()

            # Train classifier; handle exception if arguments are incorrect
            try:
                self.classifier_SVC.fit(self.dataset_X_train, self.dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the SVC model. Check your arguments and try again.")
                print("Here is the exception message:")
                print(e)
                self.classifier_SVC = None
                return

            # Evaluate metrics of model
            self.accuracy_SVC = self.classifier_SVC.score(self.dataset_X_test, self.dataset_y_test)
            self.cross_val_scores_SVC = cross_val_score(self.classifier_SVC, self.attributes, self.labels, cv=self.cv)

            self.roc_auc_SVC = roc_auc_score(self.classifier_SVC.predict(self.dataset_X_test),
                                             self.classifier_SVC.predict_proba(self.dataset_X_test)[::, 1])

            # Output results
            self._output_classifier_results(model="SVC")

    def predict_SVC(self, dataset_X=None):
        """
        Classifies each datapoint in dataset_X using the SVC model. Returns the predicted classification.

        predict_SVC() can only run after run_SVC() has successfully trained the SVC model.
        """
        # Check that run_SVC() has already been called
        if self.classifier_SVC is None:
            print("The SVC model seems to be missing. Have you called run_SVC() yet?")
            return
        
        # Try to make the prediction; handle exception if dataset_X isn't a valid input
        try:
            y_prediction = self.classifier_SVC.predict(dataset_X)
        except Exception as e:
            print("The SVC model failed to run. Check your inputs and try again.")
            print("Here is the exception message:")
            print(e)
            return
        
        print("\nSVC predictions:\n", y_prediction, "\n")
        return y_prediction

    def run_nu_SVC(self):
        """
        Runs NuSVC model.
        Per scikit-learn's documentation, NuSVC is similar to SVC, but uses a parameter, nu, to set the number of
        support vectors.
        Parameters per scikit-learn's documentation:

            – nu: An upper bound on the fraction of margin errors and a lower bound of the fraction of support vectors.
            Should be in the interval (0, 1]. (Default is 0.5)
            
            – C: Regularization parameter. The strength of the regularization is inversely proportional to C.
            Must be strictly positive. The penalty is a squared l2 penalty. (Default is 1.0)

            – kernel: Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’,
            ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is
            used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape
            (n_samples, n_samples). (Default is "rbf")
            
            – degree: Degree of the polynomial kernel function ("poly"). Ignored by all other kernels. (Default is 3)
            
            – gamma: Kernel coefficient for "rbf", "poly", and "sigmoid". If gamma="scale", then it uses
            1 / (n_features * training_samples.var()) as value of gamma. IF gamma="auto", it uses 1 / n_features.
            (Default is "scale")
            
            – coef0: Independent term in kernel function. It is only significant in "poly" and "sigmoid". (Default is 0.0)
            
            – shrinking: Whether to use the shrinking heuristic. (Default is True)
            
            – probability: Whether to enable probability estimates. This must be enabled prior to calling fit, will slow
            down that method as it internally uses 5-fold cross-validation, and predict_proba may be inconsistent with
            predict. (Default is False)
            
            – tol: Tolerance for stopping criterion. (Default is 1e-3, or 0.001)
            
            – cache_size: Specify the size of the kernel cache in MB. (Default is 200)
            
            – class_weight: Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are
            supposed to have weight one. The “balanced” mode uses the values of y to automatically adjust weights
            inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
            (Default is None)
            
            – verbose: Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in
            libsvm that, if enabled, may not work properly in a multithreaded context. (Default is False)
            
            – max_iter: Hard limit on iterations within solver, or -1 for no limit. (Default is -1)
            
            – decision_function_shape: Whether to return a one-vs-rest (‘ovr’) decision function of shape
            (n_samples, n_classes) as all other classifiers, or the original one-vs-one (‘ovo’) decision function of
            libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one (‘ovo’) is always
            used as multi-class strategy. The parameter is ignored for binary classification. (Default is "ovr")
            
            – break_ties: If true, decision_function_shape='ovr', and number of classes > 2, predict will break ties
            according to the confidence values of decision_function; otherwise the first class among the tied classes is
            returned. Please note that breaking ties comes at a relatively high computational cost compared to a simple
            predict. (Default is False)
            
            – random_state: Controls the pseudo random number generation for shuffling the data for probability
            estimates. Ignored when probability is False. Pass an int for reproducible output across multiple function
            calls. (Default is None)

            – cv: the number of folds to use for cross validation of model (defaults to None)

        The implementation is based on libsvm.
        """
        if self._check_inputs():
            # Initialize classifier
            self.classifier_nu_SVC = self._create_SVC_model(is_nu=True)
            
            # Split data, if needed; if testing/training sets are still None, call _split_data()
            if self.dataset_X_test is None:
                self._split_data()

            # Train classifier; handle exception if arguments are incorrect
            try:
                self.classifier_nu_SVC.fit(self.dataset_X_train, self.dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the NuSVC model. Check your arguments and try again.")
                print("Here is the exception message:")
                print(e)
                self.classifier_nu_SVC = None
                return

            # Evaluate metrics of model
            self.accuracy_nu_SVC = self.classifier_nu_SVC.score(self.dataset_X_test, self.dataset_y_test)
            self.cross_val_scores_nu_SVC = cross_val_score(self.classifier_nu_SVC, self.attributes, self.labels, cv=self.cv)

            self.roc_auc_nu_SVC = roc_auc_score(self.classifier_nu_SVC.predict(self.dataset_X_test),
                                                self.classifier_nu_SVC.predict_proba(self.dataset_X_test)[::, 1])
            
            # Output results
            self._output_classifier_results(model="NuSVC")

    def predict_nu_SVC(self, dataset_X=None):
        """
        Classifies each datapoint in dataset_X using the NuSVC model. Returns the predicted classification.

        predict_SVC() can only run after run_nu_SVC() has successfully trained the NuSVC model.
        """
        # Check that run_nu_SVC() has already been called
        if self.classifier_nu_SVC is None:
            print("The NuSVC model seems to be missing. Have you called run_nu_SVC() yet?")
            return
        
        # Try to make the prediction; handle exception if dataset_X isn't a valid input
        try:
            y_prediction = self.classifier_nu_SVC.predict(dataset_X)
        except Exception as e:
            print("The NuSVC model failed to run. Check your inputs and try again.")
            print("Here is the exception message:")
            print(e)
            return
        
        print("\nNuSVC predictions:\n", y_prediction, "\n")
        return y_prediction
    
    def run_linear_SVC(self):
        """
        Runs LinearSVC model.
        Parameters per scikit-learn's documentation:

            – penalty: Specifies the norm used in the penalization. The ‘l2’ penalty is the standard used in SVC. The
            ‘l1’ leads to coef_ vectors that are sparse. (Default is "l2")

            – loss: Specifies the loss function. ‘hinge’ is the standard SVM loss (used e.g. by the SVC class) while
            ‘squared_hinge’ is the square of the hinge loss. (Default is "squared_hinge")

            – dual: Select the algorithm to either solve the dual or primal optimization problem.
            Prefer dual=False when n_samples > n_features. (Default is True)
            
            – tol: Tolerance for stopping criterion. (Default is 1e-4, or 0.0001)
            
            – C: Regularization parameter. The strength of the regularization is inversely proportional to C. Must be
            strictly positive. (Default is 1.0)
            
            – multi_class: Determines the multi-class strategy if y contains more than two classes. "ovr" trains
            n_classes one-vs-rest classifiers, while "crammer_singer" optimizes a joint objective over all classes.
            While crammer_singer is interesting from a theoretical perspective as it is consistent, it is seldom used
            in practice as it rarely leads to better accuracy and is more expensive to compute. If "crammer_singer" is
            chosen, the options loss, penalty and dual will be ignored. (Default is "ovr")
            
            – fit_intercept: Whether to calculate the intercept for this model. If set to false, no intercept will be
            used in calculations (i.e. data is expected to be already centered). (Default is True)
            
            – intercept_scaling: When self.fit_intercept is True, instance vector x becomes [x, self.intercept_scaling],
            i.e. a “synthetic” feature with constant value equals to intercept_scaling is appended to the instance
            vector. The intercept becomes intercept_scaling * synthetic feature weight Note! the synthetic feature
            weight is subject to l1/l2 regularization as all other features. To lessen the effect of regularization on
            synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.
            (Default is 1)
            
            – class_weight: Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are
            supposed to have weight one. The “balanced” mode uses the values of y to automatically adjust weights
            inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
            (Default is None)
            
            – verbose: Enable verbose output. Note that this setting takes advantage of a per-process runtime setting
            in liblinear that, if enabled, may not work properly in a multithreaded context. (Default is 0)
            
            – random_state: Controls the pseudo random number generation for shuffling the data for the dual coordinate
            descent (if dual=True). When dual=False the underlying implementation of LinearSVC is not random and
            random_state has no effect on the results. Pass an int for reproducible output across multiple function
            calls. (Default is None)
            
            – max_iter: The maximum number of iterations to be run. (Default is 1000)

            – cv: the number of folds to use for cross validation of model (defaults to None)
        """
        if self._check_inputs():
            # Initialize classifier
            self.classifier_linear_SVC = self._create_linear_SVC_model()

            # Split data, if needed; if testing/training sets are still None, call _split_data()
            if self.dataset_X_test is None:
                self._split_data()

            # Train classifier; handle exception if arguments are incorrect
            try:
                self.classifier_linear_SVC.fit(self.dataset_X_train, self.dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the LinearSVC model. Check your arguments and try again.")
                print("Here is the exception message:")
                print(e)
                self.classifier_linear_SVC = None
                return

            # Evaluate metrics of model
            self.accuracy_linear_SVC = self.classifier_linear_SVC.score(self.dataset_X_test, self.dataset_y_test)
            self.cross_val_scores_linear_SVC =\
                cross_val_score(self.classifier_linear_SVC, self.attributes, self.labels, cv=self.cv)
            
            # Output results
            self._output_classifier_results(model="LinearSVC")

    def predict_linear_SVC(self, dataset_X=None):
        """
        Classifies each datapoint in dataset_X using the LinearSVC model. Returns the predicted classification.

        predict_linear_SVC() can only run after run_linear_SVC() has successfully trained the LinearSVC model.
        """
        # Check that run_linear_SVC() has already been called
        if self.classifier_linear_SVC is None:
            print("The LinearSVC model seems to be missing. Have you called run_linear_SVC() yet?")
            return
        
        # Try to make the prediction; handle exception if dataset_X isn't a valid input
        try:
            y_prediction = self.classifier_linear_SVC.predict(dataset_X)
        except Exception as e:
            print("The LinearSVC model failed to run. Check your inputs and try again.")
            print("Here is the exception message:")
            print(e)
            return
        
        print("\nLinearSVC predictions:\n", y_prediction, "\n")
        return y_prediction

    # Wrappers for SVM regression classes

    def run_SVR(self):
        """
        Runs SVR model.
        Parameters per scikit-learn's documentation:

            – kernel: Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’,
            ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is
            used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape
            (n_samples, n_samples). (Default is "rbf")
            
            – degree: Degree of the polynomial kernel function ("poly"). Ignored by all other kernels. (Default is 3)
            
            – gamma: Kernel coefficient for "rbf", "poly", and "sigmoid". If gamma="scale", then it uses
            1 / (n_features * training_samples.var()) as value of gamma. IF gamma="auto", it uses 1 / n_features.
            (Default is "scale")
            
            – coef0: Independent term in kernel function. It is only significant in "poly" and "sigmoid". (Default is 0.0)

            – tol: Tolerance for stopping criterion. (Default is 1e-3, or 0.001)

            – C: Regularization parameter. The strength of the regularization is inversely proportional to C.
            Must be strictly positive. The penalty is a squared l2 penalty. (Default is 1.0)

            – epsilon: Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is
            associated in the training loss function with points predicted within a distance epsilon from the actual
            value. (Default is 0.1)

            – shrinking: Whether to use the shrinking heuristic. (Default is True)

            – cache_size: Specify the size of the kernel cache in MB. (Default is 200)

            – verbose: Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in
            libsvm that, if enabled, may not work properly in a multithreaded context. (Default is False)

            – max_iter: Hard limit on iterations within solver, or -1 for no limit. (Default is -1)

            – cv: the number of folds to use for cross validation of model (defaults to None)
        """
        if self._check_inputs():
            # Initialize regression model
            self.regressor_SVR = self._create_SVR_model(is_nu=False)

                # Split data, if needed; if testing/training sets are still None, call _split_data()
            if self.dataset_X_test is None:
                self._split_data()

            # Train regression model; handle exception if arguments are incorrect and/or if labels isn't
            # quantitative data
            try:
                self.regressor_SVR.fit(self.dataset_X_train, self.dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the SVR model. Check you arguments and try again.")
                print("Does labels only contain quantitative data?")
                print("Here is the exception message:")
                print(e)
                self.regressor_SVR = None
                return

            # Evaluate metrics of model
            self.mean_squared_error_SVR =\
                mean_squared_error(self.dataset_y_test, self.regressor_SVR.predict(self.dataset_X_test))
            self.r2_score_SVR = self.regressor_SVR.score(self.dataset_X_test, self.dataset_y_test)
            self.r_score_SVR = sqrt(self.r2_score_SVR)
            self.cross_val_scores_SVR = cross_val_score(self.regressor_SVR, self.attributes, self.labels, cv=self.cv)

            # Output results
            self._output_regressor_results(model="SVR")

    def predict_SVR(self, dataset_X=None):
        """
        Classifies each datapoint in dataset_X using the SVR model. Returns the predicted classification.

        predict_SVR() can only run after run_SVR() has successfully trained the SVR model.
        """
        # Check that run_SVR() has already been called
        if self.regressor_SVR is None:
            print("The SVR model seems to be missing. Have you called run_SVR() yet?")
            return
        
        # Try to make the prediction; handle exception if dataset_X isn't a valid input
        try:
            y_prediction = self.regressor_SVR.predict(dataset_X)
        except Exception as e:
            print("The SVR model failed to run. Check your inputs and try again.")
            print("Here is the exception message:")
            print(e)
            return
        
        print("\nSVR predictions:\n", y_prediction, "\n")
        return y_prediction

    def run_nu_SVR(self):
        """
        Runs NuSVR model.
        Parameters per scikit-learn's documentation:

            – nu: An upper bound on the fraction of margin errors and a lower bound of the fraction of support vectors.
            Should be in the interval (0, 1]. (Default is 0.5)
            
            – C: Regularization parameter. The strength of the regularization is inversely proportional to C.
            Must be strictly positive. The penalty is a squared l2 penalty. (Default is 1.0)

            – kernel: Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’,
            ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is
            used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape
            (n_samples, n_samples). (Default is "rbf")
            
            – degree: Degree of the polynomial kernel function ("poly"). Ignored by all other kernels. (Default is 3)
            
            – gamma: Kernel coefficient for "rbf", "poly", and "sigmoid". If gamma="scale", then it uses
            1 / (n_features * training_samples.var()) as value of gamma. IF gamma="auto", it uses 1 / n_features.
            (Default is "scale")
            
            – coef0: Independent term in kernel function. It is only significant in "poly" and "sigmoid". (Default is 0.0)
            
            – shrinking: Whether to use the shrinking heuristic. (Default is True)
                        
            – tol: Tolerance for stopping criterion. (Default is 1e-3, or 0.001)
            
            – cache_size: Specify the size of the kernel cache in MB. (Default is 200)
            
            – verbose: Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in
            libsvm that, if enabled, may not work properly in a multithreaded context. (Default is False)
            
            – max_iter: Hard limit on iterations within solver, or -1 for no limit. (Default is -1)

            – cv: the number of folds to use for cross validation of model (defaults to None)
        """
        if self._check_inputs():
            # Initialize regression model
            self.regressor_nu_SVR = self._create_SVR_model(is_nu=True)
            
            # Split data, if needed; if testing/training sets are still None, call _split_data()
            if self.dataset_X_test is None:
                self._split_data()

            # Train regression model; handle exception if arguments are incorrect and/or if labels isn't
            # quantitative data
            try:
                self.regressor_nu_SVR.fit(self.dataset_X_train, self.dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the NuSVR model. Check you arguments and try again.")
                print("Does labels only contain quantitative data?")
                print("Here is the exception message:")
                print(e)
                self.regressor_nu_SVR = None
                return

            # Evaluate metrics of model
            self.mean_squared_error_nu_SVR =\
                mean_squared_error(self.dataset_y_test, self.regressor_nu_SVR.predict(self.dataset_X_test))
            self.r2_score_nu_SVR = self.regressor_nu_SVR.score(self.dataset_X_test, self.dataset_y_test)
            self.r_score_nu_SVR = sqrt(self.r2_score_nu_SVR)
            self.cross_val_scores_nu_SVR = cross_val_score(self.regressor_nu_SVR, self.attributes, self.labels, cv=self.cv)

            # Output results
            self._output_regressor_results(model="NuSVR")

    def predict_nu_SVR(self, dataset_X=None):
        """
        Classifies each datapoint in dataset_X using the NuSVR model. Returns the predicted classification.

        predict_nu_SVR() can only run after run_nu_SVR() has successfully trained the NuSVR model.
        """
        # Check that run_nu_SVR() has already been called
        if self.regressor_nu_SVR is None:
            print("The NuSVR model seems to be missing. Have you called run_nu_SVR() yet?")
            return
        
        # Try to make the prediction; handle exception if dataset_X isn't a valid input
        try:
            y_prediction = self.regressor_nu_SVR.predict(dataset_X)
        except Exception as e:
            print("The NuSVR model failed to run. Check your inputs and try again.")
            print("Here is the exception message:")
            print(e)
            return
        
        print("\nNuSVR predictions:\n", y_prediction, "\n")
        return y_prediction

    def run_linear_SVR(self):
        """
        Runs LinearSVR model.
        Parameters per scikit-learn's documentation:

            – epsilon: Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is
            associated in the training loss function with points predicted within a distance epsilon from the actual
            value. (Default is 0.1)

            – tol: Tolerance for stopping criterion. (Default is 1e-3, or 0.001)

            – C: Regularization parameter. The strength of the regularization is inversely proportional to C.
            Must be strictly positive. The penalty is a squared l2 penalty. (Default is 1.0)

            – loss: Specifies the loss function. The epsilon-insensitive loss (standard SVR) is the L1 loss, while the
            squared epsilon-insensitive loss (‘squared_epsilon_insensitive’) is the L2 loss.
            (Default is "epsilon_insensitive")

            – fit_intercept: Whether to calculate the intercept for this model. If set to false, no intercept will be
            used in calculations (i.e. data is expected to be already centered). (Default is True)
            
            – intercept_scaling: When self.fit_intercept is True, instance vector x becomes [x, self.intercept_scaling],
            i.e. a “synthetic” feature with constant value equals to intercept_scaling is appended to the instance
            vector. The intercept becomes intercept_scaling * synthetic feature weight Note! the synthetic feature
            weight is subject to l1/l2 regularization as all other features. To lessen the effect of regularization on
            synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.
            (Default is 1)

            – dual: Select the algorithm to either solve the dual or primal optimization problem.
            Prefer dual=False when n_samples > n_features. (Default is True)

            – verbose: Enable verbose output. Note that this setting takes advantage of a per-process runtime setting
            in liblinear that, if enabled, may not work properly in a multithreaded context. (Default is 0)
            
            – random_state: Controls the pseudo random number generation for shuffling the data for the dual coordinate
            descent (if dual=True). When dual=False the underlying implementation of LinearSVC is not random and
            random_state has no effect on the results. Pass an int for reproducible output across multiple function
            calls. (Default is None)
            
            – max_iter: The maximum number of iterations to be run. (Default is 1000)

            – cv: the number of folds to use for cross validation of model (defaults to None)
        """
        if self._check_inputs():
            # Initialize regression model
            self.regressor_linear_SVR = self._create_linear_SVR_model()

            # Split data, if needed; if testing/training sets are still None, call _split_data()
            if self.dataset_X_test is None:
                self._split_data()

            # Train regression model; handle exception if arguments are incorrect and/or labels isn't
            # quantitative data
            try:
                self.regressor_linear_SVR.fit(self.dataset_X_train, self.dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the LinearSVR model. Check you arguments and try again.")
                print("Does labels only contain quantitative data?")
                print("Here is the exception message:")
                print(e)
                self.regressor_linear_SVR = None
                return
            
            # Evaluate metrics of model
            self.mean_squared_error_linear_SVR =\
                mean_squared_error(self.dataset_y_test, self.regressor_linear_SVR.predict(self.dataset_X_test))
            self.r2_score_linear_SVR = self.regressor_linear_SVR.score(self.dataset_X_test, self.dataset_y_test)
            self.r_score_linear_SVR = sqrt(self.r2_score_linear_SVR)
            self.cross_val_scores_linear_SVR =\
                cross_val_score(self.regressor_linear_SVR, self.attributes, self.labels, cv=self.cv)
            
            # Output results
            self._output_regressor_results(model="LinearSVR")

    def predict_linear_SVR(self, dataset_X=None):
        """
        Classifies each datapoint in dataset_X using the LinearSVR model. Returns the predicted classification.

        predict_linear_SVR() can only run after run_linear_SVR() has successfully trained the LinearSVR model.
        """
        # Check that run_linear_SVR() has already been called
        if self.regressor_linear_SVR is None:
            print("The LinearSVR model seems to be missing. Have you called run_linear_SVR() yet?")
            return
        
        # Try to make the prediction; handle exception if dataset_X isn't a valid input
        try:
            y_prediction = self.regressor_linear_SVR.predict(dataset_X)
        except Exception as e:
            print("The LinearSVR model failed to run. Check your inputs and try again.")
            print("Here is the exception message:")
            print(e)
            return
        
        print("\nLinearSVR predictions:\n", y_prediction, "\n")
        return y_prediction

    # Helper methods

    def _create_SVC_model(self, is_nu):
        """
        Runs UI for getting parameters and creating SVC or NuSVC model.
        """
        if is_nu:
            print("\n==============================")
            print("= Parameter inputs for NuSVC =")
            print("==============================\n")
        else:
            print("\n============================")
            print("= Parameter inputs for SVC =")
            print("============================\n")
        
        if input("Use default parameters (Y/n)? ").lower() != "n":
            self.test_size = 0.25
            self.cv = None
            print("\n=======================================================")
            print("= End of parameter inputs; press any key to continue. =")
            input("=======================================================\n")

            if is_nu:
                return NuSVC(probability=True)
            else:
                return SVC(probability=True)
        
        print("\nIf you are unsure about a parameter, press enter to use its default value.")
        print("Invalid parameter inputs will be replaced with their default values.")
        print("If you finish entering parameters early, enter 'q' to skip ahead.\n")

        # Set defaults
        if is_nu:
            nu = 0.5
        else:
            C = 1.0
        
        self.test_size = 0.25
        self.cv = None
        kernel = "rbf"
        degree = 3
        gamma = "scale"
        coef0 = 0.0
        shrinking = True
        probability = True
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
            
            if is_nu:
                user_input = input("Enter a decimal for nu: ")
                try:
                    nu = float(user_input)
                except:
                    if user_input.lower() == "q":
                        break
            else:
                user_input = input("Enter the regularization parameter: ")
                try:
                    C = float(user_input)
                except:
                    if user_input.lower() == "q":
                        break
            
            print("Which kernel type should be used?")
            user_input =\
                input("Enter 1 for 'linear', 2 for 'poly', 3 for 'rbf', 4 for 'sigmoid', or 5 for 'precomputed': ")
            
            if user_input.lower() == "q":
                break
            elif user_input == "1":
                kernel = "linear"
            elif user_input == "2":
                kernel = "poly"
            elif user_input == "4":
                kernel = "sigmoid"
            elif user_input == "5":
                kernel = "recomputed"
            
            if kernel == "poly":
                user_input = input("Enter the degree of the kernel function: ")
                try:
                    degree = int(user_input)
                except:
                    if user_input.lower() == "q":
                        break
            
            if kernel == "rbf" or kernel == "poly" or kernel == "sigmoid":
                print("Set the kernel coefficient.")
                user_input = input("Enter 1 for 'scale', or 2 for 'auto': ")
                if user_input.lower() == "q":
                    break
                elif user_input == "2":
                    gamma = "auto"
            
            user_input = input("Enter the independent term in the kernel function: ")

            try:
                coef0 = float(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Use the shrinking heuristic (Y/n)? ").lower()

            if user_input == "q":
                break
            elif user_input == "n":
                shrinking = False
            
            user_input = input("Enter the tolerance for stopping criterion: ")

            try:
                tol = float(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Enter the kernel cache size in MB: ")

            try:
                cache_size = float(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Automatically adjust class weights (y/N)? ").lower()

            if user_input == "q":
                break
            elif user_input == "y":
                class_weight = "balanced"
            
            user_input = input("Enter the maximum number of iterations, or press enter for no limit: ")

            try:
                max_iter = int(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            print("Set the decision function.")
            user_input = input("Enter 1 for one-vs-rest, or 2 for one-vs-one: ")

            if user_input.lower() == "q":
                break
            elif user_input == "2":
                decision_function_shape = "ovo"
            
            user_input = input("Enable tie-breaking (y/N)? ").lower()

            if user_input == "q":
                break
            elif user_input == "y":
                break_ties = True
            
            user_input = input("Enter a seed for the random number generator: ")

            try:
                random_state = int(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Enable verbose logging (y/N)? ").lower()

            if user_input == "q":
                break
            elif user_input == "y":
                verbose = True
            
            break

        print("\n=======================================================")
        print("= End of parameter inputs; press any key to continue. =")
        input("=======================================================\n")

        if is_nu:
            return NuSVC(nu=nu, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking,
                         probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight,
                         verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape,
                         break_ties=break_ties, random_state=random_state)
        else:
            return SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking,
                       probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight,
                       verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape,
                       break_ties=break_ties, random_state=random_state)

    def _create_linear_SVC_model(self):
        """
        Runs UI for getting parameters and creating LinearSVC model.
        """
        print("\n==================================")
        print("= Parameter inputs for LinearSVC =")
        print("==================================\n")

        if input("Use default parameters (Y/n)? ").lower() != "n":
            self.test_size = 0.25
            self.cv = None
            print("\n=======================================================")
            print("= End of parameter inputs; press any key to continue. =")
            input("=======================================================\n")
            return LinearSVC()
        
        print("\nIf you are unsure about a parameter, press enter to use its default value.")
        print("Invalid parameter inputs will be replaced with their default values.")
        print("If you finish entering parameters early, enter 'q' to skip ahead.\n")

        # Set defaults
        self.test_size = 0.25
        self.cv = None
        penalty = "l2"
        loss = "squared_hinge"
        dual = True
        tol = 0.0001
        C = 1.0
        multi_class = 'ovr'
        fit_intercept = True
        intercept_scaling = 1
        class_weight = None
        random_state = None
        max_iter = 1000
        verbose = 0

        # Get user parameter input
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

            user_input = input("Calculate a y-intercept (Y/n)? ").lower()

            if user_input == "q":
                break
            elif user_input == "n":
                fit_intercept = False
            
            if fit_intercept:
                user_input = input("Enter a value for intercept scaling: ")
                try:
                    intercept_scaling = float(user_input)
                except:
                    if user_input.lower() == "q":
                        break
            
            print("Set the norm used in penalization.")
            user_input = input("Enter 1 for 'l1', or 2 for 'l2': ")

            if user_input.lower() == "q":
                break
            elif user_input == "1":
                penalty = "l1"
            
            print("Choose a loss function.")
            user_input = input("Enter 1 for 'hinge', or 2 for 'squared_hinge': ")

            if user_input.lower() == "q":
                break
            elif user_input == "1":
                loss = "hinge"
            
            print("Should the algorithm solve the duel or primal optimization problem?")
            user_input = input("Enter 1 for dual, or 2 for primal: ")

            if user_input.lower() == "q":
                break
            elif user_input == "2":
                dual = False
            
            user_input = input("Enter the tolerance for stopping criterion: ")

            try:
                tol = float(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Enter the regularization parameter: ")
            
            try:
                C = float(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            print("Set the multi-class strategy if there are more than two classes.")
            user_input = input("Enter 1 for one-vs-rest, or 2 for Crammer-Singer: ")

            if user_input.lower() == "q":
                break
            elif user_input == "2":
                multi_class = "crammer_singer"
            
            user_input = input("Automatically adjust class weights (y/N)? ").lower()

            if user_input == "q":
                break
            elif user_input == "y":
                class_weight = "balanced"
            
            user_input = input("Enter the maximum number of iterations: ")

            try:
                max_iter = int(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Enter a seed for the random number generator: ")

            try:
                random_state = int(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Enable verbose logging (y/N)? ").lower()

            if user_input == "y":
                verbose = 1
            
            break
        
        print("\n=======================================================")
        print("= End of parameter inputs; press any key to continue. =")
        input("=======================================================\n")
        
        return LinearSVC(penalty=penalty, loss=loss, dual=dual, tol=tol, C=C, multi_class=multi_class,
                         fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                         verbose=verbose, random_state=random_state, max_iter=max_iter)

    def _create_SVR_model(self, is_nu):
        """
        Runs UI for getting parameters and creates SVR or NuSVR model.
        """
        if is_nu:
            print("\n==============================")
            print("= Parameter inputs for NuSVR =")
            print("==============================\n")
        else:
            print("\n============================")
            print("= Parameter inputs for SVR =")
            print("============================\n")
        
        if input("Use default parameters (Y/n)? ").lower() != "n":
            self.test_size = 0.25
            self.cv = None
            print("\n=======================================================")
            print("= End of parameter inputs; press any key to continue. =")
            input("=======================================================\n")

            if is_nu:
                return NuSVR()
            else:
                return SVR()
        
        print("\nIf you are unsure about a parameter, press enter to use its default value.")
        print("Invalid parameter inputs will be replaced with their default values.")
        print("If you finish entering parameters early, enter 'q' to skip ahead.\n")

        # Set defaults
        if is_nu:
            nu = 0.5
        else:
            epsilon = 0.1
        
        self.test_size = 0.25
        self.cv = None
        kernel = "rbf"
        degree = 3
        gamma = 'scale'
        coef0 = 0.0
        tol = 0.001
        C = 1.0
        shrinking = True
        cache_size = 200
        verbose = False
        max_iter=-1

        # Get user parameter input
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
            
            if is_nu:
                user_input = input("Enter a decimal for nu: ")
                try:
                    nu = float(user_input)
                except:
                    if user_input.lower() == "q":
                        break
            else:
                user_input = input("Enter epsilon, the range from an actual value where penalties aren't applied: ")
                try:
                    epsilon = float(user_input)
                except:
                    if user_input.lower() == "q":
                        break
            
            print("Which kernel type should be used?")
            user_input =\
                input("Enter 1 for 'linear', 2 for 'poly', 3 for 'rbf', 4 for 'sigmoid', or 5 for 'precomputed': ")
            
            if user_input.lower() == "q":
                break
            elif user_input == "1":
                kernel = "linear"
            elif user_input == "2":
                kernel = "poly"
            elif user_input == "4":
                kernel = "sigmoid"
            elif user_input == "5":
                kernel = "recomputed"
            
            if kernel == "poly":
                user_input = input("Enter the degree of the kernel function: ")
                try:
                    degree = int(user_input)
                except:
                    if user_input.lower() == "q":
                        break
            
            if kernel == "rbf" or kernel == "poly" or kernel == "sigmoid":
                print("Set the kernel coefficient.")
                user_input = input("Enter 1 for 'scale', or 2 for 'auto': ")
                if user_input.lower() == "q":
                    break
                elif user_input == "2":
                    gamma = "auto"
            
            user_input = input("Enter the regularization parameter: ")
            
            try:
                C = float(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Enter the independent term in the kernel function: ")

            try:
                coef0 = float(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Enter the tolerance for stopping criterion: ")

            try:
                tol = float(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Use the shrinking heuristic (Y/n)? ").lower()

            if user_input == "q":
                break
            elif user_input == "n":
                shrinking = False
            
            user_input = input("Enter the kernel cache size in MB: ")

            try:
                cache_size = float(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Enter the maximum number of iterations, or press enter for no limit: ")

            try:
                max_iter = int(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Enable verbose logging (y/N)? ").lower()

            if user_input == "q":
                break
            elif user_input == "y":
                verbose = True
            
            break
        
        print("\n=======================================================")
        print("= End of parameter inputs; press any key to continue. =")
        input("=======================================================\n")

        if is_nu:
            return NuSVR(nu=nu, C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking,
                         tol=tol, cache_size=cache_size, verbose=verbose, max_iter=max_iter)
        else:
            return SVR(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, C=C, epsilon=epsilon,
                       shrinking=shrinking, cache_size=cache_size, verbose=verbose, max_iter=max_iter)

    def _create_linear_SVR_model(self):
        """
        Runs UI for getting parameters and creates LinearSVR model.
        """
        print("\n==================================")
        print("= Parameter inputs for LinearSVR =")
        print("==================================\n")

        if input("Use default parameters (Y/n)? ").lower() != "n":
            self.test_size = 0.25
            self.cv = None
            print("\n=======================================================")
            print("= End of parameter inputs; press any key to continue. =")
            input("=======================================================\n")
            return LinearSVR()
        
        print("\nIf you are unsure about a parameter, press enter to use its default value.")
        print("Invalid parameter inputs will be replaced with their default values.")
        print("If you finish entering parameters early, enter 'q' to skip ahead.\n")

        # Set defaults
        self.test_size = 0.25
        self.cv = None
        epsilon = 0.0
        tol = 0.0001
        C = 1.0
        loss = 'epsilon_insensitive'
        fit_intercept = True
        intercept_scaling = 1.0
        dual = True
        random_state = None
        max_iter = 1000
        verbose = 0

        # Get user parameter input
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
            
            user_input = input("Enter epsilon, the range from an actual value where penalties aren't applied: ")
            
            try:
                epsilon = float(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Enter the tolerance for stopping criterion: ")

            try:
                tol = float(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Enter the regularization parameter: ")
            
            try:
                C = float(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            print("Choose a loss function.")
            user_input = input("Enter 1 for 'epsilon_insensitive', or 2 for 'squared_epsilon_insensitive': ")

            if user_input.lower() == "q":
                break
            elif user_input == "2":
                loss = "squared_epsilon_insensitive"
            
            user_input = input("Calculate a y-intercept (Y/n)? ").lower()

            if user_input == "q":
                break
            elif user_input == "n":
                fit_intercept = False
            
            if fit_intercept:
                user_input = input("Enter a value for intercept scaling: ")
                try:
                    intercept_scaling = float(user_input)
                except:
                    if user_input.lower() == "q":
                        break
            
            print("Should the algorithm solve the duel or primal optimization problem?")
            user_input = input("Enter 1 for dual, or 2 for primal: ")

            if user_input.lower() == "q":
                break
            elif user_input == "2":
                dual = False
            
            user_input = input("Enter the maximum number of iterations: ")

            try:
                max_iter = int(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Enter a seed for the random number generator: ")

            try:
                random_state = int(user_input)
            except:
                if user_input.lower() == "q":
                    break
            
            user_input = input("Enable verbose logging (y/N)? ").lower()

            if user_input == "y":
                verbose = 1
            
            break
        
        print("\n=======================================================")
        print("= End of parameter inputs; press any key to continue. =")
        input("=======================================================\n")

        return LinearSVR(epsilon=epsilon, tol=tol, C=C, loss=loss, fit_intercept=fit_intercept,
                         intercept_scaling=intercept_scaling, dual=dual, verbose=verbose, random_state=random_state,
                         max_iter=max_iter)

    def _output_classifier_results(self, model):
        """
        Outputs model metrics after a classifier model finishes running.
        """
        if model == "SVC":
            print("\n===============")
            print("= SVC Results =")
            print("===============\n")

            print("{:<20} {:<20}".format("Accuracy:", self.accuracy_SVC))

            if self.roc_auc_SVC is not None:
                print("\n{:<20} {:<20}".format("ROC AUC:", self.roc_auc_SVC))
            
            print("\nCross Validation Scores:\n", self.cross_val_scores_SVC)
            print("\n\nCall predict_SVC() to make predictions for new data.")
        elif model == "NuSVC":
            print("\n=================")
            print("= NuSVC Results =")
            print("=================\n")

            print("{:<20} {:<20}".format("Accuracy:", self.accuracy_nu_SVC))

            if self.roc_auc_nu_SVC is not None:
                print("\n{:<20} {:<20}".format("ROC AUC:", self.roc_auc_nu_SVC))
            
            print("\nCross Validation Scores:\n", self.cross_val_scores_nu_SVC)
            print("\n\nCall predict_nu_SVC() to make predictions for new data.")
        else:
            print("\n=====================")
            print("= LinearSVC Results =")
            print("=====================\n")

            print("{:<20} {:<20}".format("Accuracy:", self.accuracy_linear_SVC))            
            print("\nCross Validation Scores:\n", self.cross_val_scores_linear_SVC)
            print("\n\nCall predict_linear_SVC() to make predictions for new data.")
        
        print("\n===================")
        print("= End of results. =")
        print("===================\n")

    def _output_regressor_results(self, model):
        """
        Outputs model metrics after a regressor model finishes running.
        """
        if model == "SVR":
            print("\n===============")
            print("= SVR Results =")
            print("===============\n")

            print("{:<20} {:<20}".format("Mean Squared Error:", self.mean_squared_error_SVR))
            print("\n{:<20} {:<20}".format("R2 Score:", self.r2_score_SVR))
            print("\n{:<20} {:<20}".format("R Score:", self.r_score_SVR))
            print("\nCross Validation Scores:\n", self.cross_val_scores_SVR)
            print("\n\nCall predict_SVR() to make predictions for new data.")
        elif model == "NuSVR":
            print("\n=================")
            print("= NuSVR Results =")
            print("=================\n")

            print("{:<20} {:<20}".format("Mean Squared Error:", self.mean_squared_error_nu_SVR))
            print("{:<20} {:<20}".format("R2 Score:", self.r2_score_nu_SVR))
            print("\n{:<20} {:<20}".format("R Score:", self.r_score_nu_SVR))
            print("\nCross Validation Scores:\n", self.cross_val_scores_nu_SVR)
            print("\n\nCall predict_nu_SVR() to make predictions for new data.")
        else:
            print("\n=====================")
            print("= LinearSVR Results =")
            print("=====================\n")

            print("{:<20} {:<20}".format("Mean Squared Error:", self.mean_squared_error_linear_SVR))
            print("{:<20} {:<20}".format("R2 Score:", self.r2_score_linear_SVR))
            print("\n{:<20} {:<20}".format("R Score:", self.r_score_linear_SVR))
            print("\nCross Validation Scores:\n", self.cross_val_scores_linear_SVR)
            print("\n\nCall predict_linear_SVR() to make predictions for new data.")
        
        print("\n===================")
        print("= End of results. =")
        print("===================\n")

    def _split_data(self):
        """
        Helper method for splitting attributes and labels into training and testing sets.

        This method runs under the assumption that all relevant instance data has been checked for correctness.
        """

        self.dataset_X_train, self.dataset_X_test, self.dataset_y_train, self.dataset_y_test =\
            train_test_split(self.attributes, self.labels, test_size=self.test_size)

    def _check_inputs(self):
        """
        Verifies if instance data is ready for use in SVM model.
        """
        # Check if attributes exists
        if self.attributes is None:
            print("attributes is missing; call set_attributes(new_attributes) to fix this! new_attributes should be a",
                  "populated dataset of independent variables.")
            return False

        # Check if labels exists
        if self.labels is None:
            print("labels is missing; call set_labels(new_labels) to fix this! new_labels should be a populated dataset",
                  "of classes.")
            return False

        # Check if attributes and labels have same number of rows (samples)
        if self.attributes.shape[0] != self.labels.shape[0]:
            print("attributes and labels don't have the same number of rows. Make sure the number of samples in each",
                  "dataset matches!")
            return False

        return True