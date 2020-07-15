from math import sqrt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, NuSVC, LinearSVC, SVR, NuSVR, LinearSVR

class SVM:
    """
    Wrapper class around scikit-learn's support vector machine functionality.
    This class supports binary and multi-class classification on a dataset, along with regression via Support Vector
    Regression (SVR).
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

    def __init__(self, attributes=None, labels=None, test_size=0.25):
        """
        Initializes a SVM object.

        The following parameters are needed to use a SVM:

            – attributes: a numpy array of the independent variables
            – labels: a numpy array of the classes (for classification) or dependent variables (for regression)
            – test_size: the proportion of the dataset to be used for testing the model (defaults to 0.25);
            the proportion of the dataset to be used for training will be the complement of test_size

        After successfully running one of the classifier methods (SVC(), nu_SVC(), or linear_SVC()), the corresponding
        classifier below will be trained:

            – classifier_SVC: a classifier trained using scikit-learn's SVC implementation
            – accuracy_SVC: the accuracy of the SVC model, based on its predictions for dataset_X_test
            – roc_auc_SVC: the area under the ROC curve for the SVC model
            – classifier_nu_SVC: a classifier trained using scikit-learn's NuSVC implementation
            – accuracy_nu_SVC: the accuracy of the NuSVC model, based on its predictions for dataset_X_test
            – roc_auc_nu_SVC: the area under the ROC curve for the NuSVC model
            – classifier_linear_SVC: a classifier trained using scikit-learn's LinearSVC implementation
            – accuracy_linear_SVC: the accuracy of the LinearSVC model, based on its predictions for dataset_X_test

        After successfully running one of the regression methods (SVR(), nu_SVR(), or linear_SVR()), the corresponding
        regression model below will be trained:

            – regression_SVR: a regression model trained using scikit-learn's SVR implementation
            – r2_score_SVR: the coefficient of determination for the SVR model
            – r_score_SVR: the correlation coefficient for the SVR model
            – regression_nu_SVR: a regression model trained using scikit-learn's NuSVR implementation
            – r2_score_nu_SVR: the coefficient of determination for the NuSVR model
            – r_score_nu_SVR: the correlation coefficient for the NuSVR model
            – regression_linear_SVR: a regression model trained using scikit-learn's LinearSVR implementation
            – r2_score_linear_SVR: the coefficient of determination for the LinearSVR model
            – r_score_linear_SVR: the correlation coefficient for the LinearSVR model
        """
        self.attributes = attributes
        self.labels = labels
        self.test_size = 0.25

        self.classifier_SVC = None
        self.accuracy_SVC = None
        self.roc_auc_SVC = None
        self.classifier_nu_SVC = None
        self.accuracy_nu_SVC = None
        self.roc_auc_nu_SVC = None
        self.classifier_linear_SVC = None
        self.accuracy_linear_SVC = None

        self.regression_SVR = None
        self.r2_score_SVR = None
        self.r_score_SVR = None
        self.regression_nu_SVR = None
        self.r2_score_nu_SVR = None
        self.r_score_nu_SVR = None
        self.regression_linear_SVR = None
        self.r2_score_linear_SVR = None
        self.r_score_linear_SVR = None

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

    def get_test_size(self):
        """
        Accessor method for test_size.

        Should return a number or None.
        """
        return self.test_size

    def get_classifier_SVC(self):
        """
        Accessor method for classifier_SVC.

        Will return None if SVC() hasn't successfully run, yet.
        """
        return self.classifier_SVC

    def get_accuracy_SVC(self):
        """
        Accessor method for accuracy_SVC.

        Will return None if SVC() hasn't successfully run, yet.
        """
        return self.accuracy_SVC

    def get_roc_auc_SVC(self):
        """
        Accessor method for roc_auc_SVC.

        Will return None if SVC() hasn't successfully run, yet.
        """
        return self.roc_auc_SVC

    def get_classifier_nu_SVC(self):
        """
        Accessor method for classifier_nu_SVC.

        Will return None if nu_SVC() hasn't successfully run, yet.
        """
        return self.classifier_nu_SVC

    def get_accuracy_nu_SVC(self):
        """
        Accessor method for accuracy_nu_SVC.

        Will return None if nu_SVC() hasn't successfully run, yet.
        """
        return self.accuracy_nu_SVC

    def get_roc_auc_nu_SVC(self):
        """
        Accessor method for roc_auc_nu_SVC.

        Will return None if nu_SVC() hasn't successfully run, yet.
        """
        return self.roc_auc_nu_SVC

    def get_classifier_linear_SVC(self):
        """
        Accessor method for classifier_linear_SVC.

        Will return None if linear_SVC() hasn't successfully run, yet.
        """
        return self.classifier_linear_SVC

    def get_accuracy_linear_SVC(self):
        """
        Accessor method for accuracy_linear_SVC.

        Will return None if linear_SVC() hasn't successfully run, yet.
        """
        return self.accuracy_linear_SVC

    def get_regression_SVR(self):
        """
        Accessor method for regression_SVR.

        Will return None if SVR() hasn't successfully run, yet.
        """
        return self.regression_SVR

    def get_r2_score_SVR(self):
        """
        Accessor method for r2_score_SVR.

        Will return None if SVR() hasn't successfully run, yet.
        """
        return self.r2_score_SVR

    def get_r_score_SVR(self):
        """
        Accessor method for r_score_SVR.

        Will return None if SVR() hasn't successfully run, yet.
        """
        return self.r_score_SVR

    def get_regression_nu_SVR(self):
        """
        Accessor method for regression_nu_SVR.

        Will return None if nu_SVR() hasn't successfully run, yet.
        """
        return self.regression_nu_SVR

    def get_r2_score_nu_SVR(self):
        """
        Accessor method for r2_score_nu_SVR.

        Will return None if nu_SVR() hasn't successfully run, yet.
        """
        return self.r2_score_nu_SVR

    def get_r_score_nu_SVR(self):
        """
        Accessor method for r_score_nu_SVR.

        Will return None if nu_SVR() hasn't successfully run, yet.
        """
        return self.r_score_nu_SVR

    def get_regression_linear_SVR(self):
        """
        Accessor method for regression_linear_SVR.

        Will return None if linear_SVR() hasn't successfully run, yet.
        """
        return self.regression_linear_SVR

    def get_r2_score_linear_SVR(self):
        """
        Accessor method for r2_score_linear_SVR.

        Will return None if linear_SVR() hasn't successfully run, yet.
        """
        return self.r2_score_linear_SVR

    def get_r_score_linear_SVR(self):
        """
        Accessor method for r_score_linear_SVR.

        Will return None if linear_SVR() hasn't successfully run, yet.
        """
        return self.r_score_linear_SVR

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

    def set_test_size(self, new_test_size=0.25):
        """
        Modifier method for test_size.

        Input should be a float between 0.0 and 1.0 or None. Defaults to 0.25. The training size will be set to the
        complement of test_size.
        """
        self.test_size = new_test_size

    # Wrappers for SVM classification classes

    def SVC(self, C=1.0, kernel="rbf", degree=3, gamma="scale", coef0=0.0, shrinking=True, probability=False, tol=0.001,
            cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape="ovr",
            break_ties=False, random_state=None):
        """
        Wrapper for scikit-learn's C-Support Vector Classification implementation.
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

        The implementation is based on libsvm. The fit time scales at least quadratically with the number of samples
        and may be impractical beyond tens of thousands of samples.
        """
        if self._check_inputs():
            # Initialize classifier
            self.classifier_SVC =\
                SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking,
                    probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight, verbose=verbose,
                    max_iter=max_iter, decision_function_shape=decision_function_shape, break_ties=break_ties,
                    random_state=random_state)

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

            # Evaluate accuracy and ROC-AUC of model using testing set and actual classification
            self.accuracy_SVC = self.classifier_SVC.score(self.dataset_X_test, self.dataset_y_test)

            if probability:
                self.roc_auc_SVC = roc_auc_score(self.classifier_SVC.predict(self.dataset_X_test),
                                                 self.classifier_SVC.predict_proba(self.dataset_X_test)[::, 1])

    def nu_SVC(self, nu=0.5, kernel="rbf", degree=3, gamma="scale", coef0=0.0, shrinking=True, probability=False,
               tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape="ovr",
               break_ties=False, random_state=None):
        """
        Wrapper for scikit-learn's Nu-Support Vector Classification implementation.
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

        The implementation is based on libsvm.
        """
        if self._check_inputs():
            # Initialize classifier
            self.classifier_nu_SVC =\
                NuSVC(nu=nu, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking,
                      probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight,
                      verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape,
                      break_ties=break_ties, random_state=random_state)
            
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

            # Evaluate accuracy and ROC-AUC of model using testing set and actual classification
            self.accuracy_nu_SVC = self.classifier_nu_SVC.score(self.dataset_X_test, self.dataset_y_test)

            if probability:
                self.roc_auc_nu_SVC = roc_auc_score(self.classifier_nu_SVC.predict(self.dataset_X_test),
                                                    self.classifier_nu_SVC.predict_proba(self.dataset_X_test)[::, 1])

    def linear_SVC(self, penalty="l2", loss="squared_hinge", dual=True, tol=0.0001, C=1.0, multi_class='ovr',
                   fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None,
                   max_iter=1000):
        """
        Wrapper for scikit-learn's Linear Support Vector Classification implementation. Per scikit-learn's documentation,
        LinearSVC is similar to SVC with a linear kernel, but implemented with liblinear instead of libsvm, providing
        more flexibility in choice of penalties and loss functions. LinearSVC should also scale better to large sample
        sizes. LinearSVC supports both dense and sparse input, and the multiclass support is handled according to a
        one-vs-the-rest scheme.
        Parameters per scikit-learn's documentation:

            – penalty: Specifies the norm used in the penalization. The ‘l2’ penalty is the standard used in SVC. The
            ‘l1’ leads to coef_ vectors that are sparse. (Default is "l2")

            – loss: Specifies the loss function. ‘hinge’ is the standard SVM loss (used e.g. by the SVC class) while
            ‘squared_hinge’ is the square of the hinge loss. (Default is "squared_hinge")

            – dual: Select the algorithm to either solve the dual or primal optimization problem.
            Prefer dual=False when n_samples > n_features. (Default is True)
            
            – tol: Tolerance for stopping criteria. (Default is 1e-4, or 0.0001)
            
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
        """
        if self._check_inputs():
            # Initialize classifier
            self.classifier_linear_SVC =\
                LinearSVC(penalty=penalty, loss=loss, dual=dual, tol=tol, C=C, multi_class=multi_class,
                          fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                          verbose=verbose, random_state=random_state, max_iter=max_iter)

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

            # Evaluate accuracy of model using testing set and actual classification
            self.accuracy_linear_SVC = self.classifier_linear_SVC.score(self.dataset_X_test, self.dataset_y_test)

    # Wrappers for SVM regression classes

    def SVR(self, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True,
            cache_size=200, verbose=False, max_iter=-1):
        """
        Wrapper for scikit-learn's Epsilon-Support Vector Regression implementation. Per scikit-learn's documentation,
        this implementation is based on libsvm. Scaling to tens of thousands of samples is difficult, as the fit time
        complexity is more than quadratic with the number of samples. For large datasets, consider using LinearSVR by
        calling linear_SVR().
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
        """
        if self._check_inputs():
            # Initialize regression model
            self.regression_SVR =\
                SVR(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, C=C, epsilon=epsilon,
                    shrinking=shrinking, cache_size=cache_size, verbose=verbose, max_iter=max_iter)

                # Split data, if needed; if testing/training sets are still None, call _split_data()
            if self.dataset_X_test is None:
                self._split_data()

            # Train regression model; handle exception if arguments are incorrect and/or if labels isn't
            # quantitative data
            try:
                self.regression_SVR.fit(self.dataset_X_train, self.dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the SVR model. Check you arguments and try again.")
                print("Does labels only contain quantitative data?")
                print("Here is the exception message:")
                print(e)
                self.regression_SVR = None
                return

            # Get coefficient of determination for model
            self.r2_score_SVR = self.regression_SVR.score(self.dataset_X_test, self.dataset_y_test)
            self.r_score_SVR = sqrt(self.r2_score_SVR)

    def nu_SVR(self, nu=0.5, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=0.001,
               cache_size=200, verbose=False, max_iter=-1):
        """
        Wrapper for scikit-learn's Nu Support Vector Regression implementation. Per scikit-learn's documentation,
        NuSVR uses the parameter nu to control the number of support vectors, similar to NuSVC. Yet unlike NuSVC,
        nu replaces the parameter epsilon of epsilon-SVR, not C. This implementation is based on libsvm.
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
        """
        if self._check_inputs():
            # Initialize regression model
            self.regression_nu_SVR =\
                NuSVR(nu=nu, C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking, tol=tol,
                      cache_size=cache_size, verbose=verbose, max_iter=max_iter)
            
            # Split data, if needed; if testing/training sets are still None, call _split_data()
            if self.dataset_X_test is None:
                self._split_data()

            # Train regression model; handle exception if arguments are incorrect and/or if labels isn't
            # quantitative data
            try:
                self.regression_nu_SVR.fit(self.dataset_X_train, self.dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the NuSVR model. Check you arguments and try again.")
                print("Does labels only contain quantitative data?")
                print("Here is the exception message:")
                print(e)
                self.regression_nu_SVR = None
                return

            # Get coefficient of determination for model
            self.r2_score_nu_SVR = self.regression_nu_SVR.score(self.dataset_X_test, self.dataset_y_test)
            self.r_score_nu_SVR = sqrt(self.r2_score_nu_SVR)

    def linear_SVR(self, epsilon=0.0, tol=0.0001, C=1.0, loss='epsilon_insensitive', fit_intercept=True,
                   intercept_scaling=1.0, dual=True, verbose=0, random_state=None, max_iter=1000):
        """
        Wrapper for scikit-learn's Linear Support Vector Regression implementation. Per scikit-learn's documentation,
        LinearSVR is similar to SVR with a linear kernel, but is implemented with liblinear instead of libsvm. This
        provides greater flexibility in choice of penalties and loss functions, and should scale better to large sample
        sizes. LinearSVM supports both dense and sparse input.
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
        """
        if self._check_inputs():
            # Initialize regression model
            self.regression_linear_SVR =\
                LinearSVR(epsilon=epsilon, tol=tol, C=C, loss=loss, fit_intercept=fit_intercept,
                          intercept_scaling=intercept_scaling, dual=dual, verbose=verbose, random_state=random_state,
                          max_iter=max_iter)

            # Split data, if needed; if testing/training sets are still None, call _split_data()
            if self.dataset_X_test is None:
                self._split_data()

            # Train regression model; handle exception if arguments are incorrect and/or labels isn't
            # quantitative data
            try:
                self.regression_linear_SVR.fit(self.dataset_X_train, self.dataset_y_train)
            except Exception as e:
                print("An exception occurred while training the LinearSVR model. Check you arguments and try again.")
                print("Does labels only contain quantitative data?")
                print("Here is the exception message:")
                print(e)
                self.regression_linear_SVR = None
                return
            
            # Get coefficient of determination and correlation coefficient for model
            self.r2_score_linear_SVR = self.regression_linear_SVR.score(self.dataset_X_test, self.dataset_y_test)
            self.r_score_linear_SVR = sqrt(self.r2_score_linear_SVR)

    # Helper methods

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

        # Check if test_size is a number
        if self.test_size is not None and not isinstance(self.test_size, (int, float)):
            print("test_size must be None or a number; call set_test_size(new_test_size) to fix this!")
            return False

        return True