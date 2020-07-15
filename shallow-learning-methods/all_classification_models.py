from contextlib import redirect_stderr, redirect_stdout
import io
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, NuSVC, LinearSVC
import time
from xgboost import XGBClassifier

class AllClassificationModels:
    """
    Wrapper class around all supported classification models: LogisticRegression, MLPClassifier, RandomForest, SVC,
    NuSVC, LinearSVC, and XGBClassifier.
    AllClassificationModels runs every available classification algorithm on the given dataset and outputs the mean
    accuracy, ROC-AUC, and execution time of each successful model when all_classification_models() is run.
    """
    def __init__(self, attributes=None, labels=None, test_size=0.25, verbose=False):
        """
        Initializes an AllClassificationModels object.

        The following parameters are needed to use an AllClassificationModels object:

            – attributes: a numpy array of the desired independent variables (Default is None)
            – labels: a numpy array of the classes (Default is None)
            – test_size: the proportion of the dataset to be used for testing the model;
            the proportion of the dataset to be used for training will be the complement of test_size (Default is 0.25)
            – verbose: specifies whether or not to ouput any and all logging during model training (Default is False)

            Note: These are the only parameters allowed. All other parameters for each model will use their default
            values. For more granular control, please instantiate each model individually.

        The following instance data is found after running all_classification_models() successfully:

            – logistic_regression: a reference to the LogisticRegression model
            – MLP: a reference to the MLPClassifier model
            – random_forest: a reference to the RandomForest model
            – SVC: a reference to the SVC model
            – nu_SVC: a reference to the NuSVC model
            – linear_SVC: a reference to the LinearSVC model
            – XGB_classifier: a reference to the XGBClassifier model

        After running all_classification_models(), the mean accuracy, ROC-AUC (if available), and execution time for
        each model that ran successfully will be displayed in tabular form. Any models that failed to run will be listed.
        """
        self.attributes = attributes
        self.labels = labels
        self.test_size = test_size
        self.verbose = verbose

        self.logistic_regression = LogisticRegression(verbose=self.verbose)
        self.MLP = MLPClassifier(verbose=self.verbose)
        self.random_forest = RandomForestClassifier(verbose=self.verbose)
        self.SVC = SVC(verbose=self.verbose, probability=True)
        self.nu_SVC = NuSVC(verbose=self.verbose, probability=True)
        self.linear_SVC = LinearSVC(verbose=self.verbose)
        self.XGB_classifier = XGBClassifier(verbosity=int(self.verbose))

        self._classification_models = {"Model": ["Accuracy", "ROC-AUC", "Time"]}
        self._failures = []

    # Accessor methods

    def get_attributes(self):
        """
        Accessor method for attributes.

        If an AllClassificationModels object is initialized without specifying attributes, attributes will be None.
        all_classification_models() cannot be called until attributes is a populated numpy array of independent variables;
        call set_attributes(new_attributes) to fix this.
        """
        return self.attributes

    def get_labels(self):
        """
        Accessor method for labels.

        If an AllClassificationModels object is initialized without specifying labels, labels will be None.
        all_classification_models() cannot be called until labels is a populated numpy array of classes;
        call set_labels(new_labels) to fix this.
        """
        return self.labels

    def get_test_size(self):
        """
        Accessor method for test_size.

        Should return a number or None.
        """
        return self.test_size

    def get_verbose(self):
        """
        Accessor method for verbose.

        Will default to False if not set by the user.
        """
        return self.verbose

    def get_all_classification_models(self):
        """
        Accessor method that returns a list of all models.

        All models within the list will be None if all_classification_models() hasn't been called, yet.
        """
        return [self.logistic_regression, self.MLP, self.random_forest, self.SVC, self.nu_SVC, self.linear_SVC,
                self.XGB_classifier]

    def get_logistic_regression(self):
        """
        Accessor method for logistic_regression.

        Will return None if all_classification_models() hasn't been called, yet.
        """
        return self.logistic_regression

    def get_MLP(self):
        """
        Accessor method for MLP.

        Will return None if all_classification_models() hasn't been called, yet.
        """
        return self.MLP

    def get_random_forest(self):
        """
        Accessor method for random_forest.

        Will return None if all_classification_models() hasn't been called, yet.
        """
        return self.random_forest

    def get_SVC(self):
        """
        Accessor method for SVC.

        Will return None if all_classification_models() hasn't been called, yet.
        """
        return self.SVC
    
    def get_nu_SVC(self):
        """
        Accessor method for nu_SVC.

        Will return None if all_classification_models() hasn't been called, yet.
        """
        return self.nu_SVC
    
    def get_linear_SVC(self):
        """
        Accessor method for linear_SVC.

        Will return None if all_classification_models() hasn't been called, yet.
        """
        return self.linear_SVC

    def get_XGB_classifier(self):
        """
        Accessor method for XGB_classifier.

        Will return None if all_classification_models() hasn't been called, yet.
        """
        return self.XGB_classifier

    # Modifier methods

    def set_attributes(self, new_attributes=None):
        """
        Modifier method for attributes.

        Input should be a numpy array of independent variables. Defaults to None.
        """
        self.attributes = new_attributes

    def set_labels(self, new_labels=None):
        """
        Modifier method for labels.

        Input should be a numpy array of classes. Defaults to None.
        """
        self.labels = new_labels

    def set_test_size(self, new_test_size=0.25):
        """
        Modifier method for test_size.

        Input should be a number or None. Defaults to 0.25.
        """
        self.test_size = new_test_size

    def set_verbose(self, new_verbose=False):
        """
        Modifier method for verbose.

        Input should be a truthy/falsy value. Defaults to False.
        """
        self.verbose = new_verbose

    # Classification functionality

    def all_classification_models(self):
        """
        Driver method for running all classification models with given attributes and labels.
        all_classification_models() first trains the models and determines their mean accuracy, ROC-AUC, and execution
        time via _all_classification_models_runner(). Then, all_classification_models() calls _print_results() to
        format and print each successful model's measurements, while also listing any failed models.

        If verbose is True, all verbose logging for each model will be enabled.
        If verbose is False, all logging to stdout and stderr will be suppressed.
        """

        # Call helper method for running all classification models; suppress output, if needed
        if not self.verbose:
            suppress_output = io.StringIO()
            with redirect_stderr(suppress_output), redirect_stdout(suppress_output):
                self._all_classification_models_runner()
        else:
            self._all_classification_models_runner()
        
        # Print results
        self._print_results()

    # Helper methods

    def _all_classification_models_runner(self):
        """
        Helper method that runs all models using the given dataset and all default parameters.
        After running all models, each model is determined to be either a success or failure, and relevant data
        (accuracy, ROC-AUC, execution time) is recorded.

        _all_classification_models_runner() may only be called by all_classification_models().
        """

        # Split dataset
        dataset_X_train, dataset_X_test, dataset_y_train, dataset_y_test =\
            train_test_split(self.attributes, self.labels, test_size=self.test_size)

        # Run and time all models; identify each as success or failure
        try:
            start_time = time.time()
            self.logistic_regression.fit(dataset_X_train, dataset_y_train)
            end_time = time.time()
            self._classification_models["LogisticRegression"] =\
                [self.logistic_regression.score(dataset_X_test, dataset_y_test),
                roc_auc_score(self.logistic_regression.predict(dataset_X_test),
                              self.logistic_regression.predict_proba(dataset_X_test)[::, 1]),
                end_time - start_time]
        except:
            self._failures.append("LogisticRegression")

        try:        
            start_time = time.time()
            self.MLP.fit(dataset_X_train, dataset_y_train)
            end_time = time.time()
            self._classification_models["MLPClassifier"] =\
                [self.MLP.score(dataset_X_test, dataset_y_test),
                    roc_auc_score(self.MLP.predict(dataset_X_test), self.MLP.predict_proba(dataset_X_test)[::, 1]),
                    end_time - start_time]
        except:
            self._failures.append("MLPClassifier")

        try:        
            start_time = time.time()
            self.random_forest.fit(dataset_X_train, dataset_y_train)
            end_time = time.time()
            self._classification_models["RandomForest"] =\
                [self.random_forest.score(dataset_X_test, dataset_y_test),
                    roc_auc_score(self.random_forest.predict(dataset_X_test),
                                self.random_forest.predict_proba(dataset_X_test)[::, 1]),
                    end_time - start_time]
        except:
            self._failures.append("RandomForest")
        
        try:
            start_time = time.time()
            self.SVC.fit(dataset_X_train, dataset_y_train)
            end_time = time.time()
            self._classification_models["SVC"] =\
                [self.SVC.score(dataset_X_test, dataset_y_test),
                    roc_auc_score(self.SVC.predict(dataset_X_test), self.SVC.predict_proba(dataset_X_test)[::, 1]),
                    end_time - start_time]
        except:
            self._failures.append("SVC")

        try:
            start_time = time.time()
            self.nu_SVC.fit(dataset_X_train, dataset_y_train)
            end_time = time.time()
            self._classification_models["NuSVC"] =\
                [self.nu_SVC.score(dataset_X_test, dataset_y_test),
                    roc_auc_score(self.nu_SVC.predict(dataset_X_test), self.nu_SVC.predict_proba(dataset_X_test)[::, 1]),
                    end_time - start_time]
        except:
            self._failures.append("NuSVC")

        try:
            start_time = time.time()
            self.linear_SVC.fit(dataset_X_train, dataset_y_train)
            end_time = time.time()
            self._classification_models["LinearSVC"] =\
                [self.linear_SVC.score(dataset_X_test, dataset_y_test), "Not Available", end_time - start_time]
        except:
            self._failures.append("LinearSVC")

        try:
            start_time = time.time()
            self.XGB_classifier.fit(dataset_X_train, dataset_y_train)
            end_time = time.time()
            self._classification_models["XGBClassifier"] =\
                [self.XGB_classifier.score(dataset_X_test, dataset_y_test),
                    roc_auc_score(self.XGB_classifier.predict(dataset_X_test),
                                  self.XGB_classifier.predict_proba(dataset_X_test)[::, 1]),
                    end_time - start_time]
        except:
            self._failures.append("XGBClassifier")
    
    def _print_results(self):
        """
        Helper method that prints results of _all_classification_models_runner() in tabular form.

        _print_results() may only be called by all_classification_models() after all models have attempted to run.
        """

        # Print models that didn't fail
        print("\nResults:\n")

        for model, data in self._classification_models.items():
            print("{:<20} {:<20} {:<20} {:<20}".format(model, data[0], data[1], data[2]))

        print()

        # Print failures, if any
        if len(self._failures) > 0:
            print("The following models failed to run:\n")

            for entry in self._failures:
                print(entry)
        
        print()