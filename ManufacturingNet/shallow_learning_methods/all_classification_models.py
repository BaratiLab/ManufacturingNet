from contextlib import redirect_stderr, redirect_stdout
import io
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, NuSVC, LinearSVC
import time
from xgboost import XGBClassifier

class AllClassificationModels:
    """
    Wrapper class around all supported classification models: LogisticRegression, RandomForest, SVC,
    NuSVC, LinearSVC, and XGBClassifier.

    AllClassificationModels runs every available classification algorithm on the given dataset and outputs the mean
    accuracy, ROC-AUC, and execution time of each successful model when run() is run.
    """
    def __init__(self, attributes=None, labels=None):
        """
        Initializes an AllClassificationModels object.
        """
        self.attributes = attributes
        self.labels = labels
        self.test_size = None
        self.verbose = None

        self.logistic_regression = None
        self.random_forest = None
        self.SVC = None
        self.nu_SVC = None
        self.linear_SVC = None
        self.XGB_classifier = None

        self._classification_models = {"Model": ["Accuracy", "ROC-AUC", "Time (seconds)"]}
        self._failures = []

    # Accessor methods

    def get_attributes(self):
        """
        Accessor method for attributes.
        """
        return self.attributes

    def get_labels(self):
        """
        Accessor method for labels.
        """
        return self.labels

    def get_all_classification_models(self):
        """
        Accessor method that returns a list of all models.
        """
        return [self.logistic_regression, self.random_forest, self.SVC, self.nu_SVC, self.linear_SVC,
                self.XGB_classifier]

    def get_logistic_regression(self):
        """
        Accessor method for logistic_regression.
        """
        return self.logistic_regression

    def get_random_forest(self):
        """
        Accessor method for random_forest.
        """
        return self.random_forest

    def get_SVC(self):
        """
        Accessor method for SVC.
        """
        return self.SVC
    
    def get_nu_SVC(self):
        """
        Accessor method for nu_SVC.
        """
        return self.nu_SVC
    
    def get_linear_SVC(self):
        """
        Accessor method for linear_SVC.
        """
        return self.linear_SVC

    def get_XGB_classifier(self):
        """
        Accessor method for XGB_classifier.
        """
        return self.XGB_classifier

    # Modifier methods

    def set_attributes(self, new_attributes=None):
        """
        Modifier method for attributes.
        """
        self.attributes = new_attributes

    def set_labels(self, new_labels=None):
        """
        Modifier method for labels.
        """
        self.labels = new_labels

    # Classification functionality

    def run(self):
        """
        Driver method for running all classification models with given attributes and labels.
        """
        # Get parameters; create models
        self._create_models()

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

    def _create_models(self):
        """
        Prompts user for parameter input and instantiates all the classifier models.
        """
        print("\n==========================================")
        print("= All Classifier Models Parameter Inputs =")
        print("==========================================\n")

        # Get user input for verbose, test_size
        user_input = input("Enable verbose logging (y/N)? ").lower()

        if user_input == "y":
            self.verbose = True
        else:
            self.verbose = False
        
        user_input = input("What fraction of the dataset should be used for testing? Enter a decimal: ")

        try:
            self.test_size = float(user_input)
        except:
            self.test_size = 0.25
        
        print("\n=======================================================")
        print("= End of parameter inputs; press any key to continue. =")
        input("=======================================================\n")

        # Create models
        self.logistic_regression = LogisticRegression(verbose=self.verbose)
        self.random_forest = RandomForestClassifier(verbose=self.verbose)
        self.SVC = SVC(verbose=self.verbose, probability=True)
        self.nu_SVC = NuSVC(verbose=self.verbose, probability=True)
        self.linear_SVC = LinearSVC(verbose=self.verbose)
        self.XGB_classifier = XGBClassifier(verbosity=int(self.verbose))

    def _all_classification_models_runner(self):
        """
        Helper method that runs all models using the given dataset and all default parameters.
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
        """

        # Print models that didn't fail
        print("\n===========")
        print("= Results =")
        print("===========\n")

        for model, data in self._classification_models.items():
            print("{:<20} {:<20} {:<20} {:<20}".format(model, data[0], data[1], data[2]))

        print()

        # Print failures, if any
        if len(self._failures) > 0:
            print("The following models failed to run:\n")

            for entry in self._failures:
                print(entry)
        
        print()