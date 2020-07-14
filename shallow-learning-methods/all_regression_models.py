from contextlib import redirect_stderr, redirect_stdout
import io
from linear_regression import LinearRegression
from random_forest import RandomForest
from svm import SVM
import time

class AllRegressionModels:
    """
    Wrapper class around all supported regression models: LinearRegression, RandomForest, SVR, NuSVR, and LinearSVR.
    AllRegressionModels runs every available regression algorithm on the given dataset and outputs the coefficient of
    determination and execution time of each successful model when all_regression_models() is run.
    """
    def __init__(self, attributes=None, labels=None, test_size=0.25, verbose=False):
        """
        Initializes an AllRegressionModels object.

        The following parameters are needed to use an AllRegressionModels object:

            – attributes: a numpy array of the desired independent variables (Default is None)
            – labels: a numpy array of the desired dependent variables (Default is None)
            – test_size: the proportion of the dataset to be used for testing the model;
            the proportion of the dataset to be used for training will be the complement of test_size (Default is 0.25)
            – verbose: specifies whether or not to ouput any and all logging during model training (Default is False)

            Note: These are the only parameters allowed. All other parameters for each model will use their default
            values. For more granular control, please instantiate each model individually.

        The following instance data is found after running all_regression_models() successfully:

            – linear_regression: a reference to the LinearRegression object
            – random_forest: a reference to the RandomForest object
            – SVM: a reference to the SVM object
        
        After running all_regression_models(), the coefficient of determination and execution time for each model that
        ran successfully will be displayed in tabular form. Any models that failed to run will be listed.
        """
        self.attributes = attributes
        self.labels = labels
        self.test_size = test_size
        self.verbose = verbose

        self.linear_regression = None
        self.random_forest = None
        self.SVM = None

        self._regression_models = {"Model": ["R2 Score", "Time"]}
        self._failures = []

    # Accessor methods

    def get_attributes(self):
        """
        Accessor method for attributes.

        If an AllRegressionModels object is initialized without specifying attributes, attributes will be None.
        all_regression_models() cannot be called until attributes is a populated numpy array of independent variables;
        call set_attributes(new_attributes) to fix this.
        """
        return self.attributes

    def get_labels(self):
        """
        Accessor method for labels.

        If an AllRegressionModels object is initialized without specifying labels, labels will be None.
        all_regression_models() cannot be called until labels is a populated numpy array of dependent variables;
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

    def get_all_regression_models(self):
        """
        Accessor method that returns a list of all models.

        All models within the list will be None if all_regression_models() hasn't been called, yet.
        """
        return [self.linear_regression, self.random_forest, self.SVM]

    def get_linear_regression(self):
        """
        Accessor method for linear_regression.

        Will return None if all_regression_models() hasn't been called, yet.
        """
        return self.linear_regression

    def get_random_forest(self):
        """
        Accessor method for random_forest.

        Will return None if all_regression_models() hasn't been called, yet.
        """
        return self.random_forest

    def get_SVM(self):
        """
        Accessor method for SVM.

        Will return None if all_regression_models() hasn't been called, yet.
        """
        return self.SVM

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

        Input should be a numpy array of dependent variables. Defaults to None.
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

    # Regression functionality

    def all_regression_models(self):
        """
        Driver method for running all regression models with given attributes and labels.
        all_regression_models() first trains the models and determines their coefficients of determination and
        execution time via _all_regression_models_runner(). Then, all_regression_models() calls _print_results() to
        format and print each successful model's coefficient of determination and execution time, while also listing
        any failed models.

        If verbose is True, all verbose logging for each model will be enabled.
        If verbose is False, all logging to stdout and stderr will be suppressed.
        """

        # Call helper method for running all regression models; suppress output, if needed
        if not self.verbose:
            suppress_output = io.StringIO()
            with redirect_stderr(suppress_output), redirect_stdout(suppress_output):
                self._all_regression_models_runner()
        else:
            self._all_regression_models_runner()
        
        # Print results
        self._print_results()
        
    # Helper methods

    def _all_regression_models_runner(self):
        """
        Helper method that instantiates and runs all models using the given dataset and all default parameters.
        After running all models, each model is determined to be either a success or failure, and relevant data
        (R2 score, execution time) is recorded.

        _all_regression_models_runner() may only be called by all_regression_models().
        """

        # Instantiate all models
        self.linear_regression = LinearRegression(self.attributes, self.labels, self.test_size)
        self.random_forest = RandomForest(self.attributes, self.labels, self.test_size)
        self.SVM = SVM(self.attributes, self.labels, self.test_size)

        # Run and time all models; identify each as success or failure
        start_time = time.time()
        self.linear_regression.linear_regression()
        end_time = time.time()

        if not self.linear_regression.get_regression():
            self._failures.append("LinearRegression")
        else:
            self._regression_models["LinearRegression"] =\
                [self.linear_regression.get_r2_score(), end_time - start_time]

        start_time = time.time()
        self.random_forest.random_forest_regressor(verbose=self.verbose)
        end_time = time.time()

        if not self.random_forest.get_regressor_RF():
            self._failures.append("RandomForest")
        else:
            self._regression_models["RandomForest"] =\
                [self.random_forest.get_regressor_r2_score(), end_time - start_time]
        
        start_time = time.time()
        self.SVM.SVR(verbose=self.verbose)
        end_time = time.time()

        if not self.SVM.get_regression_SVR():
            self._failures.append("SVR")
        else:
            self._regression_models["SVR"] = [self.SVM.get_r2_score_SVR(), end_time - start_time]
        
        start_time = time.time()
        self.SVM.nu_SVR(verbose=self.verbose)
        end_time = time.time()

        if not self.SVM.get_regression_nu_SVR():
            self._failures.append("NuSVR")
        else:
            self._regression_models["NuSVR"] = [self.SVM.get_r2_score_nu_SVR(), end_time - start_time]

        start_time = time.time()
        self.SVM.linear_SVR(verbose=self.verbose)
        end_time = time.time()

        if not self.SVM.get_regression_linear_SVR():
            self._failures.append("LinearSVR")
        else:
            self._regression_models["LinearSVR"] = [self.SVM.get_r2_score_linear_SVR(), end_time - start_time]
    
    def _print_results(self):
        """
        Helper method that prints results of _all_regression_models_runner() in tabular form.

        _print_results() may only be called by all_regression_models() after all models have attempted to run.
        """

        # Print models that didn't fail
        print("\nResults:\n")

        for model, data in self._regression_models.items():
            print("{:<20} {:<20} {:<20}".format(model, data[0], data[1]))

        print()

        # Print failures, if any
        if len(self._failures) > 0:
            print("The following models failed to run:\n")

            for entry in self._failures:
                print(entry)
        
        print()