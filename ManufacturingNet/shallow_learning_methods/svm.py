from math import sqrt
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC, NuSVC, LinearSVC, SVR, NuSVR, LinearSVR

class SVM:
    """
    Class model for support vector machine (SVM) models.
    """

    def __init__(self, attributes=None, labels=None):
        """
        Initializes a SVM object.
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
        """
        return self.attributes

    def get_labels(self):
        """
        Accessor method for labels.
        """
        return self.labels

    def get_classifier_SVC(self):
        """
        Accessor method for classifier_SVC.
        """
        return self.classifier_SVC

    def get_accuracy_SVC(self):
        """
        Accessor method for accuracy_SVC.
        """
        return self.accuracy_SVC

    def get_roc_auc_SVC(self):
        """
        Accessor method for roc_auc_SVC.
        """
        return self.roc_auc_SVC
    
    def get_cross_val_scores_SVC(self):
        """
        Accessor method for cross_val_scores_SVC.
        """
        return self.cross_val_scores_SVC

    def get_classifier_nu_SVC(self):
        """
        Accessor method for classifier_nu_SVC.
        """
        return self.classifier_nu_SVC

    def get_accuracy_nu_SVC(self):
        """
        Accessor method for accuracy_nu_SVC.
        """
        return self.accuracy_nu_SVC

    def get_roc_auc_nu_SVC(self):
        """
        Accessor method for roc_auc_nu_SVC.
        """
        return self.roc_auc_nu_SVC
    
    def get_cross_val_scores_nu_SVC(self):
        """
        Accessor method for cross_val_scores_nu_SVC.
        """
        return self.cross_val_scores_nu_SVC

    def get_classifier_linear_SVC(self):
        """
        Accessor method for classifier_linear_SVC.
        """
        return self.classifier_linear_SVC

    def get_accuracy_linear_SVC(self):
        """
        Accessor method for accuracy_linear_SVC.
        """
        return self.accuracy_linear_SVC
    
    def get_cross_val_scores_linear_SVC(self):
        """
        Accessor method for cross_val_scores_linear_SVC.
        """
        return self.cross_val_scores_linear_SVC

    def get_regressor_SVR(self):
        """
        Accessor method for regressor_SVR.
        """
        return self.regressor_SVR
    
    def get_mean_squared_error_SVR(self):
        """
        Accessor method for mean_squared_error_SVR.
        """
        return self.mean_squared_error_SVR

    def get_r2_score_SVR(self):
        """
        Accessor method for r2_score_SVR.
        """
        return self.r2_score_SVR

    def get_r_score_SVR(self):
        """
        Accessor method for r_score_SVR.
        """
        return self.r_score_SVR
    
    def get_cross_val_scores_SVR(self):
        """
        Accessor method for cross_val_scores_SVR.
        """
        return self.cross_val_scores_SVR

    def get_regressor_nu_SVR(self):
        """
        Accessor method for regressor_nu_SVR.
        """
        return self.regressor_nu_SVR
    
    def get_mean_squared_error_nu_SVR(self):
        """
        Accessor method for mean_squared_error_nu_SVR.
        """
        return self.mean_squared_error_nu_SVR

    def get_r2_score_nu_SVR(self):
        """
        Accessor method for r2_score_nu_SVR.
        """
        return self.r2_score_nu_SVR

    def get_r_score_nu_SVR(self):
        """
        Accessor method for r_score_nu_SVR.
        """
        return self.r_score_nu_SVR
    
    def get_cross_val_scores_nu_SVR(self):
        """
        Accessor method for cross_val_scores_nu_SVR.
        """
        return self.cross_val_scores_nu_SVR

    def get_regressor_linear_SVR(self):
        """
        Accessor method for regressor_linear_SVR.
        """
        return self.regressor_linear_SVR
    
    def get_mean_squared_error_linear_SVR(self):
        """
        Accessor method for mean_squared_error_linear_SVR.
        """
        return self.mean_squared_error_linear_SVR

    def get_r2_score_linear_SVR(self):
        """
        Accessor method for r2_score_linear_SVR.
        """
        return self.r2_score_linear_SVR

    def get_r_score_linear_SVR(self):
        """
        Accessor method for r_score_linear_SVR.
        """
        return self.r_score_linear_SVR
    
    def get_cross_val_scores_linear_SVR(self):
        """
        Accessor method for cross_val_scores_linear_SVR.
        """
        return self.cross_val_scores_linear_SVR

    # Modifier Methods

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

    # Wrappers for SVM classification classes

    def run_SVC(self):
        """
        Runs SVC model.
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
        Classifies each datapoint in dataset_X using the SVC model. Returns the predicted classifications.
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
        Classifies each datapoint in dataset_X using the NuSVC model. Returns the predicted classifications.
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
        Classifies each datapoint in dataset_X using the LinearSVC model. Returns the predicted classifications.
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
        Predicts the output of each datapoint in dataset_X using the SVR model. Returns the predictions.
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
        Predicts the output of each datapoint in dataset_X using the NuSVR model. Returns the predictions.
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
        Predicts the output of each datapoint in dataset_X using the LinearSVR model. Returns the predictions.
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