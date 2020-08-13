from math import sqrt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, roc_auc_score, r2_score
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

class RandomForest:
    """
    Class framework for random forest classification and regression models.
    """

    def __init__(self, attributes=None, labels=None):
        """
        Initializes a RandomForest object.
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
        
        self.gridsearch = False
        self.gs_params = None
        self.gs_results = None    
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

    def get_classifier(self):
        """
        Accessor method for classifier.
        """
        return self.classifier

    def get_accuracy(self):
        """
        Accessor method for accuracy.
        """
        return self.accuracy

    def get_roc_auc(self):
        """
        Accessor method for roc_auc.
        """
        return self.roc_auc
    
    def get_cross_val_scores_classifier(self):
        """
        Accessor method for cross_val_scores_classifier.
        """
        return self.cross_val_scores_classifier

    def get_regressor(self):
        """
        Accessor method for regressor.
        """
        return self.regressor

    def get_r2_score(self):
        """
        Accessor method for r2_score.
        """
        return self.r2_score

    def get_r_score(self):
        """
        Accessor method for r_score.
        """
        return self.r_score

    def get_mean_squared_error(self):
        """
        Accessor method for mean_squared_error.
        """
        return self.mean_squared_error
    
    def get_cross_val_scores_regressor(self):
        """
        Accessor method for cross_val_scores_regressor.
        """
        return self.cross_val_scores_regressor

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

    # Wrappers for RandomForest classes

    def run_classifier(self):
        """
        Provides random forest's classifier functionality.
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
            #self.roc_auc = roc_auc_score(self.classifier.predict(dataset_X_test),
                                         #self.classifier.predict_proba(dataset_X_test)[::, 1])
            self.cross_val_scores_classifier = cross_val_score(self.classifier, self.attributes, self.labels, cv=self.cv)

            # Output results
            self._output_classifier_results()

    def run_regressor(self):
        """
        Provides random forest's regressor functionality.
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
            input("=======================================================\n")
            
            if classifier:
                return RandomForestClassifier()
            else:
                return RandomForestRegressor()
        
        print("\nIf you are unsure about a parameter, press enter to use its default value.")
        print("Invalid parameter inputs will be replaced with their default values.")
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

            print('\n')
            user_input = input("Use Grid Search to find the best parameters? Enter y/N: ").lower()

            if user_input == "q":
                break
            elif user_input == "y":

                self.gridsearch = True
                params = {}
                print("Enter the max_features for the best split. ")
                user_input = input("Options: 1-auto, 2-sqrt, 3-log2. Enter 'all' for all options. (Example input: 1,2) : ")
                if user_input == 'q':
                    self.gridsearch = False
                    break
                elif user_input == "all":
                    feat_params = ['auto', 'sqrt', 'log2']
                else:
                    feat_dict = {1:'auto', 2:'sqrt', 3:'log2'}
                    feat_params_int = list(map(int, list(user_input.split(","))))
                    feat_params = []
                    for each in feat_params_int:
                        feat_params.append(feat_dict.get(each))

                params['max_features'] = feat_params

                user_input = input("Enter the list of num_estimators to try out (Example input: 1,2,3) : ")
                if user_input == "q":
                    break
                n_est_params = list(map(int, list(user_input.split(","))))
                params['n_estimators'] = n_est_params

                
                if classifier:
                    user_input = input("Enter the criterion to be tried for (Options: 1-gini, 2-entropy. Enter 'all' for all options): ")
                    if user_input == "q":
                        break
                    elif user_input == "all":
                        crit_params = ['gini', 'entropy']
                    else:
                        crit_dict = {1:'gini', 2:'entropy'}
                        crit_params_int = list(user_input.split(",")) 
                        crit_params = []
                        for each in crit_params_int:
                            crit_params.append(crit_dict.get(each))

                else:
                    user_input = input("Enter the criterion to be tried for (Options: 1-mse, 2-mae. Enter 'all' for all options): ")
                    if user_input == "q":
                        break
                    elif user_input == "all":
                        crit_params = ['mse', 'mae']
                    else:
                        crit_dict = {1:'mse', 2:'mae'}
                        crit_params_int = list(user_input.split(",")) 
                        crit_params = []
                        for each in crit_params_int:
                            crit_params.append(crit_dict.get(each))

                params['criterion'] = crit_params

                user_input = input('Enter the maximum depth of trees to try for (Example input: 1,2,3) : ')
                if user_input == "q":
                    break
                max_dep_params = list(map(int, list(user_input.split(","))))
                params['max_depth'] = max_dep_params

                self.gs_params = params


            print('\n')

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

                user_input = input("Automatically balance the class weights (y/N)? ").lower()

                if user_input == "q":
                    break
                elif user_input == "y":
                    class_weight = "balanced"
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
        	if self.gridsearch:
        		acc_scorer = make_scorer(accuracy_score)

        		clf = RandomForestClassifier()
        		dataset_X_train, dataset_X_test, dataset_y_train, dataset_y_test = train_test_split(self.attributes, self.labels, test_size=self.test_size)
        		
        		#Run the grid search
        		grid_obj = GridSearchCV(clf, self.gs_params, scoring=acc_scorer)
        		grid_obj = grid_obj.fit(dataset_X_train, dataset_y_train)

        		#Set the clf to the best combination of parameters
        		clf = grid_obj.best_estimator_

        		print('Best Grid Search Parameters: ')
        		print(grid_obj.best_params_)

        		# Fit the best algorithm to the data. 
        		clf.fit(dataset_X_train, dataset_y_train)
        		predictions = clf.predict(dataset_X_test)
        		self.gs_result = accuracy_score(dataset_y_test, predictions)

        	return RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                          min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                          min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                          max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                          min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score,
                                          n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start,
                                          class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)
        else:

            if self.gridsearch:
                #acc_scorer = make_scorer(accuracy_score)

                clf = RandomForestRegressor()
                dataset_X_train, dataset_X_test, dataset_y_train, dataset_y_test = train_test_split(self.attributes, self.labels, test_size=self.test_size)
                
                #Run the grid search
                grid_obj = GridSearchCV(clf, self.gs_params, scoring='r2')
                grid_obj = grid_obj.fit(dataset_X_train, dataset_y_train)

                #Set the clf to the best combination of parameters
                clf = grid_obj.best_estimator_

                print('Best Grid Search Parameters: ')
                print(grid_obj.best_params_)

                # Fit the best algorithm to the data. 
                clf.fit(dataset_X_train, dataset_y_train)
                predictions = clf.predict(dataset_X_test)
                self.gs_result = r2_score(dataset_y_test, predictions)

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
        #print("\n{:<20} {:<20}".format("ROC AUC:", self.roc_auc))
        print("\nCross Validation Scores: ", self.cross_val_scores_classifier)
        if self.gridsearch:
        	print("\nGrid Search Score: ", self.gs_result)
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
        print("\nCross Validation Scores:", self.cross_val_scores_regressor)
        if self.gridsearch:
            print("\nGrid Search Score:", self.gs_result)
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