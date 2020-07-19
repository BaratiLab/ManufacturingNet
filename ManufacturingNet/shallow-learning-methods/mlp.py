from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import numpy as np  

class MLP:
    """
    Wrapper class around scikit-learn's multi-layer perceptron (MLP) classifier functionality.

    MLPClassifier provides multi-label classification on non-linear datasets using an underlying neural network.
    """

    def __init__(self, attributes=None, labels=None, test_size=0.25, hidden_layer_sizes=(100,), activation='relu',
                 solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001,
                 power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False,
                 momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
    			 beta_2=0.999, epsilon=1e-08, n_iter_no_change=10):
        """
        Initializes an MLP object.

        The following parameters are needed to create a logistic regression model:

            - attributes: A numpy array of the desired independent variables (Defaults to None).
            - labels: A numpy array of the desired dependent variables (Defaults to None).
            - test_size: The proportion of the dataset to be used for testing the model;
            the proportion of the dataset to be used for training will be the complement of test_size (Defaults to 0.25).
            - hidden_layer_sizes: The ith element represents the number of neurons in the ith hidden layer. (Defaults to (100,))
            - activation: Activation function for the hidden layer. (Defaults to "relu")
            - solver: The solver for weight optimization. "lbfgs" is an optimizer in the family of quasi-Newton methods.
            "sgd" refers to stochastic gradient descent. "adam" refers to a stochastic gradient-based optimizer proposed
            by Kingma, Diederik, and Jimmy Ba. Note: "adam" works pretty well for large datasets, but "lbfgs" can
            converge faster and perform better for small datasets. (Defaults to "adam")
            - alpha: L2 penalty (regularization term) parameter. (Defaults to 0.0001)
            - batch_size: Size of minibatches for stochastic optimizers. If the solver is "lbfgs", the classifier will
            not use minibatch. When set to "auto", batch_size=min(200, n_samples). (Default is "auto")
            - learning_rate: Learning rate schedule for weight updates. "constant" is a constant learning rate given by
            learning_rate_init. "invscaling" gradually decreases the learning rate at each time step ‘t’ using an
            inverse scaling exponent of ‘power_t’. effective_learning_rate = learning_rate_init / pow(t, power_t).
            "adaptive" keeps the learning rate constant to ‘learning_rate_init’ as long as training loss keeps
            decreasing. Each time two consecutive epochs fail to decrease training loss by at least tol, or fail to
            increase validation score by at least tol if ‘early_stopping’ is on, the current learning rate is divided by
            5. "learning_rate" is used only when solver="sgd". (Defaults to "constant")
            - learning_rate_init: The initial learning rate used. It controls the step-size in updating the weights.
            Only used when solver=’sgd’ or ‘adam’. (Defaults to 0.001)
            - power_t: The exponent for inverse scaling learning rate. It is used in updating effective learning rate
            when the learning_rate is set to ‘invscaling’. Only used when solver=’sgd’. (Defaults to 0.5)
            - max_iter: Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or
            this number of iterations. For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of
            epochs (how many times each data point will be used), not the number of gradient steps. (Defaults to 200)
            - shuffle: Whether to shuffle samples in each iteration. Only used when solver='sgd' or 'adam'.
            (Defaults to True)
            - random_state: Determines random number generation for weights and bias initialization, train-test split
            if early stopping is used, and batch sampling when solver=’sgd’ or ‘adam’. Pass an int for reproducible
            results across multiple function calls. (Defaults to None)
            - tol: Tolerance for the optimization. When the loss or score is not improving by at least tol for
            n_iter_no_change consecutive iterations, unless learning_rate is set to ‘adaptive’, convergence is
            considered to be reached and training stops. (Defaults to 1e-4, or 0.0001)
            - verbose: Whether to print progress messages to stdout. (Defaults to False)
            - warm_start: When True, reuse the solution of the previous call to fit as initialization, otherwise, just
            erase the previous solution. (Defaults to False)
            - momentum: Momentum for gradient descent update. Should be between 0 and 1. Only used when solver='sgd'.
            (Defaults to 0.9)
            - nesterovs_momentum: Whether to use Nesterov's momentum. Only used when solver='sgd' and momentum > 0.
            (Defaults to True)
            - early_stopping: Whether to use early stopping to terminate training when validation score is not improving.
            If set to true, it will automatically set aside 10% of training data as validation and terminate training
            when validation score is not improving by at least tol for n_iter_no_change consecutive epochs. The split is
            stratified, except in a multilabel setting. Only effective when solver=’sgd’ or ‘adam’. (Defaults to False)
            - validation_fraction: The proportion of training data to set aside as validation set for early stopping.
            Must be between 0 and 1. Only used if early_stopping is True. (Defaults to 0.1)
            - beta_1: Exponential decay rate for estimates of first moment vector in adam, should be in [0, 1). Only
            used when solver='adam'. (Defaults to 0.9)
            - beta_2: Exponential decay rate for estimates of second moment vector in adam, should be in [0, 1). Only
            used when solver=’adam’. (Defaults to 0.999)
            - epsilon: Value for numerical stability in adam. Only used when solver='adam'. (Defaults to 1e-8)
            - n_iter_no_change: Maximum number of epochs to not meet tol improvement. Only effective when solver='sgd' or
            'adam'. (Defaults to 10)
            - max_fun: Only used when solver=’lbfgs’. Maximum number of loss function calls. The solver iterates until
            convergence (determined by ‘tol’), number of iterations reaches max_iter, or this number of loss function
            calls. Note that number of loss function calls will be greater than or equal to the number of iterations for
            the MLPClassifier. (Defaults to 15000)

        The following instance data is found after successfully running MLP():

            - MLP_classifier: A reference to the MLPClassifier model.
            - classes: Class labels for each output.
            - coefs: The ith eleemnt in the list represents the weight matrix corresponding to layer i.
            - loss: The current loss computed with the loss function.
            - intercept: The ith element in the list represents the bias vector corresponding to layer i + 1.
            - n_iter: The number of iterations the solver has run.
            - n_layers: The number of layers.
            - n_outputs: The number of outputs.
            - out_activation: Name of the output activation function.
            - roc_auc: The area under the ROC curve for the model.
            - precision_scores: List of precision scores for all class labels.
            - accuracy: The mean accuracy of the model on the given dataset and labels.
            - recall_scores: List of recall scores for all class labels.
        """

        self.attributes = attributes
        self.labels = labels
        self.test_size = test_size

        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose 
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon=1e-08
        self.n_iter_no_change = n_iter_no_change

        self.MLP_classifier = None
        self.classes = None
        self.coefs = None
        self.loss = None
        self.intercept = None
        self.n_iter = None
        self.n_layers = None
        self.n_outputs = None
        self.out_activation = None

        self.roc_auc = None
        self.precision_scores = []
        self.accuracy = None
        self.recall_scores = []

    # Accessor methods

    def get_attributes(self):
        """
        Accessor method for attributes.

        If a MLP object is constructed without specifying attributes, attributes will be None.
        MLP() cannot be called until attributes is a populated numpy array; call set_attributes(new_attributes) to fix
        this.
        """
        return self.attributes

    def get_labels(self):
        """
        Accessor method for labels.

        If a MLP object is constructed without specifying labels, labels will be None.
        MLP() cannot be called until labels is a populated numpy array; call set_labels(new_labels) to fix this.
        """
        return self.labels

    def get_MLP_classifier(self):
        """
        Accessor method for MLP_classifier.

        Will return None if MLP() hasn't been called, yet.
        """
        return self.MLP_classifier
    
    def get_classes(self):
        """
        Accessor method for classes.

        Will return None if MLP() hasn't been called, yet.
        """
        return self.classes

    def get_coefs(self):
        """
        Accessor method for coefs.

        Will return None if MLP() hasn't been called, yet.
        """
        return self.coefs

    def get_n_iter(self):
        """
        Accessor method for number of iterations for all classes.

        Will return None if MLP() hasn't been called, yet.
        """
        return self.n_iter

    def get_accuracy(self):
        """
        Accessor method for accuracy.

        Will return None if MLP() hasn't been called, yet.
        """
        return self.accuracy

    def get_roc_auc(self):
        """
        Accessor method for roc-auc score.

        Will return None if MLP() hasn't been called, yet.
        """
        return self.roc_auc

    def get_precision(self, label):
        """
        Accessor method for precision of the given label.

        Will return None if MLP() hasn't been called, yet.
        """
        precision = self.precision_scores.get(label)
        return precision

    def get_precision_scores(self):
        """
        Accessor method for all precision scores.

        Will return None if MLP() hasn't been called, yet.
        """
        return self.precision_scores
    
    def get_recall(self, label):
        """
        Accessor method for recall of the given label.

        Will return None if MLP() hasn't been called, yet.
        """
        recall = self.recall_scores.get(label)
        return recall

    def get_recall_scores(self):
        """
        Accessor method for all recall scores.

        Will return None if MLP() hasn't been called, yet.
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

    def set_test_size(self, new_test_size=0.25):
        """
        Modifier method for test_size.

        Input should be a number or None. Defaults to 0.25.
        """
        self.test_size  = new_test_size

    # Wrapper for logistic regression model

    def MLP(self):
        """
        Performs logistic regression on dataset using scikit-learn's logistic_model and returns the resultant array of
        coefficients.
        """
        if self._check_inputs():
            # Instantiate MLP() object
            self.MLP_classifier =\
                MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation, solver=self.solver,
                              alpha=self.alpha, batch_size=self.batch_size, learning_rate=self.learning_rate,
                              learning_rate_init=self.learning_rate_init, power_t=self.power_t, max_iter=self.max_iter,
                              shuffle=self.shuffle, random_state=self.random_state, tol=self.tol, verbose=self.verbose,
    						  warm_start=self.warm_start, momentum=self.momentum,
                              nesterovs_momentum=self.nesterovs_momentum, early_stopping=self.early_stopping,
    						  validation_fraction=self.validation_fraction, beta_1=self.beta_1, beta_2=self.beta_2,
                              epsilon=self.epsilon, n_iter_no_change=self.n_iter_no_change)

            # Split into training and testing set
            dataset_X_train, dataset_X_test, dataset_y_train, dataset_y_test = \
                train_test_split(self.attributes,self.labels,test_size=self.test_size)

            # Train the model and get resultant coefficients; handle exception if arguments are incorrect
            try:
                self.MLP_classifier.fit(dataset_X_train, np.ravel(dataset_y_train))
            except Exception as e:
                print("An exception occurred while training the MLP classifier model. Check your arguments and try again.")
                print("Here is the exception message:")
                print(e)
                self.MLP_classifier = None
                return

            self.classes = self.MLP_classifier.classes_
            self.coefs = self.MLP_classifier.coefs_
            self.n_iter = self.MLP_classifier.n_iter_
            self.loss = self.MLP_classifier.loss_
            self.n_layers_ = self.MLP_classifier.n_layers_
            self.n_outputs_ = self.MLP_classifier.n_outputs_
            self.out_activation_ = self.MLP_classifier.out_activation_

            # Make predictions using testing set
            y_prediction = self.MLP_classifier.predict(dataset_X_test)
            y_pred_probas = self.MLP_classifier.predict_proba(dataset_X_test)[::, 1]

            # Metrics
            self.accuracy = accuracy_score(y_prediction, dataset_y_test)
            self.roc_auc = roc_auc_score(y_prediction, y_pred_probas)

            self.precision_scores = { each : precision_score(dataset_y_test, y_prediction, pos_label=each) \
                                                                    for each in self.classes}
            self.recall_scores = { each : recall_score(dataset_y_test, y_prediction, pos_label=each) \
                                                                    for each in self.classes}

    # Helper method for checking inputs

    def _check_inputs(self):
        """
        Verifies if the instance data is ready for use in logistic regression model.
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