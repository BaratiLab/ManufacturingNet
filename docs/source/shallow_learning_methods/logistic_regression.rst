*******************
Logistic Regression
*******************

A logistic regression model uses a set of independent variables/features to determine the probability of an observation
falling into a specific category. Despite its name, logistic regression is commonly used for classification problems:
When paired with a decision function, logistic regression models can be used to provide binary or multinomial
classification by basing its decision on the probability distribution for the features.

Ultimately, logistic regression is a form of regression analysis, but because we anticipate most users choosing it for
classification purposes, we've listed it under both Classification and Regression to increase discoverability.

ManufacturingNet's logistic regression functionality is provided through the **LogRegression** class.

*LogRegression(attributes=None, labels=None)*

Parameters
==========

When initializing a LogRegression object, the following parameters need to be passed:

- **attributes** *(numpy array, default=None)*: A numpy array of the values of the independent variable(s).
- **labels** *(numpy array, default=None)*: A numpy array of the class labels.

When the run() method is called, the following parameters can be modified:

- **test_size** *(float, default=0.25)*: The proportion of the dataset to be used for testing the model; the proportion of the dataset to be used for training will be the complement of test_size.
- **cv** *(integer, default=None)*: The number of folds to use for cross validation.
- **graph_results** *(boolean, default=False)*: Determines whether to plot the ROC curve. Supported for binary classification only.
- **penalty** *('l1', 'l2', 'elasticnet', or 'none'; default='l2')*: Specifies the penalization norm.
- **dual** *(boolean, default=False)*: If True, dual formulation is used; else, primal formulation is used.
- **tol** *(float, default=0.0001)*: The acceptable margin of error for stopping criteria.
- **C** *(float, default=1.0)*: Positive number that specifies the inverse of the regularization strength.
- **fit_intercept** *(boolean, default=True)*: Determines whether to calculate an intercept for the decision function.
- **intercept_scaling** *(float, default=1)*: Used only when the solver ‘liblinear’ is used and fit_intercept is True. When enabled, a "synthetic" feature with a constant value equal to intercept_scaling is appended to the instance vector.
- **class_weight** *(boolean, default=False)*: Determines whether to automatically balance the class weights using class frequencies.
- **random_state** *(integer, default=None)*: The seed for random number generation.
- **solver** *(‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, or ‘saga’; default='lbfgs')*: The optimization algorithm.
- **max_iter** *(integer, default=100)*: Sets the maximum number of iterations the solver can take to converge.
- **multi_class** *('auto', 'ovr', or 'multinomial'; default='auto')*: Chooses whether to fit a binary problem or a multi-class problem for each label.
- **verbose** *(boolean, default=False)*: Determines whether to output logs while training.
- **warm_start** *(boolean, default=False)*: When True, the solution of the previous call is used to fit for initialization.
- **n_jobs** *(integer, default=None)*: The number of jobs to use for the computation.
- **l1_ratio** *(float, default=None)*: The Elastic-Net mixing parameter. Setting this to 0 is uses l2 penalty, setting this to 1 uses l1_penalty, and using a value between 0 and 1 is a combination of l1 and l2.

Attributes
==========

After run() successfully trains the model, the following instance data is available:

- **regression** *(model)*: The underlying logistic regression model.
- **coefficients** *(array of floats)*: An array of the coefficients from the decision function.
- **intercept** *(float)*: The intercept of the decision function, if it has one.
- **classes** *(array of multiple possible types)*: An array of the known class labels.
- **n_iter** *(array of integers or (1,))*: The actual number of iterations for all classes. If binary or multinomial, n_iter is one element. For the 'liblinear' solver, only the maximum number of iterations across all classes is given.
- **accuracy** *(float)*: The classification accuracy score.
- **roc_auc** *(float)*: The area under the receiver operating characteristic (ROC) curve from the prediction scores. Supported for binary classification only.
- **confusion_matrix** *(2D array of integers)*: A matrix where the entry in the *i* th row and *j* th column is the number of observations present in group *i* and predicted to be in group *j*. Supported for multilabel classification only.
- **cross_val_scores** *(array of floats)*: An array of the cross validation scores for the model.

Methods
=======

- **run()**: Prompts the user for the model parameters and trains a logistic regression model using attributes and labels. If successful, the above instance data is updated, and the model metrics are displayed. If the model is being used for binary classification, the ROC curve will be graphed and displayed.
- **predict(dataset_X=None)**: Uses the logistic regression model to classify the observations in dataset_X. If successful, the classifications are displayed and returned. predict() can only be called after run() has successfully trained the model.

Accessor Methods
----------------

- **get_attributes()**: Returns attributes.
- **get_labels()**: Returns labels.

Note: If attributes wasn't passed in during initialization, get_attributes() will return None. Likewise, if labels
wasn't passed in during initialization, get_labels() will return None.

- **get_classes()**: Returns classes.
- **get_regression()**: Returns classes.
- **get_coefficients()**: Returns coefficients.
- **get_n_iter()**: Returns n_iter.
- **get_accuracy()**: Returns accuracy.
- **get_roc_auc()**: Returns roc_auc.
- **get_confusion_matrix()**: Returns confusion_matrix.
- **get_cross_val_scores()**: Returns cross_val_scores.

Note: If run() hasn't successfully executed yet, the above accessor methods will return None.

Modifier Methods
----------------

- **set_attributes(new_attributes)**: Sets attributes to new_attributes. If new_attributes isn't specified, attributes is set to None.
- **set_labels(new_labels)**: Sets labels to new_labels. If new_labels isn't specified, labels is set to None.

Example Usage
=============

.. code-block:: python
    :linenos:

    from ManufacturingNet.models import LogRegression
    from pandas import read_csv

    dataset = read_csv('/path/to/dataset.csv')
    dataset = dataset.to_numpy()
    attributes = dataset[:, 0:5]                    # Columns 1-5 contain our features
    labels = dataset[:, 5]                          # Column 6 contains our class labels
    log_model = LinRegression(attributes, labels)
    log_model.run()                                 # This will trigger the command-line interface for parameter input

    new_data_X = read_csv('/path/to/new_data_X.csv')
    new_data_X = new_data_X.to_numpy()
    classifications = log_model.predict(new_data_X) # This will return and output classifications for new_data_X