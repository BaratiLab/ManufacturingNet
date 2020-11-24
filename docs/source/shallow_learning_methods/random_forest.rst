*************
Random Forest
*************

A random forest model utilizes multiple underlying decision trees to provide classification and regression functionality.
Each decision tree trains on a portion of the dataset; all of the trees' predictions are eventually combined for a final
prediction, hence the "forest" model. This technique is called bagging.

ManufacturingNet's random forest functionality is provided through the **RandomForest** class.

*RandomForest(attributes=None, labels=None)*

Go To
=====

    - :ref:`classification`
    - :ref:`regression`

Universal Parameters
====================

When initializing a RandomForest object, the following parameters need to be passed:

- **attributes** *(numpy array, default=None)*: A numpy array of the values of the independent variable(s).
- **labels** *(numpy array, default=None)*: A numpy array of the values of the dependent variable or the class labels.

When run_classifier() or run_regressor() is called, the following parameters can be modified:

- **test_size** *(float, default=0.25)*: The proportion of the dataset to be used for testing the model; the proportion of the dataset to be used for training will be the complement of test_size.
- **cv** *(integer, default=None)*: The number of folds to use for cross validation.
- **n_estimators** *(integer, default=100)*: The number of decision trees.
- **max_depth** *(integer, default=None)*: The maximum depth of each tree.
- **min_samples_split** *(integer or float, default=2)*: The number of samples required to split an internal node.
- **min_samples_leaf** *(integer or float, default=1)*: The number of samples required to reach a leaf node.
- **min_weight_fraction_leaf** *(float, default=0.0)*: The minimum weighted fraction of the weight of all input samples required to reach a leaf node.
- **max_features** *('auto', 'sqrt', 'log2', integer, or float; default='auto')*: The number of features to consider when determining the best split.
    - If max_features is an integer, then max_features features is considered.
    - If max_features is a fraction, then the number of features considered is int(max_features * n_features).
    - If max_features is 'auto' or 'sqrt', then the number of features considered is sqrt(n_features).
    - If max_features is 'log2', then the number of features considered is log2(n_features).
    - If max_features is None, then all features are considered.
- **max_leaf_nodes** *(integer, default=None)*: Determines the maximum number of leaf nodes allowed per tree. If None, then trees can have an unlimited number of leaf nodes.
- **min_impurity_decrease** *(float, default=0.0)*: The impurity decrease required to split a node.
- **bootstrap** *(boolean, default=True)*: Determines whether bootstrap samples are used when growing trees.
- **oob_score** *(boolean, default=False)*: Determines whether to use out-of-bag samples.
- **n_jobs** *(integer, default=None)*: The number of jobs to run in parallel. If None, n_jobs is 1. If -1, all processors are used.
- **random_state** *(integer, default=None)*: The seed for random number generation.
- **verbose** *(boolean, default=False)*: Determines whether to output logs when fitting and predicting.
- **warm_start** *(boolean, default=False)*: Determines whether to reuse the solution of the previous call to add more estimators.
- **ccp_alpha** *(non-negative float, default=0.0)*: Complexity parameter for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity less than ccp_alpha will be pruned.
- **max_samples** *(int or float, default=None)*: If bootstrap is True, max_samples is the number of samples to draw for training each base estimator.

Universal Methods
=================

Accessor Methods
----------------

- **get_attributes()**: Returns attributes.
- **get_labels()**: Returns labels.

Note: If attributes wasn't passed in during initialization, get_attributes() will return None. Likewise, if labels
wasn't passed in during initialization, get_labels() will return None.

Modifier Methods
----------------

- **set_attributes(new_attributes=None)**: Sets attributes to new_attributes. If new_attributes isn't specified, attributes is set to None.
- **set_labels(new_labels=None)**: Sets labels to new_labels. If new_labels isn't specified, labels is set to None.

--------------

.. _classification:

Classification
==============

Classifier Parameters
---------------------

The following parameters can be modified when run_classifier() is called:

- **graph_results** *(boolean, default=False)*: Determines whether to plot the ROC curve. Supported for binary classification only.
- **criterion** *('gini' or 'entropy', default='gini')*: Determines the metric for calculating split quality. 'gini' is for Gini impurity, and 'entropy' is for information gain.
- **class_weight** *(boolean, default=False)*: Determines whether to automatically balance the class weights.

Classifier Attributes
---------------------

After run_classifier() successfully trains the classifier model, the following instance data is available:

- **classifier** *(model)*: The underlying random forest classifier model.
- **accuracy** *(float)*: The classification accuracy score.
- **roc_auc** *(float)*: The area under the receiver operating characteristic (ROC) curve from the prediction scores. Supported for binary classification only.
- **confusion_matrix** *(2D array of integers)*: A matrix where the entry in the *i* th row and *j* th column is the number of observations present in group *i* and predicted to be in group *j*. Supported for multilabel classification only.
- **cross_val_scores_classifier** *(array of floats)*: An array of the cross validation scores for the classifier model.
- **feature_importances_classifier** *(array of floats)*: An array of the feature importances for the classifier model. The higher the score, the more useful the feature is for prediction.

Classifier Methods
------------------

- **run_classifier()**: Prompts the user for the model parameters and trains a random forest classifier model using attributes and labels. If successful, the classifier instance data is updated, and the model metrics are displayed. If the model is being used for binary classification, the ROC curve will be graphed and displayed.
- **predict_classifier(dataset_X=None)**: Uses the random forest classifier model to classify the observations in dataset_X. If successful, the classifications are displayed and returned. predict_classifier() can only be called after run_classifier() has successfully trained the classifier.

Classifier Accessor Methods
***************************

- **get_classifier()**: Returns classifier.
- **get_accuracy()**: Returns accuracy.
- **get_roc_auc()**: Returns roc_auc.
- **get_confusion_matrix()**: Returns confusion_matrix.
- **get_cross_val_scores_classifier()**: Returns cross_val_scores_classifier.
- **get_feature_importances_classifier()**: Returns feature_importances_classifier.

Note: If run_classifier() hasn't successfully executed yet, the above accessor methods will return None.

Classifier Example Usage
------------------------

.. code-block:: python
    :linenos:

    from ManufacturingNet.models import RandomForest
    from pandas import read_csv

    dataset = read_csv('/path/to/dataset.csv')
    dataset = dataset.to_numpy()
    attributes = dataset[:, 0:5]                                         # Columns 1-5 contain our features
    labels = dataset[:, 5]                                               # Column 6 contains our class labels
    random_forest_model = RandomForest(attributes, labels)
    random_forest_model.run_classifier()                                 # This will trigger the command-line interface for parameter input

    new_data_X = read_csv('/path/to/new_data_X.csv')
    new_data_X = new_data_X.to_numpy()
    classifications = random_forest_model.predict_classifier(new_data_X) # This will return and output classifications for new_data_X

----------

.. _regression:

Regression
==========

Regressor Parameters
--------------------

The following parameter can be modified when run_regressor() is called:

- **criterion**: *('mse' or 'mae', default='mse')*: Determines the metric for calculating split quality. 'mse' is for mean squared error, and 'mae' is for mean absolute error.

Regressor Attributes
--------------------

After run_regressor() successfully trains the regressor model, the following instance data is available:

- **regressor** *(model)*: The underlying random forest regressor model.
- **mean_squared_error** *(float)*: The average squared differences between the estimated and actual values of the test dataset.
- **r_score** *(float)*: The correlation coefficient for the regressor model.
- **r2_score** *(float)*: The coefficient of determination for the regressor model.
- **cross_val_scores_regressor** *(array of floats)*: An array of the cross validation scores for the regressor model.
- **feature_importances_regressor** *(array of floats)*: An array of the feature importances for the regressor model. The higher the score, the more useful the feature is for prediction.

Regressor Methods
-----------------

- **run_regressor()**: Prompts the user for the model parameters and trains a random forest regressor model using attributes and labels. If successful, the regressor instance data is updated, and the model metrics are displayed.
- **predict_regressor(dataset_X=None)**: Uses the random forest regressor model to make predictions for the features in dataset_X. If successful, the predictions are displayed and returned. predict_regressor() can only be called after run_regressor() has successfully trained the regressor.

Regressor Accessor Methods
**************************

- **get_regressor()**: Returns regressor.
- **get_mean_squared_error()**: Returns mean_squared_error.
- **get_r_score()**: Returns r_score.
- **get_r2_score()**: Returns r2_score.
- **get_cross_val_scores_regressor()**: Returns get_cross_val_scores_regressor.
- **get_feature_importances_regressor()**: Returns feature_importances_regressor.

Note: If run_regressor() hasn't successfully executed yet, the above accessor methods will return None.

Regressor Example Usage
-----------------------

.. code-block:: python
    :linenos:

    from ManufacturingNet.models import RandomForest
    from pandas import read_csv

    dataset = read_csv('/path/to/dataset.csv')
    dataset = dataset.to_numpy()
    attributes = dataset[:, 0:5]                                     # Columns 1-5 contain our features
    labels = dataset[:, 5]                                           # Column 6 contains our dependent variable
    random_forest_model = RandomForest(attributes, labels)
    random_forest_model.run_regressor()                              # This will trigger the command-line interface for parameter input

    new_data_X = read_csv('/path/to/new_data_X.csv')
    new_data_X = new_data_X.to_numpy()
    predictions = random_forest_model.predict_regressor(new_data_X)  # This will return and output predictions for new_data_X