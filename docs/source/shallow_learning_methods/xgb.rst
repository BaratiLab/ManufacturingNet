*******
XGBoost
*******

XGBoost (eXtreme Gradient Boosting) is a machine learning library that utilizes gradient boosting to provide fast
parallel tree boosting. As such, XGBoost provides fast and accurate classification and regression functionality that
scales for large datasets.

ManufacturingNet's XGBoost functionality is provided through the **XGBoost** class.

*XGBoost(attributes=None, labels=None)*

Go To
=====

    - :ref:`class`
    - :ref:`reg`

Universal Parameters
====================

When initializing a XGBoost object, the following parameters need to be passed:

- **attributes** *(numpy array, default=None)*: A numpy array of the values of the independent variable(s).
- **labels** *(numpy array, default=None)*: A numpy array of the values of the dependent variable or the class labels.

When run_classifier() or run_regressor() is called, the following parameters can be modified:

- **test_size** *(float, default=0.25)*: The proportion of the dataset to be used for testing the model; the proportion of the dataset to be used for training will be the complement of test_size.
- **cv** *(integer, default=None)*: The number of folds to use for cross validation.
- **n_estimators** *(integer, default=100)*: The number of decision trees.
- **max_depth** *(integer, default=3)*: The maximum depth of each tree.
- **learning_rate** *(float, default=0.1)*: Learning rate for the booster.
- **booster** *('gbtree', 'gblinear', or 'dart'; default='gbtree')*: The booster function.
- **n_jobs** *(integer, default=1)*: The number of parallel jobs to use during model training.
- **nthread** *(integer, default=None)*: The number of parallel jobs to use for data loading, if available.
- **gamma** *(float, default=0)*: The minimum reduction in loss needed to partition a leaf node.
- **min_child_weight** *(float, default=1)*: The minimum child weight allowed.
- **max_delta_step** *(integer, default=0)*: The maximum delta step allowed for tree weight estimation.
- **subsample** *(float, default=1)*: The training instance's subsample ratio.
- **colsample_bytree** *(float, default=1)*: The subsample column ratio for all trees.
- **colsample_bylevel** *(float, default=1)*: The subsample column ratio for all levels.
- **reg_alpha** *(float, default=0)*: The L1 regularization term.
- **reg_lambda** *(float, default=1)*: The L2 regularization term.
- **scale_pos_weight** *(float, default=1)*: The multiplier for balancing weights of opposite signs.
- **base_score** *(float, default=0.5)*: The initial prediction score for every run.
- **random_state** *(integer, default=42)*: The seed for random number generation.
- **missing** *(float, default=0)*: The value to represent missing data.
- **verbose** *(boolean, default=False)*: Determines whether to output logs during model training and testing.

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

.. _class:

Classification
==============

Classifier Parameters
---------------------

When run_classifier() is called, the following parameter can be modified:

- **graph_results** *(boolean, default=False)*: Determines whether to plot the ROC curve. Supported for binary classification only.

Classifier Attributes
---------------------

After run_classifier() successfully trains the classification model, the following instance data is available:

- **classifier** *(model)*: The underlying XGBoost classifier model.
- **precision_scores** *(array of floats)*: An array of the precision scores for all class labels.
- **recall_scores** *(array of floats)*: An array of the recall scores for all class labels.
- **accuracy** *(float)*: The classification accuracy score.
- **confusion_matrix** *(2D array of integers)*: A matrix where the entry in the *i* th row and *j* th column is the number of observations present in group *i* and predicted to be in group *j*. Supported for multilabel classification only.
- **roc_auc** *(float)*: The area under the receiver operating characteristic (ROC) curve from the prediction scores. Supported for binary classification only.
- **classes** *(array of multiple possible types)*: An array of the known class labels.
- **cross_val_scores_classifier** *(array of floats)*: An array of the cross validation scores for the classifier model.
- **feature_importances_classifier** *(array of floats)*: An array of the feature importances for the classifier model. The higher the score, the more useful the feature is for prediction.

Classifier Methods
------------------

- **run_classifier()**: Prompts the user for the model parameters and trains a XGBoost classifier model using attributes and labels. If successful, the classifier instance data is updated, and the model metrics are displayed. If the model is being used for binary classification, the ROC curve will be graphed and displayed.
- **predict_classifier(dataset_X=None)**: Uses the XGBoost classifier model to classify the observations in dataset_X. If successful, the classifications are displayed and returned. predict_classifier() can only be called after run_classifier() has successfully trained the classifier.

Classifier Accessor Methods
***************************

- **get_classifier()**: Returns classifier.
- **get_precision_scores()**: Returns precision_scores.
- **get_precision(label=None)**: Returns the precision score for the specified label.
- **get_recall_scores()**: Returns recall_scores.
- **get_recall(label=None)**: Returns the recall score for the specified label.
- **get_accuracy()**: Returns accuracy.
- **get_confusion_matrix()**: Returns confusion_matrix.
- **get_roc_auc()**: Returns roc_auc.
- **get_classes()**: Returns classes.
- **get_cross_val_scores_classifier()**: Returns cross_val_scores_classifier.
- **get_feature_importances_classifier()**: Returns feature_importances_classifier.

Note: If run_classifier() hasn't successfully executed yet, the above accessor methods will return None.

Classifier Example Usage
------------------------

.. code-block:: python
    :linenos:

    from ManufacturingNet.models import XGBoost
    from pandas import read_csv

    dataset = read_csv('/path/to/dataset.csv')
    dataset = dataset.to_numpy()
    attributes = dataset[:, 0:5]                               # Columns 1-5 contain our features
    labels = dataset[:, 5]                                     # Column 6 contains our class labels
    xgb_model = XGBoost(attributes, labels)
    xgb_model.run_classifier()                                 # This will trigger the command-line interface for parameter input

    new_data_X = read_csv('/path/to/new_data_X.csv')
    new_data_X = new_data_X.to_numpy()
    classifications = xgb_model.predict_classifier(new_data_X) # This will return and output classifications for new_data_X

----------

.. _reg:

Regression
==========

Regressor Attributes
--------------------

After run_regressor() successfully trains the classification model, the following instance data is available:

- **regressor** *(model)*: The underlying XGBoost regressor model.
- **mean_squared_error** *(float)*: The average squared differences between the estimated and actual values of the test dataset.
- **r_score** *(float)*: The correlation coefficient for the regressor model.
- **r2_score** *(float)*: The coefficient of determination for the regressor model.
- **cross_val_scores_regressor** *(array of floats)*: An array of the cross validation scores for the regressor model.
- **feature_importances_regressor** *(array of floats)*: An array of the feature importances for the regressor model. The higher the score, the more useful the feature is for prediction.

Regressor Methods
-----------------

- **run_regressor()**: Prompts the user for the model parameters and trains a XGBoost regressor model using attributes and labels. If successful, the regressor instance data is updated, and the model metrics are displayed.
- **predict_regressor(dataset_X=None)**: Uses the XGBoost regressor model to make predictions for the features in dataset_X. If successful, the predictions are displayed and returned. predict_regressor() can only be called after run_regressor() has successfully trained the regressor.

Regressor Accessor Methods
**************************

- **get_regressor()**: Returns regressor.
- **get_mean_squared_error()**: Returns mean_squared_error.
- **get_r_score()**: Returns r_score.
- **get_r2_score()**: Returns r2_score.
- **get_cross_val_scores_regressor()**: Returns cross_val_scores_regressor.
- **get_feature_importances_regressor()**: Returns feature_importances_regressor.

Regressor Example Usage
-----------------------

.. code-block:: python
    :linenos:

    from ManufacturingNet.models import XGBoost
    from pandas import read_csv

    dataset = read_csv('/path/to/dataset.csv')
    dataset = dataset.to_numpy()
    attributes = dataset[:, 0:5]                           # Columns 1-5 contain our features
    labels = dataset[:, 5]                                 # Column 6 contains our dependent variable
    xgb_model = XGBoost(attributes, labels)
    xgb_model.run_regressor()                              # This will trigger the command-line interface for parameter input

    new_data_X = read_csv('/path/to/new_data_X.csv')
    new_data_X = new_data_X.to_numpy()
    predictions = xgb_model.predict_regressor(new_data_X)  # This will return and output predictions for new_data_X