**********************
Support Vector Machine
**********************

A support vector machine (SVM) model is a set of supervised learning methods that excel with high-dimensional data. SVM
models support multiple kernels, and each decision function utilizes a subset of training points called support vectors.

ManufacturingNet supports multiple SVM models:

- SVC, NuSVC, and LinearSVC for classification, and
- SVR, NuSVR, and LinearSVR for regression.

SVC and NuSVC are similar models, differing in only one parameter; the same applies to SVR and NuSVR. LinearSVC is a
faster implementation of SVC with the linear kernel; the same applies to LinearSVR.

ManufacturingNet's SVM functionality is provided through the **SVM** class.

*SVM(attributes=None, labels=None)*

Go To
=====

    - :ref:`SVC`
    - :ref:`nu-SVC`
    - :ref:`linear-SVC`
    - :ref:`SVR`
    - :ref:`nu-SVR`
    - :ref:`linear-SVR`

Universal Parameters
====================

When initializing a SVM object, the following parameters need to be passed:

- **attributes** *(numpy array, default=None)*: A numpy array of the values of the independent variable(s).
- **labels** *(numpy array, default=None)*: A numpy array of the values of the dependent variable or the class labels.

When one of the model runner methods is called, the following parameters can be modified:

- **test_size** *(float, default=0.25)*: The proportion of the dataset to be used for testing the model; the proportion of the dataset to be used for training will be the complement of test_size.
- **cv** *(integer, default=None)*: The number of folds to use for cross validation.

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

Classification
==============

.. _SVC:

SVC
===

SVC Parameters
--------------

The following parameters can be modified when run_SVC() is called:

- **graph_results** *(boolean, default=False)*: Determines whether to plot the ROC curve. Supported for binary classification only.
- **C** *(float, default=1.0)*: Positive number that specifies the inverse of the regularization strength.
- **kernel** *('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'; default='rbf')*: Determines which kernel to use.
- **degree** *(integer, default=3)*: If kernel is 'poly', the polynomial function is of this degree.
- **gamma** *('scale' or 'auto', default='scale')*: If kernel is 'rbf', 'poly', or 'sigmoid', gamma specifies the kernel coefficient. If 'scale', gamma is equal to 1 / (n_features * X.var()). If 'auto', gamma is equal to 1 / n_features.
- **coef0** *(float, default=0.0)*: The kernel's independent term. Only useful if kernel is 'poly' or 'sigmoid'.
- **shrinking** *(boolean, default=True)*: Determines if the shrinking heuristic is used.
- **probability** *(boolean, default=False)*: Determines if probability estimates are calculated.
- **tol** *(float, default=0.001)*: The acceptable margin of error for stopping criteria.
- **cache_size** *(float, default=200)*: The kernel cache size in megabytes.
- **class_weight** *(boolean, default=False)*: Determines whether to automatically balance the class weights.
- **max_iter** *(integer, default=-1)*: Sets the maximum number of iterations the solver can take to converge. If -1, no maximum is set.
- **decision_function_shape** *('ovo' or 'ovr', default='ovr')*: Determines the decision function. If 'ovr', a one-vs-rest decision function is used. If 'ovo', a one-vs-one decision function is used.
- **break_ties** *(boolean, default=False)*: Determines whether to break ties using the decision function's confidence values.
- **random_state** *(integer, default=None)*: The seed for random number generation.
- **verbose** *(boolean, default=False)*: Determines whether to output logs when fitting and predicting.

SVC Attributes
--------------

After run_SVC() successfully trains the SVC model, the following instance data is available:

- **classifier_SVC** *(model)*: The underlying SVC model.
- **accuracy_SVC** *(float)*: The SVC model's classification accuracy score.
- **roc_auc_SVC** *(float)*: The area under the receiver operating characteristic (ROC) curve from the SVC prediction scores. Supported for binary classification only.
- **confusion_matrix_SVC** *(2D array of integers)*: A matrix where the entry in the *i* th row and *j* th column is the number of observations present in group *i* and predicted to be in group *j*. Supported for multilabel classification only.
- **cross_val_scores_SVC** *(array of floats)*: An array of the cross validation scores for the SVC model.

SVC Methods
-----------

- **run_SVC()**: Prompts the user for the model parameters and trains a SVC model using attributes and labels. If successful, the SVC instance data is updated, and the model metrics are displayed. If the model is being used for binary classification, the ROC curve will be graphed and displayed.
- **predict_SVC(dataset_X=None)**: Uses the SVC model to classify the observations in dataset_X. If successful, the classifications are displayed and returned. predict_SVC() can only be called after run_SVC() has successfully trained the classifier.

SVC Accessor Methods
********************

- **get_classifier_SVC()**: Returns classifier_SVC.
- **get_accuracy_SVC()**: Returns accuracy_SVC.
- **get_roc_auc_SVC()**: Returns roc_auc_SVC.
- **get_confusion_matrix_SVC()**: Returns confusion_matrix_SVC.
- **get_cross_val_scores_SVC()**: Returns cross_val_scores_SVC.

Note: If run_SVC() hasn't successfully executed yet, the above accessor methods will return None.

.. _nu-SVC:

NuSVC
=====

NuSVC Parameters
----------------

The following parameters can be modified when run_nu_SVC() is called:

- **graph_results** *(boolean, default=False)*: Determines whether to plot the ROC curve. Supported for binary classification only.
- **nu** *(float, default=0.5)*: A decimal for the maximum fraction of margin errors and the minimum fraction of support vectors. Should be greater than 0 and less than or equal to 1.
- **kernel** *('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'; default='rbf')*: Determines which kernel to use.
- **degree** *(integer, default=3)*: If kernel is 'poly', the polynomial function is of this degree.
- **gamma** *('scale' or 'auto', default='scale')*: If kernel is 'rbf', 'poly', or 'sigmoid', gamma specifies the kernel coefficient.
- **coef0** *(float, default=0.0)*: The kernel's independent term. Only useful if kernel is 'poly' or 'sigmoid'.
- **shrinking** *(boolean, default=True)*: Determines if the shrinking heuristic is used.
- **probability** *(boolean, default=False)*: Determines if probability estimates are calculated.
- **tol** *(float, default=0.001)*: The acceptable margin of error for stopping criteria.
- **cache_size** *(float, default=200)*: The kernel cache size in megabytes.
- **class_weight** *(boolean, default=False)*: Determines whether to automatically balance the class weights.
- **max_iter** *(integer, default=-1)*: Sets the maximum number of iterations the solver can take to converge. If -1, no maximum is set.
- **decision_function_shape** *('ovo' or 'ovr', default='ovr')*: Determines the decision function. If 'ovr', a one-vs-rest decision function is used. If 'ovo', a one-vs-one decision function is used.
- **break_ties** *(boolean, default=False)*: Determines whether to break ties using the decision function's confidence values.
- **random_state** *(integer, default=None)*: The seed for random number generation.
- **verbose** *(boolean, default=False)*: Determines whether to output logs when fitting and predicting.

NuSVC Attributes
----------------

After run_nu_SVC() successfully trains the NuSVC model, the following instance data is available:

- **classifier_nu_SVC** *(model)*: The underlying NuSVC model.
- **accuracy_nu_SVC** *(float)*: The NuSVC model's classification accuracy score.
- **roc_auc_nu_SVC** *(float)*: The area under the receiver operating characteristic (ROC) curve from the NuSVC prediction scores. Supported for binary classification only.
- **confusion_matrix_nu_SVC** *(2D array of integers)*: A matrix where the entry in the *i* th row and *j* th column is the number of observations present in group *i* and predicted to be in group *j*. Supported for multilabel classification only.
- **cross_val_scores_nu_SVC** *(array of floats)*: An array of the cross validation scores for the NuSVC model.

NuSVC Methods
-------------

- **run_nu_SVC()**: Prompts the user for the model parameters and trains a NuSVC model using attributes and labels. If successful, the NuSVC instance data is updated, and the model metrics are displayed. If the model is being used for binary classification, the ROC curve will be graphed and displayed.
- **predict_nu_SVC(dataset_X=None)**: Uses the NuSVC model to classify the observations in dataset_X. If successful, the classifications are displayed and returned. predict_nu_SVC() can only be called after run_nu_SVC() has successfully trained the classifier.

NuSVC Accessor Methods
**********************

- **get_classifier_nu_SVC()**: Returns classifier_nu_SVC.
- **get_accuracy_nu_SVC()**: Returns accuracy_nu_SVC.
- **get_roc_auc_nu_SVC()**: Returns roc_auc_nu_SVC.
- **get_confusion_matrix_nu_SVC()**: Returns confusion_matrix_nu_SVC.
- **get_cross_val_scores_nu_SVC()**: Returns cross_val_scores_nu_SVC.

Note: If run_nu_SVC() hasn't successfully executed yet, the above accessor methods will return None.

.. _linear-SVC:

LinearSVC
=========

LinearSVC Parameters
--------------------

The following parameters can be modified when run_linear_SVC() is called:

- **penalty** *('l1' or 'l2', default='l2')*: The penalization norm. 'l2' is standard for SVC models.
- **loss** *('hinge' or 'squared_hinge', default='squared_hinge')*: The loss function. 'hinge' is standard for SVM models, while 'squared_hinge' is the hinge loss squared.
- **dual** *(boolean, default=True)*: Determines whether to solve the dual or primal optimization problem.
- **tol** *(float, default=0.0001)*: The acceptable margin of error for stopping criteria.
- **C** *(float, default=1.0)*: Positive number that specifies the inverse of the regularization strength.
- **multi_class** *('ovr' or 'crammer_singer', default='ovr')*: Chooses whether to fit a binary problem or a multi-class problem for each label. Binary problems use 'ovr', while multi-class problems use 'crammer_singer'.
- **fit_intercept** *(boolean, default=True)*: Determines whether to calculate an intercept for the decision function.
- **intercept_scaling** *(float, default=1)*: If fit_intercept is True, each instance vector gains a feature with a value of intercept_scaling.
- **class_weight** *(boolean, default=False)*: Determines whether to automatically balance the class weights using class frequencies.
- **random_state** *(integer, default=None)*: The seed for random number generation.
- **max_iter** *(integer, default=1000)*: Sets the maximum number of iterations the solver can take to converge. If -1, no maximum is set.
- **verbose** *(boolean, default=False)*: Determines whether to output logs when fitting and predicting.

LinearSVC Attributes
--------------------

After run_linear_SVC() successfully trains the LinearSVC model, the following instance data is available:

- **classifier_linear_SVC** *(model)*: The underlying LinearSVC model.
- **accuracy_linear_SVC** *(float)*: The LinearSVC model's classification accuracy score.
- **cross_val_scores_linear_SVC** *(array of floats)*: An array of the cross validation scores for the LinearSVC model.

LinearSVC Methods
-----------------

- **run_linear_SVC()**: Prompts the user for the model parameters and trains a LinearSVC model using attributes and labels. If successful, the LinearSVC instance data is updated, and the model metrics are displayed.
- **predict_linear_SVC(dataset_X=None)**: Uses the LinearSVC model to classify the observations in dataset_X. If successful, the classifications are displayed and returned. predict_linear_SVC() can only be called after run_linear_SVC() has successfully trained the classifier.

LinearSVC Accessor Methods
**************************

- **get_classifier_linear_SVC()**: Returns classifier_linear_SVC.
- **get_accuracy_linear_SVC()**: Returns accuracy_linear_SVC.
- **get_cross_val_scores_linear_SVC()**: Returns cross_val_scores_linear_SVC.

Note: If run_linear_SVC() hasn't successfully executed yet, the above accessor methods will return None.

Classification Example Usage
----------------------------

.. code-block:: python
    :linenos:

    from ManufacturingNet.models import SVM
    from pandas import read_csv

    dataset = read_csv('/path/to/dataset.csv')
    dataset = dataset.to_numpy()
    attributes = dataset[:, 0:5]    # Columns 1-5 contain our features
    labels = dataset[:, 5]          # Column 6 contains our class labels
    SVM_model = SVM(attributes, labels)
    
    # These calls will trigger the command-line interfaces for SVC, NuSVC, and LinearSVC parameter input
    SVM_model.run_SVC()
    SVM_model.run_nu_SVC()
    SVM_model.run_linear_SVC()

    new_data_X = read_csv('/path/to/new_data_X.csv')
    new_data_X = new_data_X.to_numpy()

    # These calls will return and output classifications for new_data_X made by SVC, NuSVC, and LinearSVC
    classifications_SVC = SVM_model.predict_SVC(new_data_X)
    classifications_nu_SVC = SVM_model.predict_nu_SVC(new_data_X)
    classifications_linear_SVC = SVM_model.predict_linear_SVC(new_data_X)

----------

Regression
==========

.. _SVR:

SVR
===

SVR Parameters
--------------

The following parameters can be modified when run_SVR() is called:

- **epsilon** *(float, default=0.1)*: The maximum difference between predictions and actual values for which penalties aren't applied.
- **kernel** *('linear', 'poly', 'rbf', 'sigmoid', or 'precomputed'; default='rbf')*: Determines which kernel to use.
- **degree** *(integer, default=3)*: If kernel is 'poly', the polynomial function is of this degree.
- **gamma** *('scale' or 'auto', default='scale')*: If kernel is 'rbf', 'poly', or 'sigmoid', gamma specifies the kernel coefficient. If 'scale', gamma is equal to 1 / (n_features * X.var()). If 'auto', gamma is equal to 1 / n_features.
- **coef0** *(float, default=0.0)*: The kernel's independent term. Only useful if kernel is 'poly' or 'sigmoid'.
- **tol** *(float, default=0.001)*: The acceptable margin of error for stopping criteria.
- **C** *(float, default=1.0)*: Positive number that specifies the inverse of the regularization strength.
- **shrinking** *(boolean, default=True)*: Determines if the shrinking heuristic is used.
- **cache_size** *(float, default=200)*: The kernel cache size in megabytes.
- **max_iter** *(integer, default=-1)*: Sets the maximum number of iterations the solver can take to converge. If -1, no maximum is set.
- **verbose** *(boolean, default=False)*: Determines whether to output logs when fitting and predicting.

SVR Attributes
--------------

After run_SVR() successfully trains the SVR model, the following instance data is available:

- **regressor_SVR** *(model)*: The underlying SVR model.
- **mean_squared_error_SVR** *(float)*: The average squared differences between the estimated and actual values of the test dataset for the SVR model.
- **r_score_SVR** *(float)*: The correlation coefficient for the SVR model.
- **r2_score_SVR** *(float)*: The coefficient of determination for the SVR model.
- **cross_val_scores_SVR** *(array of floats)*: An array of the cross validation scores for the SVR model.

SVR Methods
-----------

- **run_SVR()**: Prompts the user for the SVR model parameters and trains a SVR model using attributes and labels. If successful, the SVR instance data is updated, and the model metrics are displayed.
- **predict_SVR(dataset_x=None)**: Uses the SVR model to make predictions for the features in dataset_X. If successful, the predictions are displayed and returned. predict_SVR() can only be called after run_SVR() has successfully trained the SVR model.

SVR Accessor Methods
********************

- **get_regressor_SVR()**: Returns regressor_SVR.
- **get_mean_squared_error_SVR()**: Returns mean_squared_error_SVR.
- **get_r_score_SVR()**: Returns r_score_SVR.
- **get_r2_score_SVR()**: Returns r2_score_SVR.
- **get_cross_val_scores_SVR()**: Returns cross_val_scores_SVR.

Note: If run_SVR() hasn't successfully executed yet, the above accessor methods will return None.

.. _nu-SVR:

NuSVR
=====

NuSVR Parameters
----------------

The following parameters can be modified when run_nu_SVR() is called:

- **nu** *(float, default=0.5)*: A decimal for the maximum fraction of margin errors and the minimum fraction of support vectors. Should be greater than 0 and less than or equal to 1.
- **kernel** *('linear', 'poly', 'rbf', 'sigmoid', or 'precomputed'; default='rbf')*: Determines which kernel to use.
- **degree** *(integer, default=3)*: If kernel is 'poly', the polynomial function is of this degree.
- **gamma** *('scale' or 'auto', default='scale')*: If kernel is 'rbf', 'poly', or 'sigmoid', gamma specifies the kernel coefficient. If 'scale', gamma is equal to 1 / (n_features * X.var()). If 'auto', gamma is equal to 1 / n_features.
- **coef0** *(float, default=0.0)*: The kernel's independent term. Only useful if kernel is 'poly' or 'sigmoid'.
- **tol** *(float, default=0.001)*: The acceptable margin of error for stopping criteria.
- **C** *(float, default=1.0)*: Positive number that specifies the inverse of the regularization strength.
- **shrinking** *(boolean, default=True)*: Determines if the shrinking heuristic is used.
- **cache_size** *(float, default=200)*: The kernel cache size in megabytes.
- **max_iter** *(integer, default=-1)*: Sets the maximum number of iterations the solver can take to converge. If -1, no maximum is set.
- **verbose** *(boolean, default=False)*: Determines whether to output logs when fitting and predicting.

NuSVR Attributes
----------------

After run_nu_SVR() successfully trains the NuSVR model, the following instance data is available:

- **regressor_nu_SVR** *(model)*: The underlying NuSVR model.
- **mean_squared_error_nu_SVR** *(float)*: The average squared differences between the estimated and actual values of the test dataset for the NuSVR model.
- **r_score_nu_SVR** *(float)*: The correlation coefficient for the NuSVR model.
- **r2_score_nu_SVR** *(float)*: The coefficient of determination for the NuSVR model.
- **cross_val_scores_nu_SVR** *(array of floats)*: An array of the cross validation scores for the NuSVR model.

NuSVR Methods
-------------

- **run_nu_SVR()**: Prompts the user for the NuSVR model parameters and trains a NuSVR model using attributes and labels. If successful, the NuSVR instance data is updated, and the model metrics are displayed.
- **predict_nu_SVR(dataset_x=None)**: Uses the NuSVR model to make predictions for the features in dataset_X. If successful, the predictions are displayed and returned. predict_nu_SVR() can only be called after run_nu_SVR() has successfully trained the NuSVR model.

NuSVR Accessor Methods
**********************

- **get_regressor_nu_SVR()**: Returns regressor_nu_SVR.
- **get_mean_squared_error_nu_SVR()**: Returns mean_squared_error_nu_SVR.
- **get_r_score_nu_SVR()**: Returns r_score_nu_SVR.
- **get_r2_score_nu_SVR()**: Returns r2_score_nu_SVR.
- **get_cross_val_scores_nu_SVR()**: Returns cross_val_scores_nu_SVR.

Note: If run_nu_SVR() hasn't successfully executed yet, the above accessor methods will return None.

.. _linear-SVR:

LinearSVR
=========

LinearSVR Parameters
--------------------

The following parameters can be modified when run_linear_SVR() is called:

- **epsilon** *(float, default=0.0)*: The maximum difference between predictions and actual values for which penalties aren't applied.
- **tol** *(float, default=0.0001)*: The acceptable margin of error for stopping criteria.
- **C** *(float, default=1.0)*: Positive number that specifies the inverse of the regularization strength.
- **loss** *('epsilon_insensitive' or 'squared_epsilon_insensitive', default='epsilon_insensitive')*: The loss function. 'epsilon_insensitive' is the L1 loss, and 'squared_epsilon_insensitive' is the L2 loss.
- **fit_intercept** *(boolean, default=True)*: Determines whether to calculate an intercept for the decision function.
- **intercept_scaling** *(float, default=1)*: If fit_intercept is True, each instance vector gains a feature with a value of intercept_scaling.
- **dual** *(boolean, default=True)*: Determines whether to solve the dual or primal optimization problem.
- **random_state** *(integer, default=None)*: The seed for random number generation.
- **max_iter** *(integer, default=1000)*: Sets the maximum number of iterations the solver can take to converge. If -1, no maximum is set.
- **verbose** *(boolean, default=False)*: Determines whether to output logs when fitting and predicting.

LinearSVR Attributes
--------------------

After run_linear_SVR() successfully trains the LinearSVR model, the following instance data is available:

- **regressor_linear_SVR** *(model)*: The underlying LinearSVR model.
- **mean_squared_error_linear_SVR** *(float)*: The average squared differences between the estimated and actual values of the test dataset for the LinearSVR model.
- **r_score_linear_SVR** *(float)*: The correlation coefficient for the LinearSVR model.
- **r2_score_linear_SVR** *(float)*: The coefficient of determination for the LinearSVR model.
- **cross_val_scores_linear_SVR** *(array of floats)*: An array of the cross validation scores for the LinearSVR model.

LinearSVR Methods
-----------------

- **run_linear_SVR()**: Prompts the user for the LinearSVR model parameters and trains a LinearSVR model using attributes and labels. If successful, the LinearSVR instance data is updated, and the model metrics are displayed.
- **predict_linear_SVR(dataset_x=None)**: Uses the LinearSVR model to make predictions for the features in dataset_X. If successful, the predictions are displayed and returned. predict_linear_SVR() can only be called after run_linear_SVR() has successfully trained the LinearSVR model.

LinearSVR Accessor Methods
**************************

- **get_regressor_linear_SVR()**: Returns regressor_linear_SVR.
- **get_mean_squared_error_linear_SVR()**: Returns mean_squared_error_linear_SVR.
- **get_r_score_linear_SVR()**: Returns r_score_linear_SVR.
- **get_r2_score_linear_SVR()**: Returns r2_score_linear_SVR.
- **get_cross_val_scores_linear_SVR()**: Returns cross_val_scores_linear_SVR.

Note: If run_linear_SVR() hasn't successfully executed yet, the above accessor methods will return None.

Regression Example Usage
------------------------

.. code-block:: python
    :linenos:

    from ManufacturingNet.models import SVM
    from pandas import read_csv

    dataset = read_csv('/path/to/dataset.csv')
    dataset = dataset.to_numpy()
    attributes = dataset[:, 0:5]    # Columns 1-5 contain our features
    labels = dataset[:, 5]          # Column 6 contains our dependent variable
    SVM_model = SVM(attributes, labels)
    
    # These calls will trigger the command-line interfaces for SVR, NuSVR, and LinearSVR parameter input
    SVM_model.run_SVR()
    SVM_model.run_nu_SVR()
    SVM_model.run_linear_SVR()

    new_data_X = read_csv('/path/to/new_data_X.csv')
    new_data_X = new_data_X.to_numpy()

    # These calls will return and output predictions for new_data_X made by SVR, NuSVR, and LinearSVR
    predictions_SVR = SVM_model.predict_SVR(new_data_X)
    predictions_nu_SVR = SVM_model.predict_nu_SVR(new_data_X)
    predictions_linear_SVR = SVM_model.predict_linear_SVR(new_data_X)