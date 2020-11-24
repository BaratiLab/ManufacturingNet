*****************
Linear Regression
*****************

A linear regression model identifies a linear relationship between one or more independent variables/features and a single
dependent variable by fitting a line of best fit to the data. The line of best fit is a linear function where the sum
of all residuals (or the distances between each datapoint and the line) is minimized. The equation of the line of best
fit has the following form:

:math:`y = C_1x_1 + C_2x_2 + ... + C_nx_n`

where :math:`y` is the dependent variable, :math:`x_i` is one of *n* features, and :math:`C_i` is the corresponding
coefficient. It is the model's job to find these coefficients.

ManufacturingNet's linear regression functionality is provided through the **LinRegression** class.

*LinRegression(attributes=None, labels=None)*

Parameters
==========

When initializing a LinRegression object, the following parameters need to be passed:

- **attributes** *(numpy array, default=None)*: A numpy array of the values of the independent variable(s).
- **labels** *(numpy array, default=None)*: A numpy array of the values of the dependent variable.

When the run() method is called, the following parameters can be modified:

- **test_size** *(float, default=0.25)*: The proportion of the dataset to be used for testing the model; the proportion of the dataset to be used for training will be the complement of test_size.
- **cv** *(integer, default=None)*: The number of folds to use for cross validation.
- **graph_results** *(boolean, default=False)*: Determines whether to graph the line of best fit and the test dataset; this feature only works for univariate regression.
- **fit_intercept** *(boolean, default=True)*: Determines whether to calculate a y-intercept for the model.
- **normalize** *(boolean, default=False)*: Determines whether to normalize/standardize the data.
- **copy_X** *(boolean, default=True)*: Determines whether to copy the dataset's features.
- **n_jobs** *(integer, default=None)*: The number of jobs to use for computation.

Attributes
==========

After run() successfully trains the model, the following instance data is available:

- **regression** *(model)*: The underlying linear regression model.
- **coefficients** *(array of floats)*: An array of the coefficients from the line of best fit equation.
- **intercept** *(float)*: The y-intercept of the line of best fit, if it has one.
- **mean_squared_error** *(float)*: The average squared differences between the estimated and actual values of the test dataset.
- **r_score** *(float)*: The correlation coefficient for the linear model.
- **r2_score** *(float)*: The coefficient of determination for the linear model.
- **cross_val_scores** *(array of floats)*: An array of the cross validation scores for the model.

Methods
=======

- **run()**: Prompts the user for the model parameters and trains a linear regression model using attributes and labels. If successful, the above instance data is updated, and the model metrics are displayed.
- **predict(dataset_X=None)**: Uses the linear regression model to make predictions for the features in dataset_X. If successful, the predictions are displayed and returned. predict() can only be called after run() has successfully trained the model.

Accessor Methods
----------------

- **get_attributes()**: Returns attributes.
- **get_labels()**: Returns labels.

Note: If attributes wasn't passed in during initialization, get_attributes() will return None. Likewise, if labels
wasn't passed in during initialization, get_labels() will return None.

- **get_regression()**: Returns regression.
- **get_coefficients()**: Returns coefficients.
- **get_intercept()**: Returns intercept.
- **get_mean_squared_error()**: Returns mean_squared_error.
- **get_r_score()**: Returns r_score.
- **get_r2_score()**: Returns r2_score.
- **get_cross_val_scores()**: Returns cross_val_scores.

Note: If run() hasn't successfully executed yet, the above accessor methods will return None.

Modifier Methods
----------------

- **set_attributes(new_attributes=None)**: Sets attributes to new_attributes. If new_attributes isn't specified, attributes is set to None.
- **set_labels(new_labels=None)**: Sets labels to new_labels. If new_labels isn't specified, labels is set to None.

Example Usage
=============

.. code-block:: python
    :linenos:

    from ManufacturingNet.models import LinRegression
    from pandas import read_csv

    dataset = read_csv('/path/to/dataset.csv')
    dataset = dataset.to_numpy()
    attributes = dataset[:, 0:5]                    # Columns 1-5 contain our features
    labels = dataset[:, 5]                          # Column 6 contains our dependent variable
    linear_model = LinRegression(attributes, labels)
    linear_model.run()                              # This will trigger the command-line interface for parameter input

    new_data_X = read_csv('/path/to/new_data_X.csv')
    new_data_X = new_data_X.to_numpy()
    predictions = linear_model.predict(new_data_X)  # This will return and output predictions for new_data_X