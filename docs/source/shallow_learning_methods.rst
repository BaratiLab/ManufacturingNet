************************
Shallow Learning Methods
************************

ManufacturingNet provides several shallow machine learning algorithms for performing supervised regression and
classification on your data.

Unlike "deep" learning, "shallow" learning cannot extract features from raw data; to use these algorithms, your features
must be predefined in your dataset. If you need automatic feature extraction (and your dataset is sufficiently massive),
check out our deep learning methods!

Shallow learning methods typically follow the below pattern:

- Split the dataset into a training and testing set
- Generate weights for the features by learning from the training set (this creates the model)
- Run the model against the testing set to determine its accuracy

At this point, it is your responsibility to either accept the model, tweak its parameters for better accuracy, or try a
different learning method altogether. Don't worry; our shallow learning library makes this easy.

Classification vs. Regression
=============================

Many shallow learning methods support both classification and regression analysis.

In a classification model, the features of a datapoint are weighted to predict one or several categories it falls into.
For example, using the characteristics of a tumor to predict if it is benign or malignant is a classification problem.

- In binary classification, datapoints are classified into one of two categories.
- In multiclass classification, datapoints are classified into three or more categories.
- In multi-label classification, datapoints can be classified into several categories.

Confusingly, multi-label classification is not the same as multiclass classification; try not to mix these up!

In a regression model, the features of a datapoint are weighted to predict a continuous output. For example, using the
characteristics of a house to predict its price is a regression problem.

A regression model uses the training dataset to generate a mathematical function of best fit; this function is linear
or non-linear, depending on the model used. To predict outputs for new datapoints, the model simply inputs the
datapoint's feature(s) into the function and calculates the result.

As a general rule, if your dataset's dependent variables are **not continuous**, you should use classification. For
example, if your dependent variables are strings or numerical categories, use a classification model.

Classification Methods
======================

    .. toctree::
            :maxdepth: 1

            Logistic Regression <shallow_learning_methods/logistic_regression>
            Random Forest <shallow_learning_methods/random_forest>
            Support Vector Machine <shallow_learning_methods/svm>
            XGBoost <shallow_learning_methods/xgb>
            Running All Classification Models <shallow_learning_methods/all_classification_models>

Regression Methods
==================

    .. toctree::
            :maxdepth: 1

            Linear Regression <shallow_learning_methods/linear_regression>
            Logistic Regression <shallow_learning_methods/logistic_regression>
            Random Forest <shallow_learning_methods/random_forest>
            Support Vector Machine <shallow_learning_methods/svm>
            XGBoost <shallow_learning_methods/xgb>
            Running All Regression Models <shallow_learning_methods/all_regression_models>

Hyperparameter Optimization
===========================

Check out how to find the best hyperparameters for your model using :doc:`GridSearch. <shallow_learning_methods/grid_search>`