**********
GridSearch
**********

When a machine learning model is instantiated, hyperparameters are typically passed in. A **hyperparameter** is a
parameter that alters how the model learns, and is specified before the model runs. For example, the *C*, *kernel*, and
*gamma* parameters for the SVC model are hyperparameters.

Choosing optimal hyperparameter values can be challenging, especially when information about the dataset is limited.
Fortunately, ManufacturingNet can use the GridSearch algorithm to optimize the model's hyperparameters for prediction
accuracy.

**GridSearch** performs hyperparameter optimization by performing an exhaustive search over the possible parameter
values. If a model supports GridSearch, hyperparameter optimization will be offered during parameter input. After
determining the best hyperparameters, the GridSearch score will be displayed.

Supported Models
================

    .. toctree::
            :maxdepth: 1

            Logistic Regression <logistic_regression>
            Random Forest <random_forest>
            SVC <svm>
            NuSVC <svm>
            SVR <svm>
            NuSVR <svm>
            XGBoost <xgb>