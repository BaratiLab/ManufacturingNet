************************
Deep Learning Methods
************************

ManufacturingNet has inbuilt implementations of various deep learning models. Deep learning methods are known to scale with data and have shown great promise in different areas of reasearch. As a package we have implemented Multi-Layer Perceptron, Convolutioanal Neural Network(CNN), Long Short Term Memory Networks(LSTM), CNN-LSTM  

Deep learning methods typically follow the below pattern:

- Split the dataset into a training and testing set
- Describe the type of task (Regression or Classification)
- Enter the hyperparmaters for the network and allow it run for desired number of epochs to get the performance.

At this point, it is your responsibility to either accept the model, tweak its parameters for better accuracy, or try a
different learning method altogether.

Classification vs Regression
=============================

Deep learning models support both classification and regression analysis.

In a classification model, the model learns features of a datapoint are uses them to predict the category un which the datapoint will fall into.
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

Deep Learning Models
======================

    .. toctree::
            :maxdepth: 1

            Deep fully connected neural network  <deep_learning_methods/MLP>
            Convolutional Neural Network - Images <deep_learning_methods/CNN_Image>
            Convolutional Neural Network - Signals <deep_learning_methods/CNN_Signal>
            Convolutional Neural Network - 3D <deep_learning_methods/CNN_3D>
            Long Short Term Memory Networks <deep_learning_methods/LSTM>
            Convolutional Long Short Term Memory Networks (CNN - LSTM) <deep_learning_methods/CNN_LSTM>
