*******
Deep Neural Network - DNN
*******

Deep Neural Network also known as Fully Connected Neural Net or Multi Layer Perceptron (MLP). DNN is a deep learning architecture which is widely used for regression as well as classification problems.  DNN is capable of learning any mapping function and have been proven to be a universal approximation algorithm.

The Deep Neural Network can be used through **DNN** class.

DNN *(attributes=None, labels=None, shuffle=True)*

Parameters
==========

When initializing a DNN object, the following parameters need to be passed:

- **attributes** *(numpy array, default=None)*: A numpy array of the features as a 2D array. The input shape must be in the form (total number of data points, Features).
- **labels** *(numpy array, default=None)*: A numpy array of the class labels for classification problem or numbers for regression problem. The input shape for labels must be in the form (total number of data points, labels)

The following quenstions and hyperparameters must be entered to construct the DNN model:

- **Default training parameters** *(boolean, default=None)*: All default training parameters will be used if True.
- **Number of neurons in each layer** *(integers, default = None)*: Number of neurons to be used in each layer including input layer and output layer. Input layer size must be same as the number of feature for each datapoint. Output layer size must be one for regression problem else must be equal to number of classes.
- **Activation function for each hidden layer** *(integer, default= 1 (ReLU))*: Numbers corresponding to activation functions. Number of inputs must be equal to number of hidden layers. By default, all the hidden layers will have ReLU activation.
- **Batchnorm requirement for each hidden layer** *(integer, default=0 (No batchnorm))*: 1 if batchnormalization required for the particular hidden layer else 0. Number of inputs must be equal to number of hidden layers. By default, no hidden layers will have batch normalization.
- **Dropout** *(float, default=0.0)*: Input must be between 0 and 1 (0 included and 1 excluded). Number of inputs must be equal to number of hidden layers. By default, all the hidden layers will dropout value equal to zero.
- **Batch_Size** *(integer)*: Sets the batch size for training the model.
- **Validation set size** *(float, default = 0.2)*: The size of the validation set over which the trained model is to be tested for results.
- **Loss_Function** *(integer)*: Sets the loss function to be used for the problem. Input a number corresponding to required loss function.
- **Optimizer** *(integer, default='Adam')*: Sets the optimizer among 'Adam' and 'SGD'. Input 1 for Adam or 2 for SGD.
- **Learning_rate** *(integer, default=0.001)*: The learning rate to be used when training the network. Input must be a non-zero and positive number.
- **Scheduler** *(integer, default=None)*: The learning rate scheduler to be used when training the network. Input 1 for None, 2 for StepLR and 3 for MultiStepLR scheduler.
- **Scheduler specific inputs:**
    - **StepLR Scheduler step** *(integer, default=None)*: Number of epochs after which learning rate needs to be changed. Input must be a non-zero and positive number.
    - **MultiStepLR Milestones** *(integers, default=None)*: Number of epochs at which learning rate needs to be changed. Input must be non-zero and positive numbers.
    - **Multiplying factor** *(float, default=None)*: Factor by which learning rate to be multiplied. Input must be a non-zero and positive number.
- **Epochs** *(integer)*: The number of epochs for which the model is to be trained. Input must be a non-zero and positive number.

Attributes
==========

After training the model, the following instance data is available:

- **Training_loss** *(float)*: The training loss after every epoch for the model.
- **Training_Accuarcy** *(float)*: The validation accuracy of the model in case of the classification problem.
- **Validation_Loss** *(float)*: The validation loss after every epoch for the model.
- **Validation_Accuracy** *(float)*: The validation accuracy after every epeoch for the model in case of the classification problem.
- **Epoch Time** *(float)*: The time required in seconds to train every epoch.
- **confusion_matrix** *(2D array of integers)*: A matrix where the entry in the *i* th row and *j* th column is the number of observations present in group *i* and predicted to be in group *j*. Supported for multilabel classification only.
- **r2 score** *(float)*: The R2 score for the validation set in case of the regression problem.
- **Training and validation loss graph**: Displays a 'Loss' vs 'Epoch' graph and saves the same graph in the root directory.
- **Training and validation accuracy graph**: Displays a 'Accuracy' vs 'Epoch' graph and saves the same graph in the root directory in case of the classification problem.
- **Validation r2 score graph**: Displays a 'Predictions' vs 'Ground truth' graph and saves the same graph in the root directory in case of the regression problem.

Methods
=======

- **get_predict(dataset_X=None)**: Uses the trained model to do predictions on a completely new data. A batch of datapoints can also be passed. The format must be same as input data (test data batch size, features).


Example Usage
=============

.. code-block:: python
    :linenos:

    from ManufacturingNet.models import DNN
    import numpy as np

    attributes = np.load('cwru_feature.npy', allow_pickle = True)
    labels = np.load("cwru_labels.npy", allow_pickle = True)
    
    model = DNN(attributes, labels)
