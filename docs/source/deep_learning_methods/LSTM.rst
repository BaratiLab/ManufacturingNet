**********
Long Short Term Memory Networks (LSTMs)
**********
LSTM is a type of Recurrent Neural Network (RNN). LSTM networks are mainly devised for classifying and making predictions based on time series data. Examples of time series data are daily weather data, stock market data, speech data.

The LSTM can be used through **LSTM** class.
The LSTM model always operates with batch size dimension being the first dimension(batch_first = True).

LSTM *(attributes=None, labels=None, shuffle=True)*

Parameters
==========

When initializing a LSTM object, the following parameters need to be passed:

- **attributes** *(numpy array, default=None)*: A numpy array of the signal reshaped as a 3D array. The input shape must be in the form (total number of data points, sequence length, number of input feature).
- **labels** *(numpy array, default=None)*: A numpy array of the class labels for classification problem or numbers for regression problem. The input shape for labels must be in the form (total number of data points, labels)

The following hyperparameters must be entered to construct the LSTM model:

- **Default training parameters** *(boolean, default=None)*: All default training parameters will be used if True.
- **LSTM input size** *(integer, default=None)*: Input feature size of the dataset. Input must be a non-zero and positive number.
- **LSTM hidden size** *(integer, default=128)*: Hidden units for LSTM layers. Input must be a non-zero and positive number.
- **Number of LSTM layers** *(integer, default=3)* Number of LSTM layers in the model. Input must be non-zero and positive number.
- **Bidirectional LSTM** *(integer, default=0 (Unidirectional))* LSTM bidirectional input. Input 1 for bidirectional LSTM else input 0 for unidirectional input.
- **LSTM output size** *(integer, default=None)*: Output size must be one for regression problem else must be equal to number of classes. Input must be non-zero and positive number.
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

- **get_predict(dataset_X=None)**: Uses the trained model to do predictions on a completely new data. A batch of datapoints can also be passed. The format must be same as input data (test data batch size, sequence length, number of input feature).

Example Usage
=============

.. code-block:: python
    :linenos:

    from ManufacturingNet.models import LSTM
    import numpy as np

    attributes = np.load('lstm_train_x.npy', allow_pickle = True)
    labels = np.load("lstm_train_y.npy", allow_pickle = True)
    
    model = LSTM(attributes, labels)
