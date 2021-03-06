*******************
Convolutional Neural Network (CNN) - Signals
*******************

A CNN is a deep learning architecture that is inspired from the human visual cortex. They are generally used in analysing visual data, signal data and mostly have applications in classification problems. In the ManufacturingNet package we have provided the CNN class for analyzing vibration signal data and also image data.

The CNN for signal can be used through **CNN2DSignal** class. In the package we have made a distinction between analyzing signal data and the image data. This distinction gives the advantage of using the powerful CNN network with both these type of datasets 

CNN2DSignal *(attributes=None, labels=None)*

Parameters
==========

When initializing a CNN2DSignal object, the following parameters need to be passed:

- **attributes** *(numpy array, default=None)*: A numpy array of the signal reshaped as a 2D array. The input shape must be in the form (total number of data points, num_channels, height, width).
- **labels** *(numpy array, default=None)*: A numpy array of the class labels.

The following hyperparameters must be entered to construct the CNN2DSignal model:

- **Number_of_Convolutions** *(integer, default=None)*: The number of convolutional layers to be used in the network.
- **kernel_size** *(integer, default = (3,3))*: The size of the kernel to be used in the convolution operation.
- **Padding** *(integer, default=(0,0))*: The image padding to be used for the network.
- **Stride** *(integer, default=(1,1))*: The stride to be used for the convolutional filter.
- **Dropout** *(float, default=0.0)*: The dropout ratio in the final layer of the network.
- **Pooling_Layers** *(boolean)*: Determines whether max pooling should be applied to the convolutional layer. If default is chosen the pooling is applied only to the last convolutional layer.
- **Pooling_Size** *(integer, default=(2,2))*: The size of the of the pooling filter representing the region over which pooling is applied.
- **Pooling_Stride** *(boolean, default=(2,2))*: The stride for the pooling filter.
- **Batch_Normalization** *(boolean, default =1)*: Determines whether or not batch normalization must be applied to the convolutional layer. By default, all the convolutional layers will have batch normalization,
- **Num_Classes** *(integer)*: The number of classes for the classification problem. Please enter 1 if you are dealing with a regression problem
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

- **get_predict(dataset_X=None)**: Uses the trained model to do predictions on a completely new data. A batch of datapoints can also be passed. The format must be same as input data (test data batch size, num_channels, height, width).


Example Usage
=============

.. code-block:: python
    :linenos:

    from ManufacturingNet.models import CNN2DSignal
    import numpy as np

    X = np.load('CWRU_dataset.npy')
    labels = np.load("CWRU_labels.npy")
    attributes = X.reshape(len(X),1,40,40)                    # Convert to required shape format 
    model = CNN2DSignal(attributes, labels)
