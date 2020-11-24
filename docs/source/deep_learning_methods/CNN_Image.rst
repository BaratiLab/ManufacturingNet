*****************
Convolutional Neural Network - Images
*****************

A CNN is a deep learning architecture that is inspired from the human visual cortex. They are generally used in analysing visual data, signal data and mostly have applications in classification problems. In the ManufacturingNet package we have provided the CNN class for analyzing vibration signal data and also image data.

The CNN for image dataset can be used through **CNN2DImage** class. In the package we have made a distinction between analyzing signal data and the image data. This distinction gives the advantage of using the powerful CNN network with both these type of datasets.

For CNN2DImage, the data needs to be in a specific format. Unlike other models, for CNN2DImage, training and validation data needs to be passed in separately in two different folders. In these folders, images needs to be stored in class specific folders. For example, if there are 3 classes, training data folder must contain 3 more folders corresponding to each class. Similar structure is required for validation data.

CNN2dImage *(train_data_address, val_data_address, shuffle = True)*

Parameters
==========

When initializing a CNN2dImage object, the following parameters need to be passed:

- **train_data_address** *(training data folder address, default=None)*: Training data folder address input in string format. 
- **val_data_address** *(validation data folder address, default=None)*: Validation data folder address input in string format. 

The following hyperparameters must be entered to construct the CNN model:

- **Input image size** *(integers, (heigth, width, channels), default=None)*: Image size for training the model. Images will be resized to the size enterd. To convert colored images to grayscale images, third argument must be pass as 1. If no resizing required, original shape needs to be entered. All the images must have the same size! Input must be non-zero and positive numbers.
- **Number_of_Convolutions** *(integer, default=None)*: The number of convolutional layers to be used in the network.
- **kernel_size** *(integer, default = (3,3))*: The size of the kernel to be used in the convolution operation.
- **Padding** *(integer, default=(0,0))*: The image padding to be used for the network.
- **Stride** *(integer, default=(1,1))*: The stride to be used for the convolutional filter.
- **Dropout** *(float, default=0.0)*: The dropout ratio in the final layer of the network.
- **Pooling_Layers** *(boolean)*: Determines whether max pooling should be applied to the convolutional layer. If default is chosen the pooling is applied only to the last convolutional layer.
- **Pooling_Size** *(integer, default=(3,3))*: The size of the of the pooling filter representing the region over which pooling is applied.
- **Pooling_Stride** *(integer, default=(2,2))*: The stride for the pooling filter.
- **Pooling_Padding** *(integer, default=(0,0))*: The padding to used for the pooling filter
- **Batch_Normalization** *(boolean, default =1)*: Determines whether or not batch normalization must be applied to the convolutional layer. By default, all the convolutional layers will have batch normalization,
- **Batch_Size** *(integer)*: Sets the batch size for training the model.
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
- **Training and validation loss graph**: Displays a 'Loss' vs 'Epoch' graph and saves the same graph in the root directory.
- **Training and validation accuracy graph**: Displays a 'Accuracy' vs 'Epoch' graph and saves the same graph in the root directory in case of the classification problem.

Methods
=======

- **get_predict(dataset_X=None)**: Uses the trained model to do predictions on a completely new data. A batch of datapoints can also be passed. The format must be same as input data (test data batch size, in_channels, height, width). Images must be converted to matrix and image size must be same as training image size(**Input image size**).


Example Usage
=============

.. code-block:: python
    :linenos:

    from ManufacturingNet.models import CNN2DImage
    import numpy as np
    
    train_data_address = train_data_g/
    val_data_address = val_data_g/
    model = CNN2DImage(train_data_address, val_data_address)
