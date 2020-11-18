***********
Preprocessing
***********

Preprocessing is a module for doing some pre-processing steps to the data attributes like normalizing, data cleaning, transforming into different types etc. Currently, there are three classes included in the Preprocessing module. 

**MeanNormalizer**:

 This class performs normalization of the data along the given axis. It calculates the mean and standard deviation along the axis.
 It subtracts the mean and divides by the standard deviation. This transformation sets the mean of data to 0 and the standard deviation to 1

The MeanNormalizer class can be used as follows:

M=MeanNormalizer(attributes,axis=0)

attributes: Raw data 
axis: 0 for column-wise and 1 for row-wise (Default axis=0)

There are two getter functions for this class:

get_normalized_data: Returns normalized data along the given axis
get_scaled_data(test_data): Returns normalized test data using the attributes mean and standard deviation

Example Usage
=============

.. code-block:: python
    :linenos:

    from ManufacturingNet.preprocessing import MeanNormalizer
    import numpy as np
    
    data = np.load('cwru_feature.npy', allow_pickle = True)[:-10]
    test_data=data[-10:]
    
    M=MeanNormalizer(data,axis=0)
    
    """Using above instance extract features"""
    
    normalized_data=M.get_normalized_data()
    scaled_data=M.get_scaled_data(test_data)


**MinMaxNormalizer**:

This class performs normalization of the data along the given axis. It calculates the min and max values along the axis.
It subtracts the min and divides by the range (range = max - min).


The MinMaxNormalizer class can be used as follows:

M=MinMaxNormalizer(attributes,axis=0)

attributes: Raw data 
axis: 0 for column-wise and 1 for row-wise (Default axis=0)

There are two getter functions for this class:

get_normalized_data: Returns normalized data along the given axis
get_scaled_data(test_data): Returns normalized test data using the attributes mean and standard deviation

Example Usage
=============

.. code-block:: python
    :linenos:

    from ManufacturingNet.preprocessing import MinMaxNormalizer
    import numpy as np
    
    data = np.load('cwru_feature.npy', allow_pickle = True)[:-10]
    test_data=data[-10:]
    
    M=MinMaxNormalizer(data,axis=0)
    
    """Using above instance extract features"""
    
    normalized_data=M.get_normalized_data()
    scaled_data=M.get_scaled_data(test_data)
   
   
**QuantileNormalizer**:

Scaling using median and quantiles consists of subtracting the median to all the observations and then dividing by the interquartile difference. 
It Scales features using statistics that are robust to outliers.
Default inter-quartile range is 0.25 - 0.75 (q1 - q2).
    
The QuantileNormalizer class can be used as follows:

M=MinMaxNormalizer(attributes,axis=0,q1=0.25,q2=0.75)

attributes: Raw data 
axis: 0 for column-wise and 1 for row-wise (Default axis=0)
q1: percentile of the data for locating the 1st quantile (Default q1:0.25)
q3: percentile of the data for locating the 3rd quantile (Default q1:0.75)

There are two getter functions for this class:

get_normalized_data: Returns normalized data along the given axis
get_scaled_data(test_data): Returns normalized test data using the attributes mean and standard deviation

Example Usage
=============

.. code-block:: python
    :linenos:

    from ManufacturingNet.preprocessing import QuantileNormalizer
    import numpy as np
    
    data = np.load('cwru_feature.npy', allow_pickle = True)[:-10]
    test_data=data[-10:]
    
    M=QuantileNormalizer(data,axis=0)
    
    """Using above instance extract features"""
    
    normalized_data=M.get_normalized_data()
    scaled_data=M.get_scaled_data(test_data)
