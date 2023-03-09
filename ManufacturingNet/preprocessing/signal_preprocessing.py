"""signal_preprocessing is a module for performing preprocessing steps
like normalizing, data cleaning, and transforming. The normalizations
currently supported are mean, min-max, and quantile.

Read the documentation at https://manufacturingnet.readthedocs.io.
"""

import numpy as np


class MeanNormalizer():
    """
    This class performs normalizaation of the data along the given axis. 
    It calculates the mean and standard deviation along the axis.
    It subtracts the mean and divides by the standard deviation.

    This transformation sets the mean of data to 0 and the standard deviation to 1
    
    
    """
    def __init__(self, a, axis = 0):
        
        self.data = a
        self.axis = axis
        self.reshape_dim = self.data.shape[abs(self.axis-1)]
        self.data_mean = np.mean(self.data, self.axis, keepdims=True)#.reshape(self.reshape_dim,1)
        self.data_std = np.std(self.data, self.axis, keepdims=True)#.reshape(self.reshape_dim,1)
        self.normalized_data = (self.data - self.data_mean)/(self.data_std) 
        self.scaled_data = None 
        

    def get_normalized_data(self):

        """
        Accessor method for getting normalized data

        Returns an array of normalized data along the given axis
        
        """
        

        return self.normalized_data
    
    def get_scaled_data(self, test_data):

        """
        Accessor method for getting the scaled data

        Returns an array of normalized data along the given axis or None.

        """
        self.scaled_data = (test_data*(self.data_std)) + self.data_mean

        return self.scaled_data



class MinMaxNormalizer():
    """
    This class performs normalizaation of the data along the given axis. It calculates the min and max values along the axis.
    It subtracts the min and divides by the range (range = max - min).
    
    
    """
    def __init__(self, a, axis = 0):
        
        self.data = a
        self.axis = axis
        self.reshape_dim = self.data.shape[abs(self.axis-1)]
        self.data_min = np.min(self.data, self.axis, keepdims=True) #.reshape(self.reshape_dim,1)
        self.data_max = np.max(self.data, self.axis, keepdims=True) #.reshape(self.reshape_dim,1)
        self.scaled_data = None 
        self.normalized_data = (self.data - self.data_min)/(self.data_max -self.data_min) 

    def get_normalized_data(self):

        """
        Accessor method for getting normalized data

        Returns an array of normalized data along the given axis
        """
        return self.normalized_data
    
    def get_scaled_data(self, test_data):
        
        """
        Accessor method for getting the scaled data

        Returns an array of normalized data along the given axis or None.

        """
        self.scaled_data = (test_data*(self.data_max - self.data_min)) + self.data_min

        return self.scaled_data


class QuantileNormalizer():
    """
    Scaling using median and quantiles consists of subtracting the median to all the observations and then dividing by the interquartile difference. 
    It Scales features using statistics that are robust to outliers.
    Default inter-quartile range is 0.25 - 0.75 (q1 - q2).
    
    
    """
    def __init__(self, a, axis = 0, q1 = 0.25, q2 = 0.75):
        
        assert q1 > 0 and q1 < 1
        assert q2 > q1 and q2 < 1
        
        self.data = a
        self.axis = axis
        self.reshape_dim = self.data.shape[abs(self.axis-1)]
        self.IQR = np.percentile(self.data, q2, self.axis, keepdims=True) - \
                   np.percentile(self.data, q1, self.axis, keepdims=True)
        self.data_median = np.median(self.data, self.axis, keepdims=True) #.reshape(self.reshape_dim,1)
        self.scaled_data = None 
        self.normalized_data = (self.data - self.data_median)/(self.IQR) 

    def get_normalized_data(self):

        """
        Accessor method for getting normalized data

        Returns an array of normalized data along the given axis
        
        
        """
        return self.normalized_data
    
    def get_scaled_data(self, test_data):
        
        """
        Accessor method for getting the scaled data

        Returns an array of normalized data along the given axis or None.

        """
        self.scaled_data = (test_data*(self.IQR)) + self.data_median

        return self.scaled_data
