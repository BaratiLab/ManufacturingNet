import numpy as np
from sklearn.decomposition import PCA

class MeanNormalizer():
    """
    This class performs normalizaation of the data along the given axis. It calculates the mean and standard deviation along the axis.
    It subtracts the mean and divides by the standard deviation.

    This transformation sets the mean of data to 0 and the standard deviation to 1
    
    
    """
    def __init__(self, a, axis = 0):
        
        self.data = a
        self.axis = axis
        self.data_mean = np.mean(self.data, self.axis)
        self.data_std = np.std(self.data, self.axis)
        self.denormalized_data = None 
        self.normalized_data = (self.data - self.data_mean)/(self.data_std + 0.000001) 

    def get_normalized_data(self):

        """
        Accessor method for getting normalized data

        returns an array of normalized data along the given axis
        
        """
        return self.normalized_data
    
    def get_scaled_data(self, test_data):

        """
        Accessor method for getting the scaled data
        get_scaled_data() cannot be called until the 'get_normalized_data' is called.
        returns an array of normalized data along the given axis or None.

        """

        return self.denormalized_data = (self.test_data*(self.data_std + 0.000001)) + self.data_mean



class MinMaxNormalizer():
    """
    This class performs normalizaation of the data along the given axis. It calculates the min and max values along the axis.
    It subtracts the min and divides by the range (range = max - min).
    
    
    """
    def __init__(self, a, axis = 0):
        
        self.data = a
        self.axis = axis
        self.data_min = np.min(self.data, self.axis)
        self.data_max = np.max(self.data, self.axis)
        self.denormalized_data = None 
        self.normalized_data = (self.data - self.data_min)/(self.data_max -self.data_min) 

    def get_normalized_data(self):

        """
        Accessor method for getting normalized data

        returns an array of normalized data along the given axis
        
        
        """
        return self.normalized_data
    
    def get_denormalized_data(self, test_data):
        
        """
        Accessor method for getting the scaled data
        get_scaled_data() cannot be called until the 'get_normalized_data' is called.
        returns an array of normalized data along the given axis or None.

        """

        return self.denormalized_data = (self.test_data*(self.data_max - self.data_min)) + self.data_min


class QuantileNormalizer():
    """
    Scaling using median and quantiles consists of subtracting the median to all the observations and then dividing by the interquartile difference. 
    It Scales features using statistics that are robust to outliers.
    Default inter-quartile range is 0.25 - 0.75 (q1 - q2).
    
    
    """
    def __init__(self, a, axis = 0, q1 = 0.25, q2 = 0.75):
        
        self.data = a
        self.axis = axis
        self.IQR = np.percentile(self.data, q2, self.axis) - np.percentile(self.data, q1, self.axis)
        self.data_median = np.median(self.data, self.axis)
        self.denormalized_data = None 
        self.normalized_data = (self.data - self.data_median)/(self.IQR) 

    def get_normalized_data(self):

        """
        Accessor method for getting normalized data

        returns an array of normalized data along the given axis
        
        
        """
        return self.normalized_data
    
    def get_denormalized_data(self, test_data):
        
        """
        Accessor method for getting the scaled data
        get_scaled_data() cannot be called until the 'get_normalized_data' is called.
        returns an array of normalized data along the given axis or None.

        """

        return self.denormalized_data = (self.test_data*(self.IQR)) + self.data_median