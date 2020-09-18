"""signal_preprocessing is a module for performing preprocessing steps
like normalizing, data cleaning, and transforming. The normalizations
currently supported are mean, min-max, and quantile.

Read the documentation at https://manufacturingnet.readthedocs.io.
"""

import numpy as np


class MeanNormalizer:
    """This class normalizes the data along the given axis. It first
    calculates the mean and standard deviation. It then subtracts the
    mean from each value and divides by the standard deviation.

    This transformation sets the mean of the data to 0 and the standard
    deviation to 1.
    """

    def __init__(self, data, axis=0):
        """Initializes a MeanNormalizer object."""
        self.data_mean = np.mean(data, axis)
        self.data_std = np.std(data, axis)
        self.normalized_data = (data - self.data_mean) / (self.data_std)
        self.scaled_data = None

    def get_normalized_data(self):
        """Accessor method for the normalized data.
        Returns an array of normalized data along the given axis.
        """
        return self.normalized_data

    def get_scaled_data(self, test_data):
        """Accessor method for getting the scaled data.
        Returns an array of the normalized data along the given axis.
        """
        self.scaled_data = (test_data * self.data_std) + self.data_mean
        return self.scaled_data


class MinMaxNormalizer:
    """This class normalizes the data along the given axis. It first
    calculates the min and max values along the axis. Then, it subtracts
    the min from each value and divides by the range.
    """

    def __init__(self, data, axis=0):
        """Initializes a MinMaxNormalizer object."""
        self.data_min = np.min(data, axis)
        self.data_max = np.max(data, axis)
        self.normalized_data = \
            (data - self.data_min) / (self.data_max - self.data_min)
        self.scaled_data = None

    def get_normalized_data(self):
        """Accessor method for getting normalized data.
        Returns an array of normalized data along the given axis.
        """
        return self.normalized_data

    def get_scaled_data(self, test_data):
        """Accessor method for getting the scaled data.
        Returns an array of normalized data along the given axis.
        """
        self.scaled_data = \
            (test_data * (self.data_max - self.data_min)) + self.data_min
        return self.scaled_data


class QuantileNormalizer:
    """Scaling using median and quantiles consists of subtracting the
    median from all observations and dividing by the interquartile
    difference. It Scales features using statistics that are robust to
    outliers.

    Default interquartile range is 0.25-0.75 (q1-q3).
    """

    def __init__(self, data, axis=0, q1=0.25, q3=0.75):
        """Initializes a QuantileNormalizer object."""
        assert q1 > 0 and q1 < 1
        assert q3 > q1 and q3 < 1

        self.IQR = np.percentile(data, q3, axis) - \
            np.percentile(data, q1, axis)
        self.data_median = np.median(data, axis)
        self.normalized_data = (data - self.data_median) / (self.IQR)
        self.scaled_data = None

    def get_normalized_data(self):
        """Accessor method for getting normalized data.
        Returns an array of normalized data along the given axis.
        """
        return self.normalized_data

    def get_scaled_data(self, test_data):
        """Accessor method for getting the scaled data.
        Returns an array of normalized data along the given axis.
        """
        self.scaled_data = (test_data * self.IQR) + self.data_median
        return self.scaled_data
