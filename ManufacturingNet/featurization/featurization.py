"""Featurizer is used for extracting statistical features from raw
signals produced by sensors. Currently, there are 20 features supported.

View the documentation at https://manufacturingnet.readthedocs.io/.
"""

import numpy as np
import scipy.stats


class Featurizer:
    """Featurizer currently supports the 20 features below. Each
    supported feature is contained within its own method.
    """

    def mean(self, a, axis=0):
        """The mean is found by summing all numbers in a dataset and
        dividing by the total number of datapoints.
        """
        try:
            ans = np.mean(a, axis)
            return ans
        except Exception as e:
            print("An exception occurred. Here is the message:\n", e)

    def median(self, a, axis=0):
        """Returns the median of the dataset.
        To find the median, the observations are arranged in order
        from smallest to largest value.
        If there is an odd number of observations, the median is the
        middle value.
        If there is an even number of observations, the median is the
        average of the two middle values.
        """
        try:
            ans = np.median(a, axis)
            return ans
        except Exception as e:
            print("An exception occurred. Here is the message:\n", e)

    def min(self, a, axis=0):
        """Returns the minimum number in a dataset."""
        try:
            ans = np.min(a, axis)
            return ans
        except Exception as e:
            print("An exception occurred. Here is the message:\n", e)

    def max(self, a, axis=0):
        """Returns the maximum number in a dataset."""
        try:
            ans = np.max(a, axis)
            return ans
        except Exception as e:
            print("An exception occurred. Here is the message:\n", e)

    def peak_to_peak(self, a, axis=0):
        """Returns the difference between the maximum and minimum
        numbers in a dataset.
        """
        try:
            ans = np.max(a, axis) - np.min(a, axis)
            return ans
        except Exception as e:
            print("An exception occurred. Here is the message:\n", e)

    def variance(self, a, axis=0):
        """Variance describes how much a random variable differs from
        its expected value.
        The variance is defined as the average of the squares of the
        differences between the individual (observed) and the expected
        value. That means it is always positive.
        In practice, it is a measure of how much something changes.
        """
        try:
            ans = np.var(a, axis)
            return ans
        except Exception as e:
            print("An exception occurred. Here is the message:\n", e)

    def rms(self, a, axis=0):
        """The RMS value of a set of values is the square root of the
        arithmetic mean of the squares of the values, or the square of
        the function that defines the continuous waveform.
        In the case of the RMS statistic of a random process, the
        expected value is used instead of the mean.
        """
        try:
            ans = np.sqrt(np.mean(a ** 2, axis))
            return ans
        except Exception as e:
            print("An exception occurred. Here is the message:\n", e)

    def abs_mean(self, a, axis=0):
        """The absolute mean value of a set of values is the arithmetic
        mean of all the absolute values in a given set of numbers.
        """
        try:
            ans = np.mean(np.absolute(a), axis)
            return ans
        except Exception as e:
            print("An exception occurred. Here is the message:\n", e)

    def shapefactor(self, a, axis=0):
        """Shape factor refers to a value that is affected by an
        object's shape but is independent of its dimensions.
        It is a ratio of RMS value to the absolute mean of a given set
        of numbers.
        """
        try:
            ans = np.sqrt(np.mean(a ** 2, axis)) / \
                np.mean(np.absolute(a), axis)
            return ans
        except Exception as e:
            print("An exception occurred. Here is the message:\n", e)

    def impulsefactor(self, a, axis=0):
        """Impulse factor refers to a value that is affected by an
        absolute maximum values.
        It is a ratio of maximum of absolute values to the absolute
        mean of a given set of numbers.
        """
        try:
            ans = np.max(np.absolute(a), axis) / np.mean(np.absolute(a), axis)
            return ans
        except Exception as e:
            print("An exception occurred. Here is the message:\n", e)

    def crestfactor(self, a, axis=0):
        """Crest factor refers to a value that is affected by an
        absolute maximum values.
        It is a ratio of maximum of absolute values to the RMS value of
        a given set of numbers.
        Crest factor indicates how extreme the peaks are in a wave.
        Crest factor 1 indicates no peaks.
        """
        try:
            ans = np.max(np.absolute(a), axis) / np.sqrt(np.mean(a ** 2, axis))
            return ans
        except Exception as e:
            print("An exception occurred. Here is the message:\n", e)

    def clearancefactor(self, a, axis=0):
        """Clearance factor is peak value divided by the squared mean
        value of the square roots of the absolute amplitudes.
        """
        try:
            ans = np.max(np.absolute(a), axis)
            ans /= ((np.mean(np.sqrt(np.absolute(a)), axis)) ** 2)
            return ans
        except Exception as e:
            print("An exception occurred. Here is the message:\n", e)

    def std(self, a, axis=0):
        """Standard deviation is a measure of the amount of variation
        or dispersion of a set of values.
        A low standard deviation indicates that the values tend to be
        close to the mean of the set, while a high standard deviation
        indicates that the values are spread out over a wider range.
        """
        try:
            ans = np.std(a, axis)
            return ans
        except Exception as e:
            print("An exception occurred. Here is the message:\n", e)

    def skew(self, a, axis=0):
        """Skewness is a measure of the asymmetry of the distribution
        of a real-valued observations about its mean.
        Skewness can be positive, zero, negative, or undefined.
        """
        try:
            ans = scipy.stats.skew(a, axis)
            return ans
        except Exception as e:
            print("An exception occurred. Here is the message:\n", e)

    def kurtosis(self, a, axis=0):
        """Kurtosis measures how heavily the tails of a distribution differ
        from the tails of a normal distribution. In other words,
        kurtosis determines if the tails of a given distribution
        contain extreme values.
        """
        try:
            ans = scipy.stats.kurtosis(a, axis)
            return ans
        except Exception as e:
            print("An exception occurred. Here is the message:\n", e)

    def abslogmean(self, a, axis=0):
        """The absolute logarithmic mean takes a mod of each value,
        followed by log, and then finds the mean of the resultant log
        values.
        """
        try:
            ans = np.mean(np.log(np.abs(a)), axis)
            return ans
        except Exception as e:
            print("An exception occurred. Here is the message:\n", e)

    def meanabsdev(self, a, axis=0):
        """The mean absolute deviation is the average of the absolute
        deviations from a central point.
        It is a summary statistic of statistical dispersion or
        variability.
        """
        try:
            if axis == 0:
                ans = np.mean(np.abs(a - np.mean(a, axis)), axis)
            else:
                ans = np.mean(
                    np.abs(a - np.mean(a, axis).reshape(a.shape[0], 1)), axis)

            return ans
        except Exception as e:
            print("An exception occurred. Here is the message:\n", e)

    def medianabsdev(self, a, axis=0):
        """The median absolute deviation of a data set is the median of
        the absolute deviations from a central point.
        It is a summary statistic of statistical dispersion or
        variability.
        """
        try:
            if axis == 0:
                ans = np.median(np.abs(a - np.median(a, axis)), axis)
            else:
                ans = np.median(
                    np.abs(a - np.median(a, axis).reshape(a.shape[0], 1)), axis)

            return ans
        except Exception as e:
            print("An exception occurred. Here is the message:\n", e)

    def midrange(self, a, axis=0):
        """The mid-range or mid-extreme of a dataset is the mean of the
        maximum and minimum values in a data set.
        """
        try:
            ans = (np.max(a, axis) + np.min(a, axis)) / 2
            return ans
        except Exception as e:
            print("An exception occurred. Here is the message:\n", e)

    def coeff_var(self, a, axis=0):
        """The coefficient of variation, also known as relative
        standard deviation, is a standardized measure of dispersion of
        a distribution. It is often expressed as a percentage, and is
        defined as the ratio of the standard deviation to the mean.
        """
        try:
            ans = scipy.stats.variation(a, axis)
            return ans
        except Exception as e:
            print("An exception occurred. Here is the message:\n", e)
