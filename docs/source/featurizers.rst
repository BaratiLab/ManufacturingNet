***********
Featurizer
***********

Featurizer is a library developed for extracting statistical features from raw signals produced by sensors. Currently, there are 20 features in the library. These features are tested on vibration signals, accelerometer signals, etc.

The Featurizer can be used through **Featurization** class.

f = Featurizer()

The following features can be extracted from the data:

- **mean** *(data = None(default), axis = 0(default)*: Input a numpy array of data and axis along which feature needs to be extracted. The statistical mean is an arithmetic mean process, in that it adds up all numbers in a data set, and then divides the total by the number of data points.
- **median** *(data = None(default), axis = 0(default)*: Input a numpy array of data and axis along which feature needs to be extracted. To find the median, the observations are arranged in order from smallest to largest value. If there is an odd number of observations, the median is the middle value. If there is an even number of observations, the median is the average of the two middle values.
- **min** *(data = None(default), axis = 0(default)*: Input a numpy array of data and axis along which feature needs to be extracted. This function returns the smallest number in a set of numbers.
- **max** *(data = None(default), axis = 0(default)*: Input a numpy array of data and axis along which feature needs to be extracted. This function returns the largest number in a set of numbers.
- **peak_to_peak** *(data = None(default), axis = 0(default)*: Input a numpy array of data and axis along which feature needs to be extracted. This function returns the difference between the maximun and minimum numbers in a set of numbers.
- **variance** *(data = None(default), axis = 0(default)*: Input a numpy array of data and axis along which feature needs to be extracted. Variance describes how much a random variable differs from its expected value. The variance is defined as the average of the squares of the differences between the individual (observed) and the expected value. That means it is always positive. In practice, it is a measure of how much something changes.
- **rms** *(data = None(default), axis = 0(default)*: Input a numpy array of data and axis along which feature needs to be extracted. The RMS value of a set of values is the square root of the arithmetic mean of the squares of the values, or the square of the function that defines the continuous waveform. In the case of the RMS statistic of a random process, the expected value is used instead of the mean.
- **abs_mean** *(data = None(default), axis = 0(default)*: Input a numpy array of data and axis along which feature needs to be extracted. The abs_mean value of a set of values is the arithmetic mean of all the absolute values in a given set of numbers.
- **shapefactor** *(data = None(default), axis = 0(default)*: Input a numpy array of data and axis along which feature needs to be extracted. Shape factor refers to a value that is affected by an object's shape but is independent of its dimensions. It is a ratio of RMS value to the absolute mean of a given set of numbers.
- **impulsefactor** *(data = None(default), axis = 0(default)*: Input a numpy array of data and axis along which feature needs to be extracted. Impulse factor refers to a value that is affected by an absolute maximum values. It is a ratio of maximum of absolute values to the absolute mean of a given set of numbers.
- **crestfactor** *(data = None(default), axis = 0(default)*: Input a numpy array of data and axis along which feature needs to be extracted. Crest factor refers to a value that is affected by an absolute maximum values. It is a ratio of maximum of absolute values to the RMS value of a given set of numbers. Crest factor indicates how extreme the peaks are in a wave. Crest factor 1 indicates no peaks.
- **clearancefactor** *(data = None(default), axis = 0(default)*: Input a numpy array of data and axis along which feature needs to be extracted. Clearance factor is peak value divided by the squared mean value of the square roots of the absolute amplitudes.
- **std** *(data = None(default), axis = 0(default)*: Input a numpy array of data and axis along which feature needs to be extracted. In statistics, the standard deviation is a measure of the amount of variation or dispersion of a set of values. A low standard deviation indicates that the values tend to be close to the mean of the set, while a high standard deviation indicates that the values are spread out over a wider range.
- **skew** *(data = None(default), axis = 0(default)*: Input a numpy array of data and axis along which feature needs to be extracted. In statistics, skewness is a measure of the asymmetry of the distribution of a real-valued observations about its mean. The skewness value can be positive, zero, negative, or undefined.
- **kurtosis** *(data = None(default), axis = 0(default)*: Input a numpy array of data and axis along which feature needs to be extracted. Kurtosis is a statistical measure that defines how heavily the tails of a distribution differ from the tails of a normal distribution. In other words, kurtosis identifies whether the tails of a given distribution contain extreme values.
- **abslogmean** *(data = None(default), axis = 0(default)*: Input a numpy array of data and axis along which feature needs to be extracted. abslogmean is a statistical measure which stands for absolute logarithmic mean of a series of observations. Its takes a mod of each value followed by log and then a mean of the resultant log values.
- **meanabsdev** *(data = None(default), axis = 0(default)*: Input a numpy array of data and axis along which feature needs to be extracted. meanabsdev is a statistical measure which stands for mean absolute deviation of a series of observations. The average absolute deviation, or mean absolute deviation (MAD), of a data set is the average of the absolute deviations from a central point. It is a summary statistic of statistical dispersion or variability.
- **medianabsdev** *(data = None(default), axis = 0(default)*: Input a numpy array of data and axis along which feature needs to be extracted. medianabsdev is a statistical measure which stands for median absolute deviation of a series of observations. The median absolute deviation of a data set is the meadian of the absolute deviations from a central point. It is a summary statistic of statistical dispersion or variability.
- **midrange** *(data = None(default), axis = 0(default)*: Input a numpy array of data and axis along which feature needs to be extracted. In statistics, the mid-range or mid-extreme of a set of statistical data values is the arithmetic mean of the maximum and minimum values in a data set.
- **coeff_var** *(data = None(default), axis = 0(default)*: Input a numpy array of data and axis along which feature needs to be extracted. coeff_var stands for coefficient of variation. In statistics, the coefficient of variation (CV), also known as relative standard deviation (RSD), is a standardized measure of dispersion of a distribution. It is often expressed as a percentage, and is defined as the ratio of the standard deviation to the mean.


Example Usage
=============

.. code-block:: python
    :linenos:

    from ManufacturingNet.Featurization import Featurizer
    import numpy as np
    
    data = np.load("CWRU_raw_data.npy, allow_pickle=True")
    
    f = Featurizer()
    
    # Using above instance extract features
    
    mean = f.mean(data, axis=1)
    median = f.median(data, axis=1)
    min = f.min(data, axis=1)
