#!/usr/bin/env python
# coding: utf-8

# In[1]:


class Featurizer():
    """
    There are currently 20 feature extraction functions. Each feature function has its own description and explanation.
    1. mean
    2. median
    3. min
    4. max
    5. peak_to_peak
    6. variance
    7. rms
    8. absolute_mean
    9. shape_factor
    10. impulse_factor
    11. crest_factor
    12. clearance_factor
    13. std
    14. skewness
    15. kurtosis
    16. abslogmean
    17. meanabsdev
    18. medianabsdev
    19. midrange
    20. coeff_var
    """
    
    import numpy as np
    import scipy.stats

    
    def mean(self, a, axis = 0):
        """
        The statistical mean is an arithmetic mean process, 
        in that it adds up all numbers in a data set, 
        and then divides the total by the number of data points.
        
        Input:
        A numpy array (1D or 2D array), and
        axis = 0 or 1, along which, the mean operation to be performed. 
        By default, axis = 0
        
        If axis = 0, mean operation is performed column-wise,
        and for axis = 1, mean operation is performed row-wise.
        
        Output:
        Output is float64
        For 1-D array, a number is returned.
        For 2-D array, an array with mean values is returned.
        For 2-D array with axis = 0, output dimension is equal to number of columns.
        For 2-D array with axis = 1, output dimension is equal to number of rows.
        
        Example:
        a = [[1,2,3],[4,5,6]]
        a.shape = (2,3)
        Featurizer().mean(a,0) = array([2.5, 3.5, 4.5]), shape = (3,) which is equal to number of columns
        Featurizer().mean(a,1) = array([2, 5]), shape = (2,) which is equal to number of rows
        
        """
        try:
            ans = self.np.mean(a,axis)
            return ans
        except :
            print('Error: Axis value can only be 0 or 1')
            
    def median(self,a,axis = 0):
        """
         To find the median, the observations are arranged in order from smallest to largest value. 
         If there is an odd number of observations, the median is the middle value. 
         If there is an even number of observations, the median is the average of the two middle values.
        
        Input:
        A numpy array (1D or 2D array), and
        axis = 0 or 1, along which, the median operation to be performed. 
        By default, axis = 0
        
        If axis = 0, median operation is performed column-wise,
        and for axis = 1, the operation is performed row-wise.
        
        Output:
        Output is float64
        For 1-D array, a number is returned.
        For 2-D array, an array with median values is returned.
        For 2-D array with axis = 0, output dimension is equal to number of columns.
        For 2-D array with axis = 1, output dimension is equal to number of rows.
        
        Example:
        a = [[1,2,3],[4,5,6]]
        a.shape = (2,3)
        Featurizer().median(a,0) = array([2.5, 3.5, 4.5]), shape = (3,) which is equal to number of columns
        Featurizer().median(a,1) = array([2, 5]), shape = (2,) which is equal to number of rows
        
        """
        try:
            ans = self.np.median(a,axis)
            return ans
        except :
            print('Error: Axis value can only be 0 or 1')
        
    def min(self, a, axis = 0):
        """
         This function returns the minimum number in a set of numbers
        
        Input:
        A numpy array (1D or 2D array), and
        axis = 0 or 1, along which, the min operation to be performed. 
        By default, axis = 0
        
        If axis = 0, min operation is performed column-wise,
        and for axis = 1, the operation is performed row-wise.
        
        Output:
        Output is float64
        For 1-D array, a number is returned.
        For 2-D array, an array with min values is returned.
        For 2-D array with axis = 0, output dimension is equal to number of columns.
        For 2-D array with axis = 1, output dimension is equal to number of rows.
        
        Example:
        a = [[1,2,3],[4,5,6]]
        a.shape = (2,3)
        Featurizer().min(a,0) = array([1, 2, 3]), shape = (3,) which is equal to number of columns
        Featurizer().min(a,1) = array([1, 4]), shape = (2,) which is equal to number of rows
        
        """
        try:
            ans = self.np.min(a,axis)
            return ans
        except :
            print('Error: Axis value can only be 0 or 1')
        
        
    def max(self, a, axis = 0):
        """
         This function returns the maximum number in a set of numbers
        
        Input:
        A numpy array (1D or 2D array), and
        axis = 0 or 1, along which, the max operation to be performed. 
        By default, axis = 0
        
        If axis = 0, max operation is performed column-wise,
        and for axis = 1, the operation is performed row-wise.
        
        Output:
        Output is float64
        For 1-D array, a number is returned.
        For 2-D array, an array with max values is returned.
        For 2-D array with axis = 0, output dimension is equal to number of columns.
        For 2-D array with axis = 1, output dimension is equal to number of rows.
        
        Example:
        a = [[1,2,3],[4,5,6]]
        a.shape = (2,3)
        Featurizer().max(a,0) = array([4, 5, 6]), shape = (3,) which is equal to number of columns
        Featurizer().max(a,1) = array([3, 6]), shape = (2,) which is equal to number of rows
        
        """
        try:
            ans = self.np.max(a,axis)
            return ans
        except :
            print('Error: Axis value can only be 0 or 1')
    
    def peak_to_peak(self, a, axis = 0):
        """
         This function returns the difference between the maximun and minimum numbers in a set of numbers
        
        Input:
        A numpy array (1D or 2D array), and
        axis = 0 or 1, along which, the peak_to_peak operation to be performed. 
        By default, axis = 0
        
        If axis = 0, peak_to_peak operation is performed column-wise,
        and for axis = 1, the operation is performed row-wise.
        
        Output:for a
        Output is float64
        For 1-D array, a number is returned.
        For 2-D array, an array with peak_to_peak values is returned.
        For 2-D array with axis = 0, output dimension is equal to number of columns.
        For 2-D array with axis = 1, output dimension is equal to number of rows.
        
        Example:
        a = [[1,2,3],[4,5,6]]
        a.shape = (2,3)
        Featurizer().peak_to_peak(a,0) = array([3,3,3]), shape = (3,) which is equal to number of columns
        Featurizer().peak_to_peak(a,1) = array([2, 2]), shape = (2,) which is equal to number of rows
        
        """
        try:
            ans = self.np.max(a,axis) - self.np.min(a,axis) 
            return ans
        except :
            print('Error: Axis value can only be 0 or 1')
            
    def variance(self, a, axis = 0):
        """
        Variance describes how much a random variable differs from its expected value.
        The variance is defined as the average of the squares of the differences 
        between the individual (observed) and the expected value. That means it is always positive.
        In practice, it is a measure of how much something changes.
        
        Input:
        A numpy array (1D or 2D array), and
        axis = 0 or 1, along which, the variance operation to be performed. 
        By default, axis = 0
        
        If axis = 0, variance operation is performed column-wise,
        and for axis = 1, the operation is performed row-wise.
        
        Output:
        Output is float64
        For 1-D array, a number is returned.
        For 2-D array, an array with variance values is returned.
        For 2-D array with axis = 0, output dimension is equal to number of columns.
        For 2-D array with axis = 1, output dimension is equal to number of rows.
        
        Example:
        a = [[1,2,3],[4,5,6]]
        a.shape = (2,3)
        Featurizer().variance(a,0) = array([2.25, 2.25, 2.25]]), shape = (3,) which is equal to 
        number of columns
        Featurizer().variance(a,1) = array([0.66666667, 0.66666667]), shape = (2,) which is equal to 
        number of rows
        
        """
        try:
            ans = self.np.var(a,axis)
            return ans
        except :
            print('Error: Axis value can only be 0 or 1')
            
    def rms(self, a, axis = 0):
        """
        The RMS value of a set of values is the square root of the arithmetic mean of the squares of the values, 
        or the square of the function that defines the continuous waveform.
        In the case of the RMS statistic of a random process, the expected value is used instead of the mean.
        
        Input:
        A numpy array (1D or 2D array), and
        axis = 0 or 1, along which, the RMS operation to be performed. 
        By default, axis = 0
        
        If axis = 0, RMS operation is performed column-wise,
        and for axis = 1, the operation is performed row-wise.
        
        Output:
        Output is float64
        For 1-D array, a number is returned.
        For 2-D array, an array with RMS values is returned.
        For 2-D array with axis = 0, output dimension is equal to number of columns.
        For 2-D array with axis = 1, output dimension is equal to number of rows.
        
        Example:
        a = [[1,2,3],[4,5,6]]
        a.shape = (2,3)
        Featurizer().rms(a,0) = array([2.91547595, 3.80788655, 4.74341649]), shape = (3,) which is equal to 
        number of columns
        Featurizer().rms(a,1) = array([2.1602469 , 5.06622805]), shape = (2,) which is equal to number of rows
        
        """
        try:
            ans = self.np.sqrt(self.np.mean(a**2,axis))
            return ans
        except :
            print('Error: Axis value can only be 0 or 1')
    
    def abs_mean(self, a, axis = 0):
        """
        The abs_mean value of a set of values is the arithmetic mean of all the absolute values in a 
        given set of numbers, 
        
        Input:
        A numpy array (1D or 2D array), and
        axis = 0 or 1, along which, the abs_mean operation to be performed. 
        By default, axis = 0
        
        If axis = 0, abs_mean operation is performed column-wise,
        and for axis = 1, the operation is performed row-wise.
        
        Output:
        Output is float64
        For 1-D array, a number is returned.
        For 2-D array, an array with abs_mean values is returned.
        For 2-D array with axis = 0, output dimension is equal to number of columns.
        For 2-D array with axis = 1, output dimension is equal to number of rows.
        
        Example:
        a = [[1,-2,3],[-4,5,6]]
        a.shape = (2,3)
        Featurizer().abs_mean(a,0) = array([2.5, 3.5, 4.5]), shape = (3,) which is equal to number of columns
        Featurizer.abs_mean(a,1) = array([2, 5]), shape = (2,) which is equal to number of rows
        
        """
        try:
            ans = self.np.mean(self.np.absolute(a), axis)
            return ans
        except :
            print('Error: Axis value can only be 0 or 1')
    
    def shapefactor(self, a, axis = 0):
        """
        Shape factor refers to a value that is affected by an object's shape but is independent of 
        its dimensions.
        It is a ratio of RMS value to the absolute mean of a given set of numbers.
        
        Input:
        A numpy array (1D or 2D array), and
        axis = 0 or 1, along which, the shapefactor operation to be performed. 
        By default, axis = 0
        
        If axis = 0, shapefactor operation is performed column-wise,
        and for axis = 1, the operation is performed row-wise.
        
        Output:
        Output is float64
        For 1-D array, a number is returned.
        For 2-D array, an array with shapefactor values is returned.
        For 2-D array with axis = 0, output dimension is equal to number of columns.
        For 2-D array with axis = 1, output dimension is equal to number of rows.
        
        Example:
        a = [[1,2,3],[4,5,6]]
        a.shape = (2,3)
        Featurizer().shapefactor(a,0) = array([1.16619038, 1.08796759, 1.05409255]), shape = (3,) which is equal to 
        number of columns
        Featurizer().shapefactor(a,1) = array([1.08012345, 1.01324561]), shape = (2,) which is equal to number 
        of rows
        
        """
        try:
            ans = self.np.sqrt(self.np.mean(a**2,axis))/self.np.mean(self.np.absolute(a), axis)
            return ans
        except :
            print('Error: Axis value can only be 0 or 1')
            
    def impulsefactor(self, a, axis = 0):
        """
        Impulse factor refers to a value that is affected by an absolute maximum values.
        It is a ratio of maximum of absolute values to the absolute mean of a given set of numbers.
        
        Input:
        A numpy array (1D or 2D array), and
        axis = 0 or 1, along which, the impulsefactor operation to be performed. 
        By default, axis = 0
        
        If axis = 0, impulsefactor operation is performed column-wise,
        and for axis = 1, the operation is performed row-wise.
        
        Output:
        Output is float64
        For 1-D array, a number is returned.
        For 2-D array, an array with impulsefactor values is returned.
        For 2-D array with axis = 0, output dimension is equal to number of columns.
        For 2-D array with axis = 1, output dimension is equal to number of rows.
        
        Example:
        a = [[1,2,3],[4,5,6]]
        a.shape = (2,3)
        Featurizer().impulsefactor(a,0) = array([1.6, 1.42857143, 1.33333333]), shape = (3,) which is equal to 
        number of columns
        Featurizer().impulsefactor(a,1) = array([1.5,1.2]), shape = (2,) which is equal to number of rows
        
        """
        try:
            ans = self.np.max(self.np.absolute(a),axis)/self.np.mean(self.np.absolute(a), axis)
            return ans
        except :
            print('Error: Axis value can only be 0 or 1')
    
    def crestfactor(self, a, axis = 0):
        """
        Crest factor refers to a value that is affected by an absolute maximum values.
        It is a ratio of maximum of absolute values to the RMS value of a given set of numbers.
        Crest factor indicates how extreme the peaks are in a wave. Crest factor 1 indicates no peaks.
        
        Input:
        A numpy array (1D or 2D array), and
        axis = 0 or 1, along which, the crestfactor operation to be performed. 
        By default, axis = 0
        
        If axis = 0, crestfactor operation is performed column-wise,
        and for axis = 1, the operation is performed row-wise.
        
        Output:
        Output is float64
        For 1-D array, a number is returned.
        For 2-D array, an array with crestfactor values is returned.
        For 2-D array with axis = 0, output dimension is equal to number of columns.
        For 2-D array with axis = 1, output dimension is equal to number of rows.
        
        Example:
        a = [[1,2,3],[4,5,6]]
        a.shape = (2,3)
        Featurizer().crestfactor(a,0) = array([1.37198868, 1.31306433, 1.26491106]), shape = (3,) which is equal to 
        number of columns
        Featurizer().crestfactor(a,1) = array([1.38873015, 1.18431305]), shape = (2,) which is equal to number of rows
        
        """
        try:
            ans = self.np.max(self.np.absolute(a),axis)/self.np.sqrt(self.np.mean(a**2,axis))
            return ans
        except :
            print('Error: Axis value can only be 0 or 1')
            
    def clearancefactor(self, a, axis = 0):
        """
        Clearance factor is peak value divided by the squared mean value of the square roots of the absolute amplitudes.
        
        Input:
        A numpy array (1D or 2D array), and
        axis = 0 or 1, along which, the clearancefactor operation to be performed. 
        By default, axis = 0
        
        If axis = 0, clearancefactor operation is performed column-wise,
        and for axis = 1, the operation is performed row-wise.
        
        Output:
        Output is float64
        For 1-D array, a number is returned.
        For 2-D array, an array with clearancefactor values is returned.
        For 2-D array with axis = 0, output dimension is equal to number of columns.
        For 2-D array with axis = 1, output dimension is equal to number of rows.
        
        Example:
        a = [[1,2,3],[4,5,6]]
        a.shape = (2,3)
        Featurizer().clearancefactor(a,0) = array([1.77777778, 1.50098818, 1.372583]), shape = (3,) which is equal to
        number of columns
        Featurizer().clearancefactor(a,1) = array([1.57054283, 1.20814337]), shape = (2,) which is equal to number of rows
        
        """
        try:
            ans = self.np.max(self.np.absolute(a),axis)/((self.np.mean(self.np.sqrt(self.np.absolute(a)),axis))**2)
            return ans
        except :
            print('Error: Axis value can only be 0 or 1')
    
    def std(self, a, axis = 0):
        """1.38873015, 1.18431305
        In statistics, the standard deviation is a measure of the amount of variation or dispersion of a set of values.
        A low standard deviation indicates that the values tend to be close to the mean of the set, 
        while a high standard deviation indicates that the values are spread out over a wider range.
        
        Input:
        A numpy array (1D or 2D array), and
        axis = 0 or 1, along which, the standard deviation operation to be performed. 
        By default, axis = 0
        
        If axis = 0, standard deviation operation is performed column-wise,
        and for axis = 1, the operation is performed row-wise.
        
        Output:
        Output is float64
        For 1-D array, a number is returned.
        For 2-D array, an array with standard deviation values is returned.
        For 2-D array with axis = 0, output dimension is equal to number of columns.
        For 2-D array with axis = 1, output dimension is equal to number of rows.
        
        Example:
        a = [[1,2,3],[4,5,6]]
        a.shape = (2,3)
        Featurizer().std(a,0) = array([1.5, 1.5, 1.5]), shape = (3,) which is equal to number of columns
        Featurizer().std(a,1) = array([0.81649658, 0.81649658]), shape = (2,) which is equal to number of rows
        
        """
        try:
            ans = self.np.std(a,axis)
            return ans
        except :
            print('Error: Axis value can only be 0 or 1')
            
    def skew(self, a, axis = 0):
        """
        In statistics, skewness is a measure of the asymmetry of the distribution of a real-valued observations 
        about its mean. The skewness value can be positive, zero, negative, or undefined
        
        Input:
        A numpy array (1D or 2D array), and
        axis = 0 or 1, along which, the skewness operation to be performed. 
        By default, axis = 0
        
        If axis = 0, skewness operation is performed column-wise,
        and for axis = 1, the operation is performed row-wise.
        
        Output:
        Output is float64
        For 1-D array, a number is returned.
        For 2-D array, an array with skew values is returned.
        For 2-D array with axis = 0, output dimension is equal to number of columns.
        For 2-D array with axis = 1, output dimension is equal to number of rows.
        
        Example:
        a = [[1,2,3],[4,5,6]]
        a.shape = (2,3)
        Featurizer().skew(a,0) = array([0., 0., 0.]), shape = (3,) which is equal to number of columns
        Featurizer().skew(a,1) = array([0.,0.]), shape = (2,) which is equal to number of rows
        
        """
        try:
            ans = self.scipy.stats.skew(a,axis)
            return ans
        except :
            print('Error: Axis value can only be 0 or 1')
    
    def kurtosis(self, a, axis = 0):
        """
        Kurtosis is a statistical measure that defines how heavily the tails of a distribution differ 
        from the tails of a normal distribution. In other words, kurtosis identifies 
        whether the tails of a given distribution contain extreme values.
        
        Input:
        A numpy array (1D or 2D array), and
        axis = 0 or 1, along which, the kurtosis operation to be performed. 
        By default, axis = 0
        
        If axis = 0, kurtosis operation is performed column-wise,
        and for axis = 1, the operation is performed row-wise.
        
        Output:
        Output is float64
        For 1-D array, a number is returned.
        For 2-D array, an array with kurtosis values is returned.
        For 2-D array with axis = 0, output dimension is equal to number of columns.
        For 2-D array with axis = 1, output dimension is equal to number of rows.
        
        Example:
        a = [[1,2,3],[4,5,6]]
        a.shape = (2,3)
        Featurizer().kurtosis(a,0) = array([-2., -2., -2.]), shape = (3,) which is equal to number of columns
        Featurizer().kurtosis(a,1) = array([-1.5, -1.5]), shape = (2,) which is equal to number of rows
        
        """
        try:
            ans = self.scipy.stats.kurtosis(a,axis)
            return ans
        except :
            print('Error: Axis value can only be 0 or 1')

    def abslogmean(self, a, axis = 0):
        """
        abslogmean is a statistical measure which stands for absolute logarithmic mean of a series of observations.
        Its takes a mod of each value followed by log and then a mean of the resultant log values
        
        Input:
        A numpy array (1D or 2D array), and
        axis = 0 or 1, along which, the abslogmean operation to be performed. 
        By default, axis = 0
        
        If axis = 0, abslogmean operation is performed column-wise,
        and for axis = 1, the operation is performed row-wise.
        
        Output:
        Output is float64
        For 1-D array, a number is returned.
        For 2-D array, an array with abslogmean values is returned.
        For 2-D array with axis = 0, output dimension is equal to number of columns.
        For 2-D array with axis = 1, output dimension is equal to number of rows.
        
        Example:
        a = [[1,2,3],[4,5,6]]
        a.shape = (2,3)
        Featurizer().abslogmean(a,0) = array([0.69314718, 1.15129255, 1.44518588]), shape = (3,) which is equal to number of columns
        Featurizer().abslogmean(a,1) = array([0.59725316, 1.59583058]), shape = (2,) which is equal to number of rows
        
        """
        try:
            ans = self.np.mean(self.np.log(abs(a)), axis)
            return ans
        except :
            print('Error: Axis value can only be 0 or 1')

    
    def meanabsdev(self, a, axis = 0):
        """
        meanabsdev is a statistical measure which stands for mean absolute deviation of a series of observations.
        The average absolute deviation, or mean absolute deviation (MAD), of a data set is the average of the absolute deviations from a central point. 
        It is a summary statistic of statistical dispersion or variability.
        
        Input:
        A numpy array (1D or 2D array), and
        axis = 0 or 1, along which, the meanabsdev operation to be performed. 
        By default, axis = 0
        
        If axis = 0, meanabsdev operation is performed column-wise,
        and for axis = 1, the operation is performed row-wise.
        
        Output:
        Output is float64
        For 1-D array, a number is returned.
        For 2-D array, an array with meanabsdev values is returned.
        For 2-D array with axis = 0, output dimension is equal to number of columns.
        For 2-D array with axis = 1, output dimension is equal to number of rows.
        
        Example:
        a = [[1,2,3],[4,5,6]]
        a.shape = (2,3)
        Featurizer().meanabsdev(a,0) = array([1.5, 1.5, 1.5]), shape = (3,) which is equal to number of columns
        Featurizer().meanabsdev(a,1) = array([0.66666667, 0.66666667]), shape = (2,) which is equal to number of rows
        
        """
        try:
            if axis == 0:
                ans = self.np.mean(abs(a-self.np.mean(a,axis)),axis)
                
            else:
                ans = self.np.mean(abs(a-self.np.mean(a,axis).reshape(a.shape[0],1)),axis)

            return ans
        except :
            print('Error: Axis value can only be 0 or 1')


    def medianabsdev(self, a, axis = 0):
        """
        medianabsdev is a statistical measure which stands for median absolute deviation of a series of observations.
        The median absolute deviation of a data set is the meadian of the absolute deviations from a central point. 
        It is a summary statistic of statistical dispersion or variability.
        
        Input:
        A numpy array (1D or 2D array), and
        axis = 0 or 1, along which, the medianabsdev operation to be performed. 
        By default, axis = 0
        
        If axis = 0, medianabsdev operation is performed column-wise,
        and for axis = 1, the operation is performed row-wise.
        
        Output:
        Output is float64
        For 1-D array, a number is returned.
        For 2-D array, an array with medianabsdev values is returned.
        For 2-D array with axis = 0, output dimension is equal to number of columns.
        For 2-D array with axis = 1, output dimension is equal to number of rows.
        
        Example:
        a = [[1,2,3],[4,5,6]]
        a.shape = (2,3)
        Featurizer().medianabsdev(a,0) = array([1.5, 1.5, 1.5]), shape = (3,) which is equal to number of columns
        Featurizer().medianabsdev(a,1) = array([1., 1.]), shape = (2,) which is equal to number of rows
        
        """
        try:
            if axis == 0:
                ans = self.np.median(abs(a-self.np.median(a,axis)),axis)
                
            else:
                ans = self.np.median(abs(a-self.np.median(a,axis).reshape(a.shape[0],1)),axis)
                
            return ans
        except :
            print('Error: Axis value can only be 0 or 1')

    def midrange(self, a, axis = 0):
        """
        In statistics, the mid-range or mid-extreme of a set of statistical data values is the
         arithmetic mean of the maximum and minimum values in a data set.
        
        Input:
        A numpy array (1D or 2D array), and
        axis = 0 or 1, along which, the midrange operation to be performed. 
        By default, axis = 0
        
        If axis = 0, midrange operation is performed column-wise,
        and for axis = 1, the operation is performed row-wise.
        
        Output:
        Output is float64
        For 1-D array, a number is returned.
        For 2-D array, an array with midrange values is returned.
        For 2-D array with axis = 0, output dimension is equal to number of columns.
        For 2-D array with axis = 1, output dimension is equal to number of rows.
        
        Example:
        a = [[1,2,3],[4,5,6]]
        a.shape = (2,3)
        Featurizer().midrange(a,0) = array([2.5, 3.5, 4.5]), shape = (3,) which is equal to number of columns
        Featurizer().midrange(a,1) = array([2., 5.]) , shape = (2,) which is equal to number of rows
        
        """
        try:
            ans = (self.np.max(a,axis)+self.np.min(a,axis))/2
            return ans
        except :
            print('Error: Axis value can only be 0 or 1')

    def coeff_var(self, a, axis = 0):
        """
        coeff_var stands for coefficient of variation
        In statistics, the coefficient of variation (CV), also known as relative standard deviation (RSD), 
        is a standardized measure of dispersion of a distribution. It is often expressed as a percentage,
        and is defined as the ratio of the standard deviation to the mean.
        
        Input:
        A numpy array (1D or 2D array), and
        axis = 0 or 1, along which, the coeff_var operation to be performed. 
        By default, axis = 0
        
        If axis = 0, coeff_var operation is performed column-wise,
        and for axis = 1, the operation is performed row-wise.
        
        Output:
        Output is float64
        For 1-D array, a number is returned.
        For 2-D array, an array with coeff_var values is returned.
        For 2-D array with axis = 0, output dimension is equal to number of columns.
        For 2-D array with axis = 1, output dimension is equal to number of rows.
        
        Example:
        a = [[1,2,3],[4,5,6]]
        a.shape = (2,3)
        Featurizer().coeff_var(a,0) = array([0.6, 0.42857143, 0.33333333]), shape = (3,) which is equal to number of columns
        Featurizer().coeff_var(a,1) = array([0.40824829, 0.16329932]), shape = (2,) which is equal to number of rows
        
        """
        try:
            ans = self.scipy.stats.variation(a,axis)
            return ans
        except :
            print('Error: Axis value can only be 0 or 1')