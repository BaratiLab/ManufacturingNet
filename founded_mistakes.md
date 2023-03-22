#### 1. Doc for featurizer, is wrong 
  
   Current: 
```
from ManufacturingNet.Featurization import Featurizer
```
Correct: 
 ```
 from ManufacturingNet.featurization import Featurizer
 ```
###### 2. For DNN_tutorial, replaced the class name DNN with DNNBase
###### 3. Not unzipping for CNN_2D_image_tutorials.ipynb, downloaded using gdown works
###### 4. Not unzipping for CNN_2D_Signal.ipynb, downloaded using gdown works
###### 5. Unzip requires high memory for cnn lstm
###### 6. For CNN_LSTM_tutorial, shape[0] variable accessed before assignment, in line 519 of CNN_LSTM_Default.py, changed that to conv_shape, from shape[0].
###### 7. For LSTM_tutorial, replaced the class name LSTM with LSTMBase
###### 8. For signal_processing.py: changed reshape to keepdims=True
###### 9. For CNN3D_tutorial, CNN_LSTM_tutorial replace Lithography files with Lithography in np.load
###### 10. changed all dowload to download_with_gdown