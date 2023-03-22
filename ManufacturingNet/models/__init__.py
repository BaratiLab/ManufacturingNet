from .linear_regression import LinRegression
from .logistic_regression import LogRegression
from .svm import SVM
from .xgb import XGBoost
from .random_forest import RandomForest
from .all_regression_models import AllRegressionModels
from .all_classification_models import AllClassificationModels
from .DNN import DNN
from .CNN import CNN2DSignal
from .CNN3D import CNN3D
from .CNN_LSTM_Default import CNNLSTM
from .CNN2DImage import CNN2DImage
from .LSTM_Default import LSTM
from .resnet import ResNet
from .vgg import VGG
from .densenet import DenseNet
from .alexnet import AlexNet
from .googlenet import GoogleNet
from .mobilenet import MobileNet
# from .DNN import DNNUser


#__add__=['LinRegression','LogticRegression','MLP','SVM','XGBoost','RandomForest','AllRegressionModels','AllClassificationModels']
__add__ = ['LinRegression',
           'LogRegression',
           'SVM','XGBoost',
           'RandomForest',
           'AllRegressionModels',
           'AllClassificationModels',
           'DNN',
           'CNN2DSignal', 
           'CNN3D', 
           'CNNLSTM', 
           'CNN2DImage', 
           'LSTM', 
           'ResNet',
           'VGG',
           'DenseNet',
           'AlexNet',
           'MobileNet',
           'GoogleNet']
