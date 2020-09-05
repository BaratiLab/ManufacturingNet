from .linear_regression import LinRegression
from .logistic_regression import LogRegression
from .svm import SVM
from .xgb import XGBoost
from .random_forest import RandomForest
from .all_regression_models import AllRegressionModels
from .all_classification_models import AllClassificationModels
from .DNN import DNNModel
from .CNN import CNN2D
from .CNN3D import CNN3D
from .CNN_LSTM_Default import CNNLSTMModel
from .CNN2DImage import CNN2DImageModel
from .LSTM_Default import LSTMModel
from .resnet import ResNet
from .vgg import VGG
from .densenet import DenseNet
from .alexnet import AlexNet
from .googlenet import GoogleNet
from .mobilenet import MobileNet


#__add__=['LinRegression','LogticRegression','MLP','SVM','XGBoost','RandomForest','AllRegressionModels','AllClassificationModels']
__add__=['LinRegression','LogRegression','SVM','XGBoost','RandomForest','AllRegressionModels','AllClassificationModels','DNNModel','CNN2D', 'CNN3D', 'CNNLSTMModel', 'CNN2DImageModel', 'LSTMModel', 'ResNet','VGG','DenseNet','AlexNet','MobileNet','GoogleNet']
