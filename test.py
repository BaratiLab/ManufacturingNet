import sys
sys.path.append('./')
import numpy as np
import torch.nn as nn

class First(nn.Module):
    def __init__(self):
        super(First, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)
    
class First():
    def __init__(self):
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


def test_featurization():
    test_array = np.random.randint(0, 100, (10, 10))
    from ManufacturingNet.featurization import Featurizer
    featurizer = Featurizer()
    test_array_mean = featurizer.mean(test_array, axis=0)
    assert (test_array_mean == np.mean(test_array, axis=0)).all()

def present_documentation():
    # from ManufacturingNet.Featurization import Featurizer
    from ManufacturingNet.Featurization import Featurizer
    # import numpy as np

def logistic_regression():
    from ManufacturingNet.models import LogRegression
    regression = LogRegression()

if __name__ == '__main__':
    present_documentation()
    test_featurization()
    # print(np.__version__)