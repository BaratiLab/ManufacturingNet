# ManufacturingNet

[Website](http://manufacturingnet.io/) | [Documentation](https://manufacturingnet.readthedocs.io/en/latest/)

ManufacturingNet provides a sustainable, open-source ecosystem of modern artificial intelligence (AI) tools for tackling diverse engineering challenges.

Written in Python3 and designed for ease of use, ManufacturingNet's machine learning library simplifies AI for manufacturing professionals.

ManufacturingNet is developed and maintained by the Mechanical and AI Lab (MAIL) at Carnegie Mellon University.

For more information, visit our website, [manufacturingnet.io.](http://manufacturingnet.io/)

## Requirements

To use ManufacturingNet, you will need a version of [Python](https://www.python.org/downloads/) greater than 3.4 installed. 

To check if Python3 is installed, open the terminal on Linux/MacOS or PowerShell on Windows and run the following command:

```bash
python3 --version
```

To install ManufacturingNet and its dependencies, you will need [pip](https://pip.pypa.io/en/stable/), the Python package manager. If you have a version of Python greater than 3.4, pip should already be installed.

To check if pip is installed, open the terminal/PowerShell and run the following command:

```bash
pip --version
```

ManufacturingNet depends on the following packages:
- [Matplotlib](https://matplotlib.org/)
- [NumPy](https://numpy.org/)
- [Pillow](https://python-pillow.org/)
- [PyTorch](https://pytorch.org/)
- [SciPy](https://www.scipy.org/)
- [Scikit-Learn](https://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/)

These packages will be automatically installed when you install ManufacturingNet.

### Handling Import Errors

The above packages should be all you need to run ManufacturingNet, but if you run into errors like `ImportError: No module named ModuleName`, try installing the module with pip like so:

```bash
pip install ModuleName
```

## Installation

After you've installed the above requirements, open the terminal/PowerShell and run the following command:

```bash
pip install DeepManufacturing
```

## Usage

To start using ManufacturingNet in any Python environment, import the library as such:

```python
import ManufacturingNet
```

If you don't need the entire library, you can import specific classes using dot notation and "from" statements. For example, to import the linear regression model, use this code:

```python
from ManufacturingNet.models import LinRegression
```

To import the feature extraction functionality, use this code:

```python
from ManufacturingNet.featurization import Featurizer
```

When in doubt, check the [documentation](https://manufacturingnet.readthedocs.io/en/latest/)!

## License
[MIT](https://choosealicense.com/licenses/mit/)
