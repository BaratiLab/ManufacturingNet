***************
Getting Started
***************

Requirements
============

To use ManufacturingNet, you will need a version of `Python <https://www.python.org/downloads/>`_ greater than 3.4
installed. 

To check if Python3 is installed, open the terminal on Linux/MacOS or PowerShell on Windows and run the following
command:

.. code::

    python3 --version

To install ManufacturingNet and its dependencies, you will need `pip <https://pip.pypa.io/en/stable/>`_, the Python
package manager. If you have a version of Python greater than 3.4, pip should already be installed.

To check if pip is installed, open the terminal/PowerShell and run the following command:

.. code::

    pip --version

ManufacturingNet depends on the following packages:

- `Matplotlib <https://matplotlib.org/>`_
- `NumPy <https://numpy.org/>`_
- `Pillow <https://python-pillow.org/>`_
- `PyTorch <https://pytorch.org/>`_
- `SciPy <https://www.scipy.org/>`_
- `Scikit-Learn <https://scikit-learn.org/stable/>`_
- `XGBoost <https://xgboost.readthedocs.io/en/latest/>`_

These packages will be automatically installed when you install ManufacturingNet.

Handling Import Errors
----------------------

The above packages should be all you need to run ManufacturingNet, but if you run into errors like
``ImportError: No module named ModuleName``, try installing the module with pip like so:

.. code::

    pip install ModuleName

Installation
============

After installing the above requirements, open the terminal/PowerShell and run the following command:

.. code::

    pip install -i https://test.pypi.org/simple/ ManufacturingNet

Usage
=====

To start using ManufacturingNet in any Python environment, import the library as such:

.. code-block:: python

    import ManufacturingNet

If you don't need the entire library, you can import specific classes using dot notation and "from" statements. For
example, to import the linear regression model, use this code:

.. code-block:: python

    from ManufacturingNet.models import LinRegression

To import the feature extraction functionality, use this code:

.. code-block:: python

    from ManufacturingNet.featurization_and_preprocessing import Featurizer

For more information about using ManufacturingNet, view the documentation page for the desired functionality:

   .. toctree::
      :maxdepth: 1

      Datasets <datasets>
      Featurizers <featurizers>
      Shallow Learning Methods <shallow_learning_methods>
      Deep Learning Methods <deep_learning_methods>