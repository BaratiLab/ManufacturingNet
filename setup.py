from setuptools import setup,find_packages

package_name ='DeepManufacturing'

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

classifiers=[
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3 :: Only',
    'Intended Audience :: Education']

setup(name=package_name,\
      version='0.0.3',\
      description='AI and Machine Learning for manufacturing related datasets',
      long_description=LONG_DESCRIPTION,
      author='Amir Barati Farimani',
      author_email='barati@andrew.cmu.edu',
      license='MIT',
      classifiers=classifiers,
      keywords=' ',
      python_requires='>=3.5, <4',
      packages=find_packages(),
      zip_safe=False)
