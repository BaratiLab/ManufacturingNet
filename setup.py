from setuptools import setup,find_packages

package_name ='ManufacturingNet'

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

classifiers=[
    'Programming Language :: Python :: 3.7',
    'Intended Audience :: Education']

setup(name=package_name,\
    version='0.0.1',\
        description='AI and Machine Learning for manufacturing related datasets',
        long_description=LONG_DESCRIPTION,
        author='Ruchti Doshi',
        author_email='ruchitsd@andrew.cmu.edu',
        license='MIT',
        classifiers=classifiers,
        keywords=' ',
        packages=["ManufacturingNet"],
        zip_safe=False)