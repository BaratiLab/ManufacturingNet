from setuptools import setup,find_packages

package_name ='ManufacturingNet'

classifiers=[
    'Programming Language :: Python :: 3.7',
    'License :: OSI Approved :: MIT license',
    'Intended Audience :: Education']

setup(name=package_name,\
    version='0.0.1',\
        description='AI and Machine Learning for manufacturing related datasets',
        long_description=open('README.md').read(),
        url-'manufacturingnet.io',
        author='Ruchti Doshi',
        author_email='ruchitsd@andrew.cmu.edu',
        license='MIT',
        classifiers=classifiers,
        keywords=' ',
        packages=find_packages(),
        zip_safe=False)