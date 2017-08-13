from setuptools import setup

install_requires = [
    'tensorflow-gpu==1.2.1',
    'Keras==2.0.6',
    'numpy==1.13.0',
    'scipy==0.19.1',
    'pydicom==0.9.9'
]

setup(
    name='kdsb17',
    description='Kaggle Data Science Bowl 2017',
    version='0.1',
    packages=['kdsb17'],
    install_requires=install_requires
)