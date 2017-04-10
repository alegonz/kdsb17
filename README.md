##### Code of first attempt at Kaggle Data Science Bowl 2017
Detailed description pending.

The first attempt consists of a 3-layer 3D convolutional autoencoder to extract high-level features of lung scans (3D arrays of CT scans in Hounsfield Units), and then learn a multi-layer perceptron upon those features for binary classification.

* **kdsb17**: \
  Contains the custom modules for data pre-processing, and building and training the models.
* **scripts**: \
  Contains the scripts to preprocess the data, train the models and predict.

##### Requirements
* Python 3
* Keras 1.2.2
* Theano 0.8.2
* Numpy 1.12.1
* Scipy 0.19
* pydicom 0.9.9
* CUDA Version 7.5.18

Data pre-processing performed using the Kaggle docker image.
Deep learning models were trained in a Amazon Web Services p2.xlarge instance running the Deep Learning AMI Amazon Linux (version 2.0).
