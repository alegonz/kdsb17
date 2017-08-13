##### Kaggle Data Science Bowl 2017
Detailed description pending.

The model consists of two submodels:
* 3D convolutional Autoencoder: Extracts high-level features of lung scans (3D arrays of CT scans in Hounsfield Units), using maximum likelihood on a mixture of Gaussians.
* Multi-layer perceptron: Performs binary classification upon the features extracted by the encoding layers of the 3D CAE submodel.

Repository contents:

* **kdsb17**: \
  Contains the custom modules for data pre-processing, and building and training the models.
* **scripts**: \
  Contains the scripts to preprocess the data, train the models and predict.

##### Requirements
* Python 3
* Keras 2.0.6
* tensorflow-gpu 1.2.1
* numpy 1.13.0
* scipy 0.19.1
* pydicom 0.9.9
* CUDA Version 8.0.61
