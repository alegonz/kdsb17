# Attempt at the classification problem featured in the Kaggle Data Science Bowl 2017

##### Notes
* **This is still work in progress.**
* For the source code and requirements please refer to [Repository info]().

## Description

This is an attempt at the classification task featured in the [Kaggle Data Science Bowl 2017](https://www.kaggle.com/c/data-science-bowl-2017). The task consists on predicting from CT lung scans whether a patient will develop cancer or not within a year. This is a particularly challenging problem given the very high dimensionality of data and the very limited number of samples.

The competition saw many creative approaches, such as those reported by the winning entries [here](https://github.com/lfz/DSB2017) (1st place), [here](http://blog.kaggle.com/2017/06/29/2017-data-science-bowl-predicting-lung-cancer-2nd-place-solution-write-up-daniel-hammack-and-julian-de-wit/) (2nd place) and [here](http://blog.kaggle.com/2017/05/16/data-science-bowl-2017-predicting-lung-cancer-solution-write-up-team-deep-breath/) (9th place). These approaches have in common that:

1. they are based on deep CNNs;
2. leverage external data, in particular the [LUNA dataset](https://luna16.grand-challenge.org/);
3. make extensive use of ensemble of models.

What I'm attempting here is a rather more "purist" (for lack of a better word) approach that uses no ensemble models and no external data. The purpose of this is simply to explore the possibility of achieving a decent classification accuracy using a single model and using solely the provided data. This model consists of a combination of two neural networks:

* Gaussian Mixture Convolutional AutoEncoder (GMCAE): Extracts high-level features of lung scans (3D arrays of CT scans in Hounsfield Units), using maximum likelihood on a mixture of Gaussians.
* CNN classifier: Performs binary classification upon the features extracted by the encoding layers of the 3D CAE submodel.

![model_overview](illustrations/model_overview.png "Model overview")

## Data details
![equation] \sqrt[n]{k}

The details of preprocessing are explained [here]().

## Model details

### Gaussian Mixture Convolutional AutoEncoder (GMCAE)

### CNN Classifier

## Current results

## Repository info
### Contents

* **kdsb17**: \
  Contains the custom modules for data pre-processing, and building and training the models.
* **scripts**: \
  Contains the scripts to preprocess the data, train the models and predict.

### Requirements
* Python 3
* Keras 2.0.6
* tensorflow-gpu 1.2.1
* numpy 1.13.0
* scipy 0.19.1
* pydicom 0.9.9
* CUDA Version 8.0.61
