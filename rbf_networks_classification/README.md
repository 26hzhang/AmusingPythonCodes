# Simple RBF Neural Networks for Two-class Classification Example
![](https://img.shields.io/badge/Python-3.6.1-brightgreen.svg) ![](https://img.shields.io/badge/Tensorflow-1.0.0-yellowgreen.svg)

It is a simple example of using RBF neural network to deal with two-class classification tasks. This task is implemented by [tensorflow](https://github.com/tensorflow/tensorflow). The directory contains four python files:
- [kmeans.py](https://github.com/IsaacChanghau/AmusingPythonCodes/tree/master/rbf_networks_classification/kmeans.py), which is implemented to find the centre vectors of hidden neurons.
- [rbf.py](https://github.com/IsaacChanghau/AmusingPythonCodes/tree/master/rbf_networks_classification/rbf.py) is the core model to achieve the RBF neural network.
- [execute.py](https://github.com/IsaacChanghau/AmusingPythonCodes/tree/master/rbf_networks_classification/execute.py), the main function to load data in `./data/` directory, `kmeans.py` and `rbf.py`, then train and test.
- [validation_mlp.py](https://github.com/IsaacChanghau/AmusingPythonCodes/tree/master/rbf_networks_classification/validation_mlp.py), since the testing dataset in `./data/` directory does not contain label information, so this multi-layer perceptron method is used to train a model and generate the testing results, then compare with the derived results via defined RBF neural network.

For the explanation and details of RBF networks, referring the following articles:
- [RBF Network](http://shomy.top/2017/02/26/rbf-network/)
- [RBF network for backpropagation supervised training](http://blog.csdn.net/zouxy09/article/details/13297881)

To train and test the RBF model, run:
```bash
$ python execute.py
```
To train and test the multi-layer perceptron model, run:
```bash
$ python validation_mlp.py
```