# Highway Networks
Implementation of highway networks in Tensorflow, highway networks is a method to construct networks with very deep (hundreds+) layers. Although it performs well in practice, its principle is quite east to understand. 

In this repository, most codes are inspired and borrowed from [lucko515/fully-connected-highway-network](https://github.com/lucko515/fully-connected-highway-network) and [fomorians/highway-cnn](https://github.com/fomorians/highway-cnn) (some modifications and restructured).

`highway_recurrent` folder contains the codes copied from [julian121266/RecurrentHighwayNetworks](https://github.com/julian121266/RecurrentHighwayNetworks) (only tensorflow implementation part, full part (theano, torch version) please go through the original repository). It is an implementation of Recurrent Highway Network (_TODO_)

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is used as training and testing data. Examples are showing in codes. More knowledge and codes about highway networks are put in reference.

### Results
**Full Connected Feed-forward Highway Networks**:
In this example, a 18 layers full connected highway neural networks are built to train and classify MNIST dataset. Training Information is shown below:
```bash
Extracting ../data/MNIST_data/train-images-idx3-ubyte.gz
Extracting ../data/MNIST_data/train-labels-idx1-ubyte.gz
Extracting ../data/MNIST_data/t10k-images-idx3-ubyte.gz
Extracting ../data/MNIST_data/t10k-labels-idx1-ubyte.gz
Epoch: 1/40   |  Current loss: 19.298761   |  Epoch time:  8.33s
Test accuracy 0.9553
Epoch: 2/40   |  Current loss:  8.394491   |  Epoch time:  8.32s
Test accuracy 0.9646
Epoch: 3/40   |  Current loss:  6.591007   |  Epoch time:  8.35s
Test accuracy 0.9651
Epoch: 4/40   |  Current loss:  5.484001   |  Epoch time:  8.31s
Test accuracy 0.9681
...
Epoch: 37/40   |  Current loss:  0.032462   |  Epoch time:  7.59s
Test accuracy 0.9752
Epoch: 38/40   |  Current loss:  0.030767   |  Epoch time:  7.59s
Test accuracy 0.9747
Epoch: 39/40   |  Current loss:  0.028955   |  Epoch time:  7.59s
Test accuracy 0.9745
Epoch: 40/40   |  Current loss:  0.028626   |  Epoch time:  7.67s
Test accuracy 0.9749
Test Accuracy:  0.9749
Validation Accuracy:  0.9722
```

**Convolutional Highway Networks**:
In this example, same, a 18 layers convolutional highway neural networks (with dropout and max-pooling layer, and input layer is traditional conv2d layer, output layer is a dense layer) are built to train and classify MNIST dataset. Training information is shown below:
Training Information:
```bash
step 100, validating accuracy 0.388
step 200, validating accuracy 0.3196
step 300, validating accuracy 0.4354
step 400, validating accuracy 0.8146
step 500, validating accuracy 0.8526
...

step 3900, validating accuracy 0.9808
step 4000, validating accuracy 0.9724
step 4100, validating accuracy 0.9724
step 4200, validating accuracy 0.9772
Test accuracy 0.9592
```

### Reference
- [Highway Networks with TensorFlow (Jim Fleming's blog post)](https://medium.com/jim-fleming/highway-networks-with-tensorflow-1e6dfa667daa)
- [lucko515/fully-connected-highway-network](https://github.com/lucko515/fully-connected-highway-network)
- [fomorians/highway-cnn](https://github.com/fomorians/highway-cnn)
- [Highway Networks](https://arxiv.org/abs/1505.00387)
- [Training Very Deep Networks](https://arxiv.org/abs/1507.06228)
- [Very Deep Learning with Highway Networks](http://people.idsia.ch/~rupesh/very_deep_learning/) (group of highway network related papers and codes)
- [THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/)
- [julian121266/RecurrentHighwayNetworks](https://github.com/julian121266/RecurrentHighwayNetworks)
- [Recurrent Highway Networks](https://arxiv.org/abs/1607.03474)