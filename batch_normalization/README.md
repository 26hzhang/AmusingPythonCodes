# Batch Normalization
Implementation of Batch Normalization in Tensorflow, Batch Normalization is a strategy to address the problem of _internal covariate shift_, the description of _internal covariate shift_ is that for deep neural networks, the distribution of each layer’s inputs changes during training, as the parameters of the previous layers change, which slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities.

Batch Normalization draws its strength from making normalization a part of the model architecture and performing the normalization for each training mini-batch, which also allows to use much higher learning rates and be less careful about initialization, as well as acts as a regularizer, in some cases eliminating the need for Dropout.

This repository is inspired by the paper [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) and [tomokishii/mnist_cnn_bn.py](https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412). Improvement and modifications are done to make the codes more readable and flexible.

More details about Batch Normalization please go through the [Reference](#Reference), which contains the information I read when I learned this technique.

### Training Results:
```bash
 Training...
  step     0: validation loss = 22.7861, validation accuracy = 0.1146
  step   200: validation loss = 0.1514, validation accuracy = 0.9562
  step   400: validation loss = 0.1149, validation accuracy = 0.9684
  ...
  step  1600: validation loss = 0.0627, validation accuracy = 0.9800
  step  1800: validation loss = 0.0553, validation accuracy = 0.9836
  step  2000: validation loss = 0.0501, validation accuracy = 0.9852
  ...
  step  3800: validation loss = 0.0426, validation accuracy = 0.9880
  step  4000: validation loss = 0.0388, validation accuracy = 0.9894
  step  4200: validation loss = 0.0373, validation accuracy = 0.9900
  ...
  step  5000: validation loss = 0.0342, validation accuracy = 0.9898

 Testing...
  test loss = 0.0295, test accuracy = 0.9894, multiclass log loss = 0.0295
```

### Reference
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
- [tomokishii/mnist_cnn_bn.py](https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412)
- [stackoverflow/How could I use Batch Normalization in TensorFlow?](https://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow)
- [Rui Shu/TENSORFLOW GUIDE: BATCH NORMALIZATION](http://ruishu.io/2016/12/27/batchnorm/)
- [tensorflow/tf.nn.batch_normalization](https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization)
- [Why does batch normalization help?](https://www.quora.com/Why-does-batch-normalization-help)
- [神经网络梯度与归一化问题总结+highway network、ResNet的思考](https://zhuanlan.zhihu.com/p/26076292)
- [深度学习中 Batch Normalization为什么效果好？](https://www.zhihu.com/question/38102762)
- [深度学习（二十九）Batch Normalization 学习笔记](http://blog.csdn.net/hjimce/article/details/50866313)
- [《Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift》阅读笔记与实现](http://blog.csdn.net/happynear/article/details/44238541)