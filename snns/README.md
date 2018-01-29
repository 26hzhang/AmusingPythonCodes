# Self-Normalizing Networks

**Note**: Codes are modified to fit Python 3.6 and Tensorflow 1.4.

Original repository: [bioinf-jku/SNNs](https://github.com/bioinf-jku/SNNs)

Tutorials and implementations for ["Self-normalizing networks"(SNNs)](https://arxiv.org/pdf/1706.02515.pdf) as suggested by Klambauer et al.

## Versions
- Python 3.6 and Tensorflow 1.4

## Note for Tensorflow 1.4 users
Tensorflow 1.4 already has the function "tf.nn.selu" and "tf.contrib.nn.alpha_dropout" that implement the SELU activation function and the suggested dropout version. 

## Tutorials
- Multilayer Perceptron ([notebook](snns_mlp_mnist.py))
- Convolutional Neural Network on MNIST ([notebook](snns_cnn_mnist.py))
- Convolutional Neural Network on CIFAR10 ([notebook](snns_cnn_cifar10.py))

## KERAS CNN scripts:
- KERAS: Convolutional Neural Network on MNIST ([python script](keras-cnn/MNIST-Conv-SELU.py))
- KERAS: Convolutional Neural Network on CIFAR10 ([python script](keras-cnn/CIFAR10-Conv-SELU.py))


## Design novel SELU functions
- How to obtain the SELU parameters alpha and lambda for arbitrary fixed points ([python codes](get_selu_parameters.py))

## Basic python functions to implement SNNs
are provided as code chunks here: [selu.py](selu.py)

## Notebooks and code to produce Figure 1 in Paper
are provided here: [Figure1](/figure1)

## Calculations and numeric checks of the theorems
- [Mathematica PDF](calculations-notes/SELU_calculations.pdf)

## UCI, Tox21 and HTRU2 data sets
- [UCI - download from original source](http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/data.tar.gz)
- [UCI - download processed version of the data set](http://www.bioinf.jku.at/people/klambauer/data_py.zip)
- [Tox21](http://bioinf.jku.at/research/DeepTox/tox21.zip)
- [HTRU2](https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip)

## Models and architectures built on Self-Normalizing Networks
### GANs
- [THINKING  LIKE  A  MACHINE - GENERATING  VISUAL RATIONALES WITH WASSERSTEIN GANS](https://pdfs.semanticscholar.org/dd4c/23a21b1199f34e5003e26d2171d02ba12d45.pdf): Both discriminator and generator trained without batch normalization.
- [Deformable Deep Convolutional Generative Adversarial Network in Microwave Based Hand Gesture Recognition System](https://arxiv.org/abs/1711.01968): The  rate  between  SELU  and  SELU+BN proves  that  SELU  itself  has  the  convergence  quality  of  BN.

### Convolutional neural networks
- [Solving internal covariate shift in deep learning with linked neurons](https://arxiv.org/abs/1712.02609): Show that ultra-deep CNNs without batch normalization can only be trained SELUs (except with the suggested method described by the authors).
- [DCASE 2017 ACOUSTIC SCENE CLASSIFICATION USING CONVOLUTIONAL NEURAL NETWORK IN TIME SERIES](http://www.cs.tut.fi/sgn/arg/dcase2017/documents/challenge_technical_reports/DCASE2017_Biho_116.pdf): Deep CNN trained without batch normalization.
- [Point-wise Convolutional Neural Network](https://arxiv.org/abs/1712.05245):  Training with SELU converges faster than training with ReLU; improved accuracy with SELU.
- [Over the Air Deep Learning Based Radio Signal Classification](https://arxiv.org/abs/1712.04578): Slight performance improvement over ReLU.
- [Convolutional neural networks for structured omics: OmicsCNN and the OmicsConv layer](https://arxiv.org/abs/1710.05918): Deep CNN trained without batch normalization.
- [Searching for Activation Functions](https://arxiv.org/abs/1710.05941): ResNet architectures trained with SELUs probably together with batch normalization.
- [EddyNet: A Deep Neural Network For Pixel-Wise Classification of Oceanic Eddies](https://arxiv.org/abs/1711.03954): Fast CNN training with SELUs. ReLU with BN better at final performance but skip connections not handled appropriately.
- [SMILES2Vec: An Interpretable General-Purpose Deep Neural Network for Predicting Chemical Properties](https://arxiv.org/abs/1712.02034): 20-layer ResNet trained with SELUs.
- [Sentiment Analysis of Tweets in Malayalam Using Long Short-Term Memory Units and Convolutional Neural Nets](https://link.springer.com/chapter/10.1007/978-3-319-71928-3_31)
- [RETUYT in TASS 2017: Sentiment Analysis for Spanish Tweets using SVM and CNN](https://arxiv.org/abs/1710.06393)

### FNNs are finally deep
- [Predicting Adolescent Suicide Attempts with Neural Networks](https://arxiv.org/abs/1711.10057): The use of the SELU activation renders batch normalization
unnecessary.
- [Improving Palliative Care with Deep Learning](https://arxiv.org/abs/1711.06402): An 18-layer neural network with SELUs performed best.
- [An Iterative Closest Points Approach to Neural Generative Models](https://arxiv.org/abs/1711.06562)
- [Retrieval of Surface Ozone from UV-MFRSR Irradiances using Deep Learning](http://uvb.nrel.colostate.edu/UVB/publications/AGU-Retrieval-Surface-Ozone-Deep-Learning.pdf): 6-10 layer networks perform best. 

### Reinforcement Learning
- [Automated Cloud Provisioning on AWS using Deep Reinforcement Learning](https://arxiv.org/abs/1709.04305): Deep CNN architecture trained with SELUs.
- [Learning to Run with Actor-Critic Ensemble](https://arxiv.org/abs/1712.08987): Second best method (actor-critic ensemble) at the NIPS2017 "Learning to Run" competition. They have
tried several activation functions and found that the activation function of Scaled Exponential Linear Units (SELU) are superior to ReLU, Leaky ReLU, Tanh and Sigmoid.

## Autoencoders
- [Replacement AutoEncoder: A Privacy-Preserving Algorithm for Sensory Data Analysis](https://arxiv.org/abs/1710.06564): Deep autoencoder trained with SELUs.
- [Application of generative autoencoder in de novo molecular design](https://arxiv.org/abs/1711.07839): Faster convergence with SELUs.

## Recurrent Neural Networks
- [Sentiment extraction from Consumer-generated noisy short texts](http://sentic.net/sentire2017meisheri.pdf): SNNs used in FC layers.
