# TransX · Embedding Entities and Relationships of Multi-relational Data
![](https://img.shields.io/badge/Python-2.7.13-brightgreen.svg) ![](https://img.shields.io/badge/Tensorflow-0.12.0-yellowgreen.svg)

This repository is converted from [thunlp/TensorFlow-TransX](https://github.com/thunlp/TensorFlow-TransX), which is a light and simple version of OpenKE based on TensorFlow, including TransE, TransH, TransR and TransD. TransX methods are used for knowledge representation learning (KRL), which encode information of relational knowledge into vectors and expand its usage into many different fields.

This repository is just a practice codes after reading several KRL related papers to help understand the process better. For more details and information on this field, please refer: [Natural Language Processing Lab at Tsinghua University -- GitHub page](https://github.com/thunlp).

Dataset can be download from [[here]](https://github.com/thunlp/TensorFlow-TransX/tree/master/data).

In this repository, the codes structure is different from the original one, some modifications and adjustments are made to let the codes more understandable and easily to use.

**TransH Training Example**:
```bash
Input Files Path : ./data/FB15K/
Input Files Path : ./data/FB15K/

Building Graph...Done...
Training started...
  step   1, loss: 2128.2130
  step   2, loss: 1002.4949
  step   3, loss: 648.3365
  ...
  step 399, loss:  18.6965
  step 400, loss:  19.4174
  step 401, loss:  19.2409
  ...
Done...
```

**Useful Information and Reference**:
- [Natural Language Processing Lab at Tsinghua University -- GitHub page](https://github.com/thunlp).
- [TransE -- Translating embeddings for modeling multi-relational data](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf).
- [TransH -- Knowledge Graph Embedding by Translating on Hyperplanes](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.486.2800&rep=rep1&type=pdf).
- [TransR -- Learning Entity and Relation Embeddings for Knowledge Graph Completion](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.698.8922&rep=rep1&type=pdf).
- [TransD -- Knowledge Graph Embedding via Dynamic Mapping Matrix](http://www.aclweb.org/anthology/P15-1067).
- [thunlp/TensorFlow-TransX](https://github.com/thunlp/TensorFlow-TransX): light and simple version of OpenKE based on TensorFlow, including TransE, TransH, TransR and TransD.
- [thunlp/Fast-TransX](https://github.com/thunlp/Fast-TransX): efficient lightweight C++ inferences for TransE and its extended models utilizing the framework of OpenKE, including TransH, TransR, TransD, TranSparse and PTransE.
- [thunlp/OpenKE](https://github.com/thunlp/OpenKE): an open-source package for knowledge embedding.
- [thunlp/TensorFlow-NRE](https://github.com/thunlp/TensorFlow-NRE): neural relation extraction implemented with LSTM in tensorflow.
- [thunlp/TransNet](https://github.com/thunlp/TransNet), source code and datasets of IJCAI2017 paper "[TransNet: Translation-Based Network Representation Learning for Social Relation Extraction](https://www.ijcai.org/proceedings/2017/0399.pdf)".
- [thunlp/KB2E](https://github.com/thunlp/KB2E): knowledge graph embeddings including TransE, TransH, TransR and PTransE.