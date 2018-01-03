# Seq2Seq for Translation or Dialogue (2)
![](https://img.shields.io/badge/Python-3.6.1-brightgreen.svg) ![](https://img.shields.io/badge/Tensorflow-1.0.0-yellowgreen.svg)

Codes from [suriyadeepan/practical_seq2seq](https://github.com/suriyadeepan/practical_seq2seq).

Details about the codes: [Suriyadeepan Ram -- Practical seq2seq](http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/).

In the original repository, the author trained the seq2seq model on several datasets and showed the results.

**Codes structures are changed and some codes are modified to fit tensorflow-1.0.0 (original is 0.12.0)**

The dataset used:

1. [CMU Pronouncing Dictionary](https://github.com/suriyadeepan/datasets/tree/master/seq2seq/CMUdict): Phoneme sequence to word (sequence of alphabets)
2. [Twitter Chat Log](https://github.com/suriyadeepan/datasets/tree/master/seq2seq/twitter): Sentence to Sentence
3. [Cornell Movie Dialog Corpus](https://github.com/suriyadeepan/datasets/tree/master/seq2seq/cornell_movie_corpus): Sentence tp Sentence

Since the (2) and (3) are similar tasks, so here I only studied the first two and copied the codes here.

The training dataset (after preprocessed) and checkpoint data are available on the original repository. Or using the shell file (pull) to download them.

For training process is test on **Python3.6 + Tensorflow v1.0.0**
