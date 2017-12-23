## Seq2Seq for Translation or Dialogue
This is an implementation of Seq2Seq to make it easy to train the seq2seq model with any corpus for translation or dialogue task.

The original codes are available here: [[suriyadeepan/easy_seq2seq]](https://github.com/suriyadeepan/easy_seq2seq). 

If you using the provided sample dataset, the `unicode error` may happens. thus, before training, you should follow the suggestions of `UnicodeError` issue to solve it: [[link]](https://github.com/suriyadeepan/easy_seq2seq/issues/31), or using the following command to convert all of the training/testing corpus to `utf-8` format: 
``` bash
$ iconv -f WINDOWS-1252 -t UTF-8//TRANSLIT old_files -o new_files
```
To train the model, Create temporary working directory prior to training:
``` bash
$ mkdir working_dir
```
Then Download test/train data from Cornell Movie Dialog Corpus and convert to `utf-8` format (as described before):
``` bash
$ cd data/
$ ./pull_data.sh
$ iconv -f WINDOWS-1252 -t UTF-8//TRANSLIT old_files -o new_files
```
Finally training or testing:
``` bash
# edit seq2seq.ini file to set: mode = train
python seq2seq_main.py
# edit seq2seq.ini file to set: mode = test
python seq2seq_main.py
```

**Test on Python3.6 + tensorflow==0.12.1**
