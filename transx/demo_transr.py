# coding:utf-8
import os
import ctypes
from models.transr_model import TransR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings

ll = ctypes.cdll.LoadLibrary
lib = ll('./c_codes/init.so')
test_lib = ll('./c_codes/test.so')

# set training and testing dataset
lib.setInPath('./data/FB15K/')
test_lib.setInPath('./data/FB15K/')

lib.init()
test_lib.init()

# set hyperparameters
batch_size = lib.getTripleTotal() // 100
entity_size = lib.getEntityTotal()
relation_size = lib.getRelationTotal()
hidden_size_e = 100
hidden_size_r = 100

# build model
transr = TransR(entity_size, relation_size, hidden_size_e, hidden_size_r, margin=1.0, learning_rate=0.001, l1_flag=True)

# train model
sess = transr.train(lib, num_steps=3000, batch_size=batch_size, sess=None)

# test model
transr.test(test_lib, sess)
