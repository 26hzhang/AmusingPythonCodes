# coding:utf-8
import os
import ctypes
from models.transd_model import TransD

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
hidden_size = 50

# build model
transd = TransD(entity_size, relation_size, hidden_size, margin=1.0, learning_rate=0.001, l1_flag=True)

# train model
sess = transd.train(lib, num_steps=500, batch_size=batch_size, sess=None)

# test model
transd.test(test_lib, sess=sess)
