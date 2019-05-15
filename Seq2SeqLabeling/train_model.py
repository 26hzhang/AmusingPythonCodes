import tensorflow as tf
from .model.config import Config
from .model.seq2seq_model import SequenceToSequence
from .utils.data_utils import batchnize_dataset, load_dataset


# Media dataset

def create_configurations():
    # dataset parameters
    tf.flags.DEFINE_string("vocab", "dataset/vocab.json", "path to the word and tag vocabularies")
    tf.flags.DEFINE_string("train_set", "dataset/train.json", "path to the training datasets")
    tf.flags.DEFINE_string("dev_set", "dataset/dev.json", "path to the development datasets")
    tf.flags.DEFINE_string("test_set", "dataset/test.json", "path to the test datasets")
    # network parameters
    tf.flags.DEFINE_string("cell_type", "lstm", "RNN cell for encoder and decoder: [lstm | gru], default: lstm")
    tf.flags.DEFINE_string("attention", "bahdanau", "attention mechanism: [bahdanau | luong], default: bahdanau")
    tf.flags.DEFINE_boolean("top_attention", True, "apply attention mechanism only on the top decoder layer")
    tf.flags.DEFINE_boolean("use_bi_rnn", False, "apply bidirectional RNN before encoder to process input embeddings")
    tf.flags.DEFINE_integer("num_units", 128, "number of hidden units in each layer")
    tf.flags.DEFINE_integer("num_layers", 2, "number of layers for encoder and decoder")
    tf.flags.DEFINE_integer("emb_dim", 200, "embedding dimension for encoder and decoder input words")
    tf.flags.DEFINE_boolean("use_beam_search", False, "use beam search strategy for decoder")
    tf.flags.DEFINE_integer("beam_size", 5, "beam size")
    tf.flags.DEFINE_boolean("use_dropout", False, "use dropout for rnn cells")
    tf.flags.DEFINE_boolean("use_residual", False, "use residual connection for rnn cells")
    tf.flags.DEFINE_boolean('use_attention_input_feeding', True, 'Use input feeding method in attentional decoder')
    tf.flags.DEFINE_integer("maximum_iterations", 300, "maximum iterations while decoder generates outputs")
    # training parameters
    tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
    tf.flags.DEFINE_string("optimizer", "adam", "Optimizer: [adagrad | sgd | rmsprop | adadelta | adam], default: adam")
    tf.flags.DEFINE_boolean("use_lr_decay", False, "apply learning rate decay for each epoch")
    tf.flags.DEFINE_float("lr_decay", 0.95, "learning rate decay factor")
    tf.flags.DEFINE_float("grad_clip", 1.0, "maximal gradient norm")
    tf.flags.DEFINE_float("keep_prob", 0.7, "dropout keep probability while training")
    tf.flags.DEFINE_integer("batch_size", 32, "batch size")
    tf.flags.DEFINE_integer("epochs", 100, "train epochs")
    tf.flags.DEFINE_integer("max_to_keep", 5, "maximum trained model to be saved")
    tf.flags.DEFINE_integer("no_imprv_tolerance", 5, "no improvement tolerance")
    tf.flags.DEFINE_string("checkpoint_path", "ckpt/", "path to save model checkpoints")
    tf.flags.DEFINE_string("summary_path", "ckpt/summary/", "path to save summaries")
    tf.flags.DEFINE_string("predict_path", "ckpt/predict/", "path to save predicted valid and test results")
    return tf.flags.FLAGS.flag_values_dict()


print("Build configurations...")
tf_config = create_configurations()
config = Config(tf_config)

print("Load datasets...")
train_data = load_dataset(tf_config["train_set"])
valid_set = batchnize_dataset(tf_config["dev_set"], config.source_dict, config.target_dict, tf_config["batch_size"],
                              shuffle=False)
test_set = batchnize_dataset(tf_config["test_set"], config.source_dict, config.target_dict, tf_config["batch_size"],
                             shuffle=False)
valid_data = batchnize_dataset(tf_config["dev_set"], config.source_dict, config.target_dict, shuffle=False)

print("Build model...")
model = SequenceToSequence(config, mode="train")
model.train(train_data, valid_set, test_set, valid_data)
