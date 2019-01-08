# Warning Filter
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_string('data_path', './dataset/ptb/', 'English Penn Treebank dataset directory')

flags.DEFINE_list('data_file', ['train', 'valid', 'test'], 'PTB dataset lists')
flags.DEFINE_list('kernel_features', [25, 50, 75, 100, 125, 150], 'kernel size as small data')
flags.DEFINE_list('kernel_width', [1,2,3,4,5,6], 'kernel width as small data')

flags.DEFINE_integer('dimension', 15, 'choose the character dimension')
flags.DEFINE_integer('batch_size', 20, 'DATA-Small batch size')
flags.DEFINE_integer('trunc_step', 35, 'truncated backpropagation step')
flags.DEFINE_integer('epoch', 25, 'train epoch size')
flags.DEFINE_integer('LSTM_hidden', 300, 'number of LSTM hidden units')
flags.DEFINE_integer('LSTM_layers', 2, 'number of LSTM layers')
flags.DEFINE_integer('Highway_layers', 1, 'number of Highway layers')

flags.DEFINE_float('learning_rate', 1.0, 'initial learning rate of model')
flags.DEFINE_float('dropout', 0.5, 'initial dropout of LSTM layers')
