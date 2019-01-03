import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# preprocess use
flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_string('data_path', './dataset/ptb/', 'English Penn Treebank dataset directory')

flags.DEFINE_list('data_file', ['train', 'valid', 'test'], 'PTB dataset lists')

flags.DEFINE_integer('dimension', 4, 'choose the character dimension')

# train setup use
flags.DEFINE_integer('batch_size', 20, 'DATA-Small batch size')
flags.DEFINE_integer('trunc_step', 35, 'truncated backpropagation step')
flags.DEFINE_integer('epoch', 25, 'train epoch size')

flags.DEFINE_float('learning_rate', 1.0, 'initial learning rate of model')
