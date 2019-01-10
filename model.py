# Warning Filter
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from config import *

class adict(dict):
    ''' Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
        One can use attributes to read/write dictionary content.
    '''
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

def conv2d(input_char, filter_num, filter_width, name="conv2d"):
    # --------------------------- Input --------------------------- #
    # input_char : shape of (batch_size * trunc_step, 1, word_maxlen, char_dimension)
    # filter_num : [25, 50, 75, 100, 125, 150] subscribed in paper
    # filter_width : [1,2,3,4,5,6] subscribed in paper.

    # --------------------------- Output --------------------------- #
    # convolution output, filtered with each kernel size and width
    with tf.variable_scope(name):
        w = tf.get_variable(name="filters", shape=[1, filter_width, FLAGS.dimension, filter_num])
        b = tf.get_variable(name='filters_bias', shape=[filter_num])

    return tf.nn.conv2d(input_char, w, strides=[1,1,1,1], padding='VALID') + b

def TDNN(input_embedded, scope='TDNN'):
    # --------------------------- Input --------------------------- #
    # input_embedded : shape of (batch_size * trunc_step, word_maxlen, char_dimension)
    # example : (700, 21, 15)

    # --------------------------- Output --------------------------- #
    # output : Concatenated version of max-over-time pooling layer

    #word_maxlen = tf.shape(input_embedded)[1]
    #dimension   = tf.shape(input_embedded)[2]
    word_maxlen = input_embedded.get_shape()[1]
    dimension   = input_embedded.get_shape()[2]

    # make channel as 1 with expand_dim
    input_char = tf.expand_dims(input_embedded, axis=1)

    layers = list()
    with tf.variable_scope(scope):
        for kernel_size, kernel_feature_size in zip(FLAGS.kernel_width, FLAGS.kernel_features):

            conv = conv2d(input_char, kernel_feature_size, kernel_size, name="kernel_%d" % kernel_size)

            pool = tf.nn.max_pool(tf.tanh(conv), ksize=(1,1,word_maxlen-kernel_size + 1,1), strides=[1,1,1,1], padding='VALID')

            # get feature map when needed
            layers.append(tf.squeeze(pool, axis=[1,2]))

        output = tf.concat(layers, axis=1)

    return output

def Affine_Transformation(input_highway, output_dim, scope="AffineTrans"):
    with tf.variable_scope(scope):
        W = tf.get_variable(name="highway_matrix", shape=[output_dim, input_highway.get_shape()[1]], dtype=tf.float32)
        b = tf.get_variable(name="highway_bias", shape=[output_dim], dtype=tf.float32)

    return tf.matmul(input_highway, tf.transpose(W)) + b

def Highway(input_highway, bias=-2.0, scope="Highway"):
    output_dim = input_highway.get_shape()[1] # 525 - sum of filter numbers
    with tf.variable_scope(scope):
        for i in range(FLAGS.Highway_layers):
            t = tf.sigmoid(Affine_Transformation(input_highway, output_dim, scope='Highway_transgate_%d' % i) + bias)
            g = tf.nn.relu(Affine_Transformation(input_highway, output_dim, scope='Highway_MLP_%d'%i) + bias)

            z = t * g + (1.0 - t) * input_highway

            if FLAGS.Highway_layers > 1:
                input_highway = z

    return z

def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.LSTM_hidden, forget_bias=0.0, reuse=False)
    if FLAGS.dropout > 0.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - FLAGS.dropout)
    return cell

def model_graph(word_maxlen=None, char_size=51, word_size=10000):
    # --------------------------- Input --------------------------- #
    # input_embedded : shape of (batch_size * trunc_step, word_maxlen, char_dimension)
    # example : (700, 21, 15)

    # --------------------------- Output --------------------------- #

    input_char = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, FLAGS.trunc_step, word_maxlen], name="input")

    # --------------------------- Embedding Characters --------------------------- #
    with tf.variable_scope("embedding"):
        char_embedding = tf.get_variable(name='char_embedding', shape=[char_size, FLAGS.dimension],  dtype=tf.float32)

        char_embedding_padded = tf.scatter_update(ref=char_embedding, indices=[0], \
                                                  updates=tf.constant(0.0, dtype=tf.float32, shape=[1, FLAGS.dimension]))

        # (batch_size, trunc_step, word_maxlen, char_dimension)
        input_embedded = tf.nn.embedding_lookup(char_embedding, input_char)

        # (batch_size * trunc_step, word_maxlen, char_dimension)
        input_embedded = tf.reshape(input_embedded, shape=[-1, word_maxlen, FLAGS.dimension])

    # --------------------------- Convolutional layer --------------------------- #
    input_highway = TDNN(input_embedded)

    # --------------------------- Highway Network --------------------------- #
    output_highway = Highway(input_highway, bias=-2.0)

    # --------------------------- LSTM Network --------------------------- #
    output_highway = tf.reshape(output_highway, shape=[FLAGS.batch_size, FLAGS.trunc_step, -1]) # (20, 35, 525)
    input_lstm = [tf.squeeze(data, [1]) for data in tf.split(output_highway, FLAGS.trunc_step, axis=1)] # (20, 525) * 35

    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(FLAGS.LSTM_layers)])
    initial_state = stacked_lstm.zero_state(FLAGS.batch_size, tf.float32)

    outputs, end_state = tf.contrib.rnn.static_rnn(stacked_lstm, inputs=input_lstm, initial_state=initial_state, dtype=tf.float32)

    logits = list()
    with tf.variable_scope("word_embedding") as scope:
        for i, output in enumerate(outputs):
            # output shape : (20, 300)
            if i != 0:
                scope.reuse_variables() # not to make new set of variables
            logits.append(Affine_Transformation(output, word_size))


    return adict(input = input_char,
                char_embedding_padded = char_embedding_padded,
                input_embedded = input_embedded,
                input_highway = input_highway,
                output_highway = output_highway,
                lstm_initial_state = initial_state,
                lstm_end_state = end_state,
                lstm_outputs = outputs,
                logits = logits)


def loss_fn(logits):
    with tf.variable_scope('Loss_function'):
        target = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, FLAGS.trunc_step], name='target_')
        targets = [tf.squeeze(data, [1]) for data in tf.split(target, FLAGS.trunc_step, axis=1)]
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits), name='loss_')

    return adict(targets=target,
                loss = loss)


def train_fn(loss_dict):
    global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)

    with tf.variable_scope('Training'):
        learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False, name='learning_rate_')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        train_var = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss_dict, train_var), clip_norm=FLAGS.grad_norm)
        train_op = optimizer.apply_gradients(zip(grads, train_var), global_step=global_step)

    return adict(learning_rate = learning_rate,
                global_step = global_step,
                grads=grads,
                train_var = train_var,
                train_op = train_op)
