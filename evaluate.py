import os
import time

import tensorflow as tf
import numpy as np

from preprocess import *
from config import *
from model import *


def main(_):

    if not os.path.exists(FLAGS.load_model + '.index'):
        print ('Checkpoint file does not exists in train_dir')
        return -1

    word_vocab, char_vocab, word_matrix, char_matrix, label_data,\
    word_maxlen, _, _, zip_dict = preprocessing(FLAGS)

    with tf.Graph().as_default(), tf.Session() as sess:
        tf.set_random_seed(1170)
        np.random.seed(seed=1170)

        with tf.variable_scope("Model"):
            ts_model = model_graph(word_maxlen=word_maxlen)

            loss_dict = loss_fn(ts_model.logits)
            ts_model.update(loss_dict)
            global_step = tf.Variable(0, dtype=tf.int32, name='global_step')

        saver = tf.train.Saver()
        #saver.restore(sess, tf.train.latest_checkpoint('./train_dir/'))
        saver.restore(sess, FLAGS.load_model)

        ts_count = 0
        test_loss = 0.0
        lstm_state = sess.run(ts_model.lstm_initial_state)
        start_time = time.time()
        for x, y in iter_(zip_dict, 'test'):
            input_ = {ts_model.input : x, ts_model.targets : y, ts_model.lstm_initial_state : lstm_state}
            loss, lstm_state_ = sess.run([ts_model.loss, ts_model.lstm_end_state], input_)
            test_loss += loss
            ts_count += 1

            if ts_count % FLAGS.verbose == 0:
                print ('count : %d, time : %.3f' % (ts_count, time.time() - start_time))
                
        # average the loss
        test_loss /= ts_count
        print ('test loss = %.3f, perplexity = %.3f' % (test_loss, np.exp(test_loss)))
        print ('Total time = %.3f' % (time.time() - start_time))


if __name__ == "__main__":
    tf.app.run()
