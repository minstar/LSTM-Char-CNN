# Warning Filter
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import numpy as np
import tensorflow as tf

from preprocess import *
from config import *
from model import *

def train():
    # --------------------------- import data --------------------------- #
    print ('----- Importing all Data -----')
    word_vocab, char_vocab, word_matrix, char_matrix, label_data,\
    word_maxlen, _, _, zip_dict = preprocessing(FLAGS)

    # --------------------------- build training graph --------------------------- #
    with tf.Graph().as_default(), tf.Session() as sess:
        tf.set_random_seed(1170)
        np.random.seed(seed=1170)

        initializer = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)
        with tf.variable_scope("Model", initializer=initializer):
            tr_model = model_graph(word_maxlen=word_maxlen, char_size=51, word_size=10000)
            loss_dict = loss_fn(tr_model.logits)
            train_op_dict = train_fn(loss_dict.loss * FLAGS.trunc_step)
            tr_model.update(loss_dict)
            tr_model.update(train_op_dict)

        # --------------------------- build valid graph --------------------------- #
        with tf.variable_scope("Model", reuse=True):
            va_model = model_graph(word_maxlen=word_maxlen, char_size=51, word_size=10000)
            va_loss_dict = loss_fn(va_model.logits)
            va_model.update(va_loss_dict)

        # --------------------------- save model --------------------------- #
        saver = tf.train.Saver(max_to_keep=10)
        if FLAGS.load_model:
            saver.restore(sess, FLAGS.load_model)
            print ('Loaded model and saved at global step', tr_model.global_step.eval())
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(tr_model.char_embedding_padded)
            print ('Training model with character embedding padded')
        # --------------------------- make training --------------------------- #
        summary_writer = tf.summary.FileWriter('./train_dir', graph=sess.graph)
        sess.run(tf.assign(tr_model.learning_rate, FLAGS.learning_rate))

        print ("----- Training Start -----")

        tr_lstm_state = sess.run(tr_model.lstm_initial_state)
        best_va_loss = None

        for epoch in range(FLAGS.epoch):
            epoch_start = time.time()
            tr_count = 0
            va_count = 0
            train_loss = 0.0
            valid_loss = 0.0

            for x, y in iter_(zip_dict, 'train'):
                input_ = {tr_model.input : x, tr_model.targets:y, tr_model.lstm_initial_state:tr_lstm_state}
                loss_, lstm_state_, global_step_, train_op_, char_embedded_pad_ = sess.run([tr_model.loss,
                                                                                          tr_model.lstm_end_state,
                                                                                          tr_model.global_step,
                                                                                          tr_model.train_op,
                                                                                          tr_model.char_embedding_padded], input_)


                train_loss += (loss_ - train_loss) / FLAGS.batch_size
                tr_count += 1

                if tr_count % FLAGS.verbose == 0:
                    print ('epoch : %d, step : %d / %d, perplexity : %.3f, current_loss : %.3f, average_loss : %.3f,  times : %.3f' % \
                           (epoch, global_step_, len(zip_dict['train']), np.exp(loss_), loss_, train_loss, time.time() - epoch_start))

            print ('epoch : %d, time : %.3f' % (epoch, time.time() - epoch_start))

            # --------------------------- evaluate --------------------------- #
            va_lstm_state = sess.run(va_model.lstm_initial_state)
            for x, y in iter_(zip_dict, 'valid'):
                input_ = {va_model.input : x, va_model.targets:y, va_model.lstm_initial_state:va_lstm_state}
                loss_, lstm_state_ = sess.run([va_model.loss, va_model.lstm_end_state], input_)

                va_count += 1
                valid_loss += loss / len(zip_dict['valid'])

                if va_count % FLAGS.verbose == 0:
                    print (" perplexity : %.3f, validation loss : %.3f, average_loss : %.3f " % (np.exp(loss_), loss_, valid_loss))

            saver.save(sess, '%s/epoch%d_%.3f.model' % ('./train_dir', epoch, valid_loss))
            print ('Successfully saved model')

            # --------------------------- decay learning rate --------------------------- #
            if not best_va_loss and np.exp(best_va_loss) - np.exp(valid_loss) > 1.0:
                print ('Needs learning rate decay, perplexity does not decrease by more than 1.0')
                cur_lr = sess.run(tr_model.learning_rate)
                print ('Current learning rate : ', cur_lr)
                cur_lr *= FLAGS.lr_decay

                if cur_lr < 1e-6:
                    break

                sess.run(tr_model.learning_rate.assign(cur_lr))
                print ('Halved learning rate : ', cur_lr)
            else:
                best_va_loss = valid_loss

            # --------------------------- Summary Write --------------------------- #
            summary = tf.Summary(value=[tf.Summary.Value(tag="Training_loss", simple_value=train_loss),
                                        tf.Summary.Value(tag='Validation_loss', simple_value=valid_loss)])

            summary_writer.add_summary(summary, global_step_)

if __name__ == "__main__":
    tf.app.run()
