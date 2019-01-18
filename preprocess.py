from __future__ import print_function

# Warning Filter
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import collections
import tensorflow as tf
import numpy as np

from config import *

class Vocab:
    def __init__(self, token2idx=None, idx2token=None):
        self.token2idx = token2idx or dict()
        self.idx2token = idx2token or dict()

    def new_token(self, token):
        # token : word or character
        # return : index of token
        if token not in self.token2idx:
            index = len(self.token2idx)
            self.token2idx[token] = index
            self.idx2token[index] = token
        return self.token2idx[token]

    def get_token(self, index):
        # index : position number of token
        # return : word or character of index
        return self.idx2token[index]

    def get_index(self, token):
        # token : word or character
        # return : index of token
        return self.token2idx[token]


def make_data():
    # --------------------------- Output --------------------------- #
    # word_vocab  : Class of Bag of Words, which has token to index dictionary and reversed dictionary
    # char_vocab  : Class of Bag of Characters, which has token to index dictionary and reversed dictionary
    # whole_word  : train, valid, test set of word index table
    # whole_char  : train, valid, test set of character index table per word
    # word_maxlen : maximum of total words length

    char_vocab = Vocab()
    word_vocab = Vocab()

    EOS = '+'
    word_vocab.new_token('<unk>')
    word_vocab.new_token(EOS)

    char_vocab.new_token('{') # start of the word at index 0
    char_vocab.new_token('}') # end of the word at index 1
    char_vocab.new_token('+') # end of the sentence at index 2

    whole_char = collections.defaultdict(dict)
    whole_word = collections.defaultdict(dict)

    word_maxlen = 0

    for file in FLAGS.data_file:

        whole_char[file] = list() # whole data
        whole_word[file] = list()

        with open(FLAGS.data_path + file + '.txt', 'r', encoding='utf-8') as f:
            line = f.readlines()

            for one_line in line:

                word_list = list()
                char_list = list()

                for word in one_line.split():
                    # word token into dictionary
                    whole_word[file].append(word_vocab.new_token(word))

                    # character token into dictionary
                    with_char = '{' + word + '}'
                    whole_char[file].append([char_vocab.new_token(c) for c in with_char])

                    # check word maxlen
                    if len(word) > word_maxlen:
                        word_maxlen = len(word)

                # end of sentence, use '+' at eos
                whole_word[file].append(word_vocab.get_index(EOS))
                with_char = '{' + EOS + '}'
                whole_char[file].append([char_vocab.get_index(c) for c in with_char])

    word_maxlen += 2
    print ('Word Max Length : ', word_maxlen) # {multibillion-dollar}
    print ('Vocabulary set size : ', len(word_vocab.token2idx))
    print ('Character set size : ', len(char_vocab.token2idx))
    print ('Token size : %d\n' % (len(whole_word['train'])))

    return word_vocab, char_vocab, whole_word, whole_char, word_maxlen

def embedding_matrix(word_data, char_data, word_maxlen):
    # --------------------------- Input --------------------------- #
    # word_data : train, valid, test data word index
    # char_data : train, valid, test data character index per word
    # word_maxlen : maximum of total words length

    # --------------------------- Output --------------------------- #
    # word_matrix : be used to lookup word embedding matrix
    # char_matrix : be used to lookup character embedding matrix
    char_matrix = dict()
    word_matrix = dict()

    for file in FLAGS.data_file:
        # generate embedding matrix
        word_matrix[file] = np.array(word_data[file], dtype=np.float32)
        char_matrix[file] = np.zeros([len(char_data[file]), word_maxlen], dtype=np.int32)

        for idx, char_list in enumerate(char_data[file]):
            char_matrix[file][idx, :len(char_list)] = char_list

    print ('Shape of Train word matrix : ', word_matrix['train'].shape)
    print ('Shape of Valid word matrix : ', word_matrix['valid'].shape)
    print ('Shape of Test word matrix  : ', word_matrix['test'].shape)

    print ('Shape of Train Character matrix : ', char_matrix['train'].shape)
    print ('Shape of Valid Character matrix : ', char_matrix['valid'].shape)
    print ('Shape of Test Character matrix  : ', char_matrix['test'].shape)
    print ()
    return word_matrix, char_matrix

def batch_loader(word_matrix, char_matrix, word_maxlen):
    # --------------------------- Input --------------------------- #
    # word_matrix : be used to lookup word embedding matrix
    # char_matrix : be used to lookup character embedding matrix
    # word_maxlen : maximum of total words length

    # --------------------------- Output --------------------------- #
    # word_matrix : be used to lookup word embedding matrix (reduced length)
    # char_matrix : be used to lookup character embedding matrix (reduced length)
    # label_data : train, valid, test target data

    label_data = dict()

    for file in FLAGS.data_file:

        word_length   = len(word_matrix[file])
        total_size    = FLAGS.batch_size * FLAGS.trunc_step
        reduced_length = word_length // total_size * total_size

        # reduce the original word, character, labeled data
        word_matrix[file] = word_matrix[file][:reduced_length]
        char_matrix[file] = char_matrix[file][:reduced_length]

        label_data[file] = word_matrix[file].copy()
        label_data[file] = label_data[file].astype(np.int32)
        label_data[file][:-1] = label_data[file][1:]
        label_data[file][-1] = word_matrix[file][0]

        assert len(label_data[file]) == len(word_matrix[file])
        print ("----- check label answer of %s -----" % (file))
        print (label_data[file])
        print (word_matrix[file])

        print ("----- check data length of %s -----" % (file))
        print ((word_matrix[file].shape), (label_data[file].shape), (char_matrix[file].shape))

        char_matrix[file] = np.reshape(char_matrix[file], newshape=(FLAGS.batch_size, -1, FLAGS.trunc_step, word_maxlen))
        char_matrix[file] = np.transpose(char_matrix[file], axes=(1,0,2,3))

        label_data[file]  = np.reshape(label_data[file], newshape=(FLAGS.batch_size, -1, FLAGS.trunc_step))
        label_data[file]  = np.transpose(label_data[file], axes=(1,0,2))

        print ("----- check convolutional shape of %s -----" % (file))
        print (char_matrix[file].shape, label_data[file].shape)
        print ()

    return word_matrix, char_matrix, label_data

def zip_file(char_matrix, label_data):
    # --------------------------- Input --------------------------- #
    # char_matrix : be used to lookup character embedding matrix
    # label_data : train, valid, test target data

    # --------------------------- Output --------------------------- #
    # zip_dict : zip of character and label list
    zip_dict = dict()
    total_zip_list = list()
    idx = 0

    for file in FLAGS.data_file:
        char_list  = list(char_matrix[file])
        label_list = list(label_data[file])
        zip_dict[file] = list(zip(char_list, label_list))
        if idx < 2:
            total_zip_list.extend(list(zip(char_list, label_list)))
        idx += 1

    return char_list, label_list, zip_dict, total_zip_list

def iter_(zip_dict, file_name):
    # for training with train file
    for x, y in zip_dict[file_name]:
        yield x, y

def total_iter_(total_zip_list):
    # for training with train + valid file
    for x, y in total_zip_list:
        yield x, y

def preprocessing(FLAGS):
    word_vocab, char_vocab, whole_word, whole_char, word_maxlen = make_data()
    word_matrix, char_matrix = embedding_matrix(whole_word, whole_char, word_maxlen)
    word_matrix, char_matrix, label_data = batch_loader(word_matrix, char_matrix, word_maxlen)
    char_list, label_list, zip_dict, total_zip_list = zip_file(char_matrix, label_data)

    return word_vocab, char_vocab, word_matrix, char_matrix, label_data, \
            word_maxlen, char_list, label_list, zip_dict, total_zip_list
