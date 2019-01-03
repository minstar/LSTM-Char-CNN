from __future__ import print_function

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

    print ('Word Max Length : ', word_maxlen) # {multibillion-dollar}
    print ('Vocabulary set size : ', len(word_vocab.token2idx))
    print ('Character set size : ', len(char_vocab.token2idx))
    print ('Token size : ', len(whole_word['train']))

    return word_vocab, char_vocab, whole_word, whole_char, word_maxlen

def embedding_matrix(word_data, char_data, max_word_len):
    # --------------------------- Input --------------------------- #
    # word_data : train, valid, test data word index
    # char_data : train, valid, test data character index per word
    # max_word_len : maximum of total words length

    # --------------------------- Output --------------------------- #
    # word_matrix : be used to lookup word embedding matrix
    # char_matrix : be used to lookup character embedding matrix
    char_matrix = dict()
    word_matrix = dict()

    for file in FLAGS.data_file:
        word_matrix[file] = np.array(word_data[file], dtype=np.float32)
        char_matrix[file] = np.zeros([len(char_data[file]), max_word_len+2], dtype=np.int32)

        for idx, char_list in enumerate(char_data[file]):
            char_matrix[file][idx, :len(char_list)] = char_list

    print ('Shape of Train word matrix : ', word_matrix['train'].shape)
    print ('Shape of Valid word matrix : ', word_matrix['valid'].shape)
    print ('Shape of Test word matrix  : ', word_matrix['test'].shape)

    print ('Shape of Train Character matrix : ', char_matrix['train'].shape)
    print ('Shape of Valid Character matrix : ', char_matrix['valid'].shape)
    print ('Shape of Test Character matrix  : ', char_matrix['test'].shape)

    return word_matrix, char_matrix

def preprocessing(FLAGS):
    word_vocab, char_vocab, whole_word, whole_char, word_maxlen = make_data()
    word_matrix, char_matrix = embedding_matrix(whole_word, whole_char, word_maxlen)

    return word_vocab, char_vocab, word_matrix, char_matrix
