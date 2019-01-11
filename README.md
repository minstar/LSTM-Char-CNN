# LSTM-Char-CNN
Re-implementation of Character-Aware Neural Language Models

Get some tensorflow code from https://github.com/mkroutikov/tf-lstm-char-cnn

Get some ambiguous metric from https://github.com/yoonkim/lstm-char-cnn

Version Type
1. python version : 3.6.0
2. tensorflow version : 1.9.0

Issues
1. selecting loss operation
2. selecting model bias term - Highway bias term has removed (doesn't exist in loaded model parameters)
3. zero padding confusing in scatter_update
