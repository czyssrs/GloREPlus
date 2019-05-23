# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
Parameter list
'''

class Hyperparams:

    # train and valid data
    train = "../data/train.txt"
    valid = "../data/valid.txt"
    word2vec_file = "../data/glove.840B.300d.zip"
    vocab_file = "../data/vocab.txt"

    # difine your own model name for training. 
    # or choose a pre-trained model for testing.
    model_dir = "../model/model-dim-512"

    # mode
    fresh_start = True
    model_select = "Transformer" # Transformer or RNN
    use_valid = True
    
    # training
    batch_size = 512 # batch size
    lr = 0.0001
    num_class = 1926 # num of relations 1925 + 1 (None)
    steps_checkpoint = 100
    vocab_size = 100000 # only keep most frequent words
    num_epochs = 500

    input_units = 300 # inputs token embedding size
    hidden_units = 300 # hidden units of transformer. defualt 512
    num_blocks = 6 # number of encoder/decoder blocks
    num_heads = 6 # attention heads defualt 8
    dropout_rate = 0.1
    sinusoid = False # If True, use sinusoid. If false, positional embedding.
    max_gradient_norm = 5.0 # gradient norm clipping
    

    # dimension of the textual relation embedding. 
    # remember to change here for different pre-trained models
    out_units = 512
    # max length of textual relation
    # set maxlen to 10 for dimension of 256 and 100
    maxlen = 20

    ### for test
    # your input textual relations
    test = "../data/test.txt"
    # the result textual relation embeddings of your input textual relations
    output_file = "../result/textual_embedding.txt"
    
    
    
    
