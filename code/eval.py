# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
Generate textual relation embedding
'''

from __future__ import print_function
import tensorflow as tf
import os, sys, codecs
import random
import math

from hyperparams import Hyperparams as hp
import data_utils as data_utils
import model
import baseline



if __name__ == '__main__':

    # hp.model_dir = sys.argv[1]
    print ("Model directory: " + hp.model_dir)
    out_file = open(hp.output_file, "w")

    # data processing
    vocab = data_utils.load_vocab()
    assert len(vocab) == hp.vocab_size + 2

    textual_ind_test, data_map_test = data_utils.data_processor_test(hp.test)
    num_batch_test = len(textual_ind_test) / hp.batch_size
    len_last_batch_test = len(textual_ind_test) % hp.batch_size
    print ("Num of test data: %d" % len(textual_ind_test))


    # construct model
    # Transformer
    if hp.model_select == "Transformer":
        g = model.Transformer(vocab_size=len(vocab),
                                    initial_embeddings=None)
        print("Transformer constructed")

    # RNN
    else:
        g = baseline.vanillaRNN(vocab_size=len(vocab),
                            initial_embeddings=None)
        print("vanilla RNN constructed")


    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config, graph=g.graph) as sess:

        # model reload
        ckpt = tf.train.get_checkpoint_state(hp.model_dir)
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        g.saver.restore(sess, ckpt.model_checkpoint_path)

        for step_test in xrange(0, num_batch_test + 1):
            if step_test == num_batch_test and len_last_batch_test!= 0:
                this_batch_textual_ind = textual_ind_test[-len_last_batch_test : ]
            else:
                this_batch_textual_ind = textual_ind_test[hp.batch_size * step_test : hp.batch_size * (step_test + 1)]

            this_ind, this_distribution = data_utils.data_padding(this_batch_textual_ind, data_map_test, vocab)

            _, _, res_emb = g.step(sess,
                                  this_ind,
                                  this_distribution,
                                  mode="test")

            result_embedding = [row.tolist() for row in res_emb]

            # write into file
            for (textual_string, res) in zip(this_batch_textual_ind, result_embedding):
                out_file.write(textual_string)
                for s in res:
                    out_file.write(" " + str(s))
                out_file.write("\n")

    out_file.close()

