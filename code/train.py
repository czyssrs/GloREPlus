# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
Train textual relation embedding
'''

from __future__ import print_function
import tensorflow as tf
import os, sys, codecs
import random
import math
import time

from hyperparams import Hyperparams as hp
import data_utils
import model
import baseline



if __name__ == '__main__':

    # hp.model_dir = sys.argv[1]
    print ("Model directory: " + hp.model_dir)
    print ("Embedding dimension: " + hp.out_units)

    # data processing
    vocab = data_utils.load_vocab()
    assert len(vocab) == hp.vocab_size + 2
    init_embedding = data_utils.create_init_embedding(vocab, hp.word2vec_file, hp.input_units)

    textual_ind_train, data_map_train = data_utils.data_processor(hp.train)
    num_batch = len(textual_ind_train) / hp.batch_size
    print ("Num of train data: %d" % len(textual_ind_train))

    if hp.use_valid:
        textual_ind_valid, data_map_valid = data_utils.data_processor(hp.valid)
        num_batch_valid = len(textual_ind_valid) / hp.batch_size
        len_last_batch_valid = len(textual_ind_valid) % hp.batch_size
        print ("Num of valid data: %d" % len(textual_ind_valid))


    # construct model
    # Transformer
    if hp.model_select == "Transformer":
        g = model.Transformer(vocab_size=len(vocab),
                                    initial_embeddings=init_embedding)
        print("Transformer constructed")


    # RNN
    else:
        g = baseline.vanillaRNN(vocab_size=len(vocab),
                            initial_embeddings=init_embedding)
        print("vanilla RNN constructed")

    init_embedding = None

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config, graph=g.graph) as sess:

        if not hp.fresh_start:
            # model reload
            ckpt = tf.train.get_checkpoint_state(hp.model_dir)
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            g.saver.restore(sess, ckpt.model_checkpoint_path)

        else:
            sess.run(tf.global_variables_initializer())

        last_ppx = float("inf")
        last_valid_ppx = float("inf")
        step_time = 0.0

        for epoch in range(1, hp.num_epochs+1):

            random.shuffle(textual_ind_train)
            loss = 0.0
            gradient_norm = 0.0

            for step in xrange(0, num_batch):
                start_time = time.time()
                this_batch_textual_ind = textual_ind_train[hp.batch_size * step : hp.batch_size * (step + 1)]
                this_ind, this_distribution = data_utils.data_padding(this_batch_textual_ind, data_map_train, vocab)

                step_mean_loss, step_gradient_norm, _ = g.step(sess,
                                                                this_ind,
                                                                this_distribution,
                                                                mode="train")
                loss += step_mean_loss
                gradient_norm += step_gradient_norm
                step_time += ((time.time() - start_time) / hp.steps_checkpoint)

                if step % hp.steps_checkpoint == 0:
                    record_ppx = math.exp(step_mean_loss) if step_mean_loss < 300 else float('inf')
                    print ("On epoch %d step %d, Loss : %.4f, Gradient norm : %.4f, step time : %.4f" % 
                        (epoch, step, record_ppx, step_gradient_norm, step_time))
                    step_time = 0.0


            loss /= num_batch
            gradient_norm /= num_batch
            train_ppx = math.exp(loss) if loss < 300 else float('inf')
            print ("On epoch %d, Loss : %.4f, Gradient norm : %.4f" %
                (epoch, train_ppx, gradient_norm))

            if hp.use_valid:
                mean_loss_valid = 0.0
                for step_valid in xrange(0, num_batch_valid + 1):
                    if step_valid == num_batch_valid and len_last_batch_valid!= 0:
                        this_batch_textual_ind = textual_ind_valid[-len_last_batch_valid : ]
                    else:
                        this_batch_textual_ind = textual_ind_valid[hp.batch_size * step_valid : hp.batch_size * (step_valid + 1)]

                    this_ind, this_distribution = data_utils.data_padding(this_batch_textual_ind, data_map_valid, vocab)

                    _, _, step_mean_loss_valid = g.step(sess,
                                                        this_ind,
                                                        this_distribution,
                                                        mode="valid")

                    mean_loss_valid += step_mean_loss_valid
                    
                if len_last_batch_valid == 0:
                    mean_loss_valid /= num_batch_valid
                else:
                    mean_loss_valid /= (num_batch_valid + 1)

                valid_ppx = math.exp(mean_loss_valid) if mean_loss_valid < 300 else float('inf')
                print ("On epoch %d, Valid loss : %.4f" %
                    (epoch, valid_ppx))

                if valid_ppx < last_valid_ppx:
                    checkpoint_path = os.path.join(hp.model_dir, "model_best.ckpt")
                    g.saver.save(sess, checkpoint_path, global_step=g.global_step.eval())
                    print ("Model saved on epoch %d global step %d." % (epoch, g.global_step.eval()))
                    last_valid_ppx = valid_ppx

            else:

                if train_ppx < last_ppx:
                    checkpoint_path = os.path.join(hp.model_dir, "model_best.ckpt")
                    g.saver.save(sess, checkpoint_path, global_step=g.global_step.eval())
                    print ("Model saved on epoch %d global step %d." % (epoch, g.global_step.eval()))
                    last_ppx = train_ppx