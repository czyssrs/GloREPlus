# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
Vanilla RNN model
'''

from __future__ import print_function
import tensorflow as tf
from hyperparams import Hyperparams as hp
import os, codecs
import math
from tqdm import tqdm


class vanillaRNN():
    def __init__(self,
                 vocab_size,
                 initial_embeddings=None,
                 ):
        self.graph = tf.Graph()
        self.vocab_size = vocab_size
        with self.graph.as_default():

            self.y = tf.placeholder(tf.float32, shape=(None, hp.num_class), name="label_distribution")
            self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen), name="textual_relations")
            self.seq_len = tf.reduce_mean(tf.ones_like(self.x, dtype=tf.int32), axis=-1) * hp.maxlen

            self.x_rev = tf.reverse(self.x, axis=tf.convert_to_tensor([1]))
            

            # word embedding
            if initial_embeddings is not None:
                self.init_emb = tf.cast(initial_embeddings, tf.float32)
                print ("Initialize with given embeddings")
                self.initial_embedding = tf.get_variable("init_embedding",
                                                         initializer=self.init_emb,
                                                         dtype=tf.float32,
                                                         trainable=True)


            else:
                self.init_emb = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3), dtype=tf.float32)
                self.initial_embedding = tf.get_variable("init_embedding",
                                                         [self.vocab_size,
                                                          hp.input_units],
                                                         initializer=self.init_emb,
                                                         dtype=tf.float32,
                                                         trainable=True)

            self.embedded_inputs = tf.nn.embedding_lookup(self.initial_embedding, self.x_rev)
            
            cell = tf.nn.rnn_cell.GRUCell(hp.out_units)

            outputs, states = tf.nn.dynamic_rnn(cell,
                                                self.embedded_inputs,
                                                sequence_length=self.seq_len,
                                                dtype=tf.float32,
                                                parallel_iterations=32)

            
            self.out_layer = tf.tanh(tf.layers.dense(states, hp.out_units))
            self.logits = tf.layers.dense(self.out_layer, hp.num_class)
               
            # Loss
            self.mean_loss = tf.reduce_mean(
                              tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
           
            # Training Scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            opt = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)

            # Gradient clipping
            params = tf.trainable_variables()
            gradients = tf.gradients(self.mean_loss, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, hp.max_gradient_norm)
            self.gradient_norm = norm
            self.update = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
               
            # Summary 
            tf.summary.scalar('mean_loss', self.mean_loss)
            self.merged = tf.summary.merge_all()

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)


    def step(self,
             session,
             this_textual_rel,
             this_distribution,
             mode):

        input_feed = {}
        # feed for inputs
        input_feed[self.x.name] = this_textual_rel
        input_feed[self.y.name] = this_distribution

        # train
        if mode == "train":
            # training
            output_feed = [self.update,
                           self.mean_loss,
                           self.gradient_norm]
            outputs = session.run(output_feed, input_feed)
            # Gradient norm, loss, no outputs
            return outputs[1], outputs[2], None
        # valid
        elif mode == "valid":
            output_feed = [self.mean_loss]
            outputs = session.run(output_feed, input_feed)
            # Gradient norm, loss, no outputs
            return None, None, outputs[0]

        # test
        else:
            output_feed = [self.out_layer]
            outputs = session.run(output_feed, input_feed)
            return None, None, outputs[0]