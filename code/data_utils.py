import itertools
import numpy as np
import random
import os
import yaml
import gzip
import unicodedata
import gensim
import tensorflow as tf
import codecs
import operator
import zipfile
from gensim.models import KeyedVectors
from hyperparams import Hyperparams as hp


zip = getattr(itertools, 'izip', zip)
REL_TAB = "##"


def format_list(list):
    """format a list into a space-separated string"""
    return " ".join(str(tok) for tok in list)


def softmax(x):
    """Compute softmax values for array x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



def shuffle_data(ori_labels, ori_corpus):
    assert (len(ori_labels) == len(ori_corpus))
    index_shuf = list(range(len(ori_labels)))
    random.shuffle(index_shuf)
    labels = [ori_labels[i] for i in index_shuf]
    corpus = [ori_corpus[i] for i in index_shuf]
    return labels, corpus



def create_vocab(file_list):
    '''
    create vacabulary.
    get data statics.
    special : "<PAD>", "<_UNK>"
    '''
    max_len_text_rel = 0
    count = {}
    vocab = {}
    vocab["<_PAD>"] = 0
    vocab["<_UNK>"] = 1

    for file in file_list:
        with open(file, "r") as f:
            for line in f.readlines():
                text_rel = line.strip("\n").split("\t")[0]
                all_count = float(line.strip("\n").split("\t")[4])
                if max_len_text_rel < len(text_rel.split(REL_TAB)):
                    max_len_text_rel = len(text_rel.split(REL_TAB))
                for word in text_rel.split(REL_TAB):
                    if word != "":
                        if word not in count:
                            count[word] = int(all_count)
                        else:
                            count[word] += int(all_count)

    sorted_count = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
    sorted_count = sorted_count[0 : hp.vocab_size]
    
    for ind, (word, _) in enumerate(sorted_count):
        vocab[word] = ind + 2

    print ("Max text relation length: %d" % max_len_text_rel)
    print ("Num of vocab: %d" % len(vocab))

    return vocab, max_len_text_rel




def read_word2vec_zip(word2vec_file):
    wordvec_map = {}
    num_words = 0
    dimension = 0
    zfile = zipfile.ZipFile(word2vec_file)
    for finfo in zfile.infolist():
        ifile = zfile.open(finfo)
        for line in ifile:
            line = line.strip()
            #print line
            entries = line.split(' ')
            if len(entries) == 2:
                continue
            word = entries[0].strip()
            vec = map(float, entries[1:])

            if word in wordvec_map:
                print ("Invalid word in embedding. Does not matter.")
                continue
            assert dimension == 0 or dimension == len(vec)

            wordvec_map[word] = np.array(vec)
            num_words += 1
            dimension = len(vec)

    return wordvec_map, num_words, dimension



def read_word2vec(word2vec_file):
    wordvec_map = {}
    num_words = 0
    dimension = 0
    with open(word2vec_file, "r") as f:
        for line in f:
            line = line.strip()
            #print line
            entries = line.split(' ')
            if len(entries) == 2:
                continue
            word = entries[0].strip()
            vec = map(float, entries[1:])

            if word in wordvec_map:
                print ("Invalid word in embedding. Does not matter.")
                continue
            # assert word not in wordvec_map
            assert dimension == 0 or dimension == len(vec)

            wordvec_map[word] = np.array(vec)
            num_words += 1
            dimension = len(vec)

    return wordvec_map, num_words, dimension


def load_vocab():
    vocab = {}
    with open(hp.vocab_file, "r") as f:
        for line in f:
            if line != "":
                #print line
                vocab[line.strip().split("\t")[0]] = int(line.strip().split("\t")[1])
    return vocab



def create_init_embedding(vocab, word2vec_file, emblen):
    '''
    create initial embedding for text relation words.
    words not in word2vec file initialized to random.
    add <_PAD> : 0
    '''
    init_embedding = np.random.uniform(-np.sqrt(3), np.sqrt(3), size = (len(vocab), emblen))

    if word2vec_file.endswith('.gz'):
        word2vec_map = KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
    elif word2vec_file.endswith('.zip'):
        word2vec_map, num_words, dimension = read_word2vec_zip(word2vec_file)
    else:
        word2vec_map, num_words, dimension = read_word2vec(word2vec_file)

    num_covered = 0

    # embedding for special symbols
    init_embedding[0][:] = np.zeros(emblen)

    for word in vocab:
        if word in word2vec_map:
            vec = word2vec_map[word]
            if len(vec) != emblen:
                raise ValueError("word2vec dimension doesn't match.")
            init_embedding[vocab[word], :] = vec
            num_covered += 1

    print ("word2vec covered: %d" % num_covered)
    return init_embedding


def data_processor(file_name):

    textual_ind = []
    data_map = {}

    this_textual_rel = ""
    this_textual_rel_map = {}
    this_weight = 0.0

    data_file = open(file_name, "r")


    line = data_file.readline()
    while(line):
        textual_rel = line.strip("\n").split("\t")[0]
        rel_id = int(line.strip("\n").split("\t")[1])
        weight = float(line.strip("\n").split("\t")[2])

        if this_textual_rel != textual_rel and this_textual_rel != "":
            # for one textual relation bucks
            textual_ind.append(this_textual_rel)
            data_map[this_textual_rel] = []

            for each_rel_id in this_textual_rel_map:
                data_map[this_textual_rel].append((each_rel_id, this_textual_rel_map[each_rel_id]))

            this_textual_rel_map.clear()
            this_weight = 0.0

        this_textual_rel_map[rel_id] = weight
        this_textual_rel = textual_rel
        line = data_file.readline()

        if line == "":
            textual_ind.append(this_textual_rel)
            data_map[this_textual_rel] = []

            for each_rel_id in this_textual_rel_map:
                data_map[this_textual_rel].append((each_rel_id, this_textual_rel_map[each_rel_id]))

    data_file.close()
    return textual_ind, data_map


def data_processor_test(test_file):

    textual_ind = []
    data_map = {}

    with open(test_file, "r") as f:
        for line in f:
            textual_ind.append(line.strip())
            if line.strip() not in data_map:
                data_map[line.strip()] = []
            data_map[line.strip()].append((0, 0.0))

    return textual_ind, data_map



def negative_sampling(batch_data, textual_rel_mapping):

    negative_data = []
    for (text_rel_ind, rel, label_prob) in batch_data:
        for _ in xrange(0, hp.negative_sample):
            negative_sample = random.randint(1, hp.num_class)
            while(negative_sample in textual_rel_mapping[tuple(text_rel_ind)]):
                negative_sample = random.randint(1, hp.num_class)
            negative_data.append((text_rel_ind, negative_sample, 5e-4))

    out_batch = batch_data + negative_data
    random.shuffle(out_batch)
    return out_batch


def data_padding(this_batch_textual_ind, data_map, vocab):

    this_ind = []
    this_distribution = []
    word_in = 0
    word_out = 0

    for textual_rel in this_batch_textual_ind:
        text_rel_ind = []
        for word in textual_rel.strip().split(REL_TAB):
            if word in vocab:
                text_rel_ind.append(vocab[word])
                word_in += 1
            else:
                text_rel_ind.append(vocab["<_UNK>"])
                word_out += 1
        if len(text_rel_ind) <= hp.maxlen:
            for _ in xrange(hp.maxlen - len(text_rel_ind)):
                text_rel_ind.append(0)
        else:
            text_rel_ind = text_rel_ind[0 : hp.maxlen]
        this_ind.append(text_rel_ind)

    for textual_rel in this_batch_textual_ind:
        this_dist = np.zeros(hp.num_class, dtype=np.float32)
        for tup in data_map[textual_rel]:
            this_dist[tup[0]] = tup[1]
        this_distribution.append(this_dist.tolist())

    return this_ind, this_distribution


def data_padding_test(this_batch):

    textual_rel = []
    label = []
    label_weights = []
    for text_rel_ind in this_batch:
        if len(text_rel_ind) <= hp.maxlen:
            for _ in xrange(hp.maxlen - len(text_rel_ind)):
                text_rel_ind.append(0)

        else:
            text_rel_ind = text_rel_ind[0 : hp.maxlen]

        textual_rel.append(text_rel_ind)
        label.append(0)
        label_weights.append(1.0)

    return textual_rel, label, label_weights






























