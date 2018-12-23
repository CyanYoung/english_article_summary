import json
import pickle as pk

import numpy as np

from gensim.corpora import Dictionary


embed_len = 200
min_freq = 3
max_vocab = 10000
seq_len1, seq_len2 = 200, 50

bos, eos = '*', '#'

pad_ind, oov_ind = 0, 1

path_word_vec = 'feat/word_vec.pkl'
path_word_ind = 'feat/word_ind.pkl'
path_embed = 'feat/embed.pkl'


def load(path):
    with open(path, 'rb') as f:
        item = pk.load(f)
    return item


def save(item, path):
    with open(path, 'wb') as f:
        pk.dump(item, f)


def add_flag(texts, bos, eos):
    flag_texts = list()
    for text in texts:
        flag_texts.append(' '.join([bos, text, eos]))
    return flag_texts


def shift(flag_text_words):
    sents = [words[:-1] for words in flag_text_words]
    labels = [words[1:] for words in flag_text_words]
    return sents, labels


def tran_dict(word_inds, off):
    off_word_inds = dict()
    for word, ind in word_inds.items():
        off_word_inds[word] = ind + off
    return off_word_inds


def tokenize(sent_words, path_word_ind):
    model = Dictionary(sent_words)
    model.filter_extremes(no_below=min_freq, no_above=1.0, keep_n=max_vocab)
    word_inds = model.token2id
    word_inds = tran_dict(word_inds, off=2)
    save(word_inds, path_word_ind)


def embed(path_word_ind, path_word_vec, path_embed):
    word_inds = load(path_word_ind)
    word_vecs = load(path_word_vec)
    vocab = word_vecs.keys()
    vocab_num = min(max_vocab + 2, len(word_inds) + 2)
    embed_mat = np.zeros((vocab_num, embed_len))
    for word, ind in word_inds.items():
        if word in vocab:
            if ind < max_vocab:
                embed_mat[ind] = word_vecs[word]
    save(embed_mat, path_embed)


def sent2ind(words, word_inds, seq_len, loc, keep_oov):
    seq = list()
    for word in words:
        if word in word_inds:
            seq.append(word_inds[word])
        elif keep_oov:
            seq.append(oov_ind)
    return pad(seq, seq_len, loc)


def pad(seq, seq_len, loc):
    if loc == 'post':
        if len(seq) < seq_len:
            return seq + [pad_ind] * (seq_len - len(seq))
        else:
            return seq[:seq_len]
    else:
        if len(seq) < seq_len:
            return [pad_ind] * (seq_len - len(seq)) + seq
        else:
            return seq[-seq_len:]


def align(sent_words, seq_len, path_sent, loc):
    word_inds = load(path_word_ind)
    pad_seqs = list()
    for words in sent_words:
        pad_seq = sent2ind(words, word_inds, seq_len, loc, keep_oov=True)
        pad_seqs.append(pad_seq)
    pad_seqs = np.array(pad_seqs)
    save(pad_seqs, path_sent)


def vectorize(paths, mode):
    with open(paths['data'], 'r') as f:
        pairs = json.load(f)
    text1s, text2s = zip(*pairs)
    text1s, text2s = list(text1s), list(text2s)
    sent1s = add_flag(text1s, bos='', eos=eos)
    sent1_words = [sent.split() for sent in sent1s]
    flag_text2s = add_flag(text2s, bos=bos, eos=eos)
    flag_text2_words = [text.split() for text in flag_text2s]
    if mode == 'train':
        tokenize(sent1_words + flag_text2_words, path_word_ind)
        embed(path_word_ind, path_word_vec, path_embed)
    if mode == 'test':
        save(text1s, paths['sent1'])
        save(text2s, paths['label'])
    else:
        sent2_words, label_words = shift(flag_text2_words)
        align(sent1_words, seq_len1, paths['sent1'], loc='pre')
        align(sent2_words, seq_len2, paths['sent2'], loc='post')
        align(label_words, seq_len2, paths['label'], loc='post')


if __name__ == '__main__':
    paths = dict()
    paths['data'] = 'data/train.json'
    paths['sent1'] = 'feat/sent1_train.pkl'
    paths['sent2'] = 'feat/sent2_train.pkl'
    paths['label'] = 'feat/label_train.pkl'
    vectorize(paths, 'train')
    paths['data'] = 'data/dev.json'
    paths['sent1'] = 'feat/sent1_dev.pkl'
    paths['sent2'] = 'feat/sent2_dev.pkl'
    paths['label'] = 'feat/label_dev.pkl'
    vectorize(paths, 'dev')
    paths['data'] = 'data/test.json'
    paths['sent1'] = 'feat/sent1_test.pkl'
    paths['label'] = 'feat/label_test.pkl'
    vectorize(paths, 'test')
