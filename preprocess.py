import json

import nltk

from random import shuffle


max_num = int(1e5)


def save(path, pairs):
    with open(path, 'w') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=4)


def clean(text):
    words = nltk.word_tokenize(text)
    return ' '.join(words)


def prepare(path_univ, path_train, path_dev, path_test):
    pairs = list()
    with open(path_univ, 'r') as f:
        for count, line in enumerate(f):
            fields = line.strip().split('\t')
            if len(fields) != 2:
                continue
            text2, text1 = fields
            text2, text1 = clean(text2), clean(text1)
            pairs.append((text1.lower(), text2.lower()))
            if count > max_num:
                break
    shuffle(pairs)
    bound1 = int(len(pairs) * 0.8)
    bound2 = int(len(pairs) * 0.9)
    save(path_train, pairs[:bound1])
    save(path_dev, pairs[bound1:bound2])
    save(path_test, pairs[bound2:])


if __name__ == '__main__':
    path_univ = 'data/univ.txt'
    path_train = 'data/train.json'
    path_dev = 'data/dev.json'
    path_test = 'data/test.json'
    prepare(path_univ, path_train, path_dev, path_test)