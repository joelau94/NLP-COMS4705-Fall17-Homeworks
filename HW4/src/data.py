import os,sys
import random
import numpy as np
from collections import defaultdict

class Data(object):

    def __init__(self, word, pos, label, action):
        super(Data, self).__init__()
        read_vocab = lambda f: { entry[0]: int(entry[1])
            for entry in map(lambda l: l.split(), open(f,'r').readlines()) }
        self.word = defaultdict(lambda: 0)
        self.word.update(read_vocab(word))
        self.pos = defaultdict(lambda: 30) # default <null>
        self.pos.update(read_vocab(pos))
        self.label = read_vocab(label)
        self.action = read_vocab(action)
        self.data = None
        self.data_size = 0
        self.cursor = 0

    def load_data(self, fname):
        feat = lambda d: map(
            lambda e: map(lambda w: self.word[w], e[:20]) + \
                map(lambda p: self.pos[p], e[20:40]) + \
                map(lambda l: self.label[l], e[40:52]) + \
                [ self.action[e[52]] ],
            d)
        self.data = feat( map(lambda l: l.split(), open(fname, 'r').readlines()) )
        self.data_size = len(self.data)

    def next(self, n=1):
        if self.cursor + n > len(self.data):
            self.cursor = 0
        if self.cursor == 0:
            random.shuffle(self.data)

        batch = np.array(self.data[self.cursor : self.cursor + n])
        X = batch[:,:-1]
        Y = batch[:,-1]

        self.cursor += n

        return X, Y

    def reset(self):
        self.cursor = 0