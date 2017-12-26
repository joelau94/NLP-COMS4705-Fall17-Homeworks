import os,sys
import random
import numpy as np
import theano
import theano.tensor as TT

from data import *
from decoder import *
from nn_utils import *


class Module(object):

    def __init__(self, name=None):
        self.name = name
        self.params = []


class Linear(Module):

    def __init__(self, input_dim, output_dim, name, use_bias=True, init_positive=False):
        super(Linear, self).__init__()
        self.name = name
        self.use_bias = use_bias
        self.W = init_weight((input_dim,output_dim), name=self.name+'_W', positive=init_positive)
        self.params += [self.W]
        if self.use_bias:
            self.b = init_bias(output_dim, name=name+'_b', positive=init_positive)
            self.params += [self.b]

    def __call__(self, _input):
        output = TT.dot(_input,self.W)
        if self.use_bias:
            output += self.b
        return output


class EmbeddingLookup(Module):

    def __init__(self, vocab_size, embedding_dim, name, use_bias=True, init_positive=False):
        super(EmbeddingLookup, self).__init__()
        self.name = name
        self.W_emb = init_weight((vocab_size, embedding_dim),name+'_W_emb', positive=init_positive)
        self.use_bias = use_bias
        self.params += [self.W_emb]
        if use_bias:
            self.b = init_bias(embedding_dim,name+'_b', positive=init_positive)
            self.params += [self.b]

    def __call__(self, index):
        if self.use_bias:
            return self.W_emb[index]+self.b
        else:
            return self.W_emb[index]


class Model(object):

    def __init__(self, name=None):
        super(Model, self).__init__()
        self.name = name
        self.params = []

    def save(self, path):
        values = {}
        for p in self.params:
            values[p.name] = p.get_value()
        np.savez(path, **values)

    def load(self, path):
        if not os.path.exists(path):
            return
        try:
            values = np.load(path)
            for p in self.params:
                if p.name in values:
                    if values[p.name].shape != p.get_value().shape:
                        raise IncompatibleParameterShapeError(p.name,p.get_value().shape,values[p.name].shape)
                    else:
                        p.set_value(values[p.name])
                        # print("Loaded parameter {}, shape {} .\n".format( p.name,values[p.name].shape ))
                else:
                    raise UndefinedParameterError(p.name)
        except UndefinedParameterError, e:
            print e.msg
            sys.exit(1)
        except IncompatibleParameterShapeError, e:
            print e.msg
            sys.exit(1)


class DepModel(Model):
    """docstring for DepModel"""
    def __init__(self, word_num, pos_num, dep_num, act_num, word_emb_dim, pos_emb_dim, dep_emb_dim, hid_1_dim, hid_2_dim, transfer, l2_reg=False, name=''):
        super(DepModel, self).__init__(name)
        self.name = name

        # if you prefer to have your own index for actions, change this.
        self.actions = ['SHIFT', 'LEFT-ARC:rroot', 'LEFT-ARC:cc', 'LEFT-ARC:number', 'LEFT-ARC:ccomp', 'LEFT-ARC:possessive', 'LEFT-ARC:prt', 'LEFT-ARC:num', 'LEFT-ARC:nsubjpass', 'LEFT-ARC:csubj', 'LEFT-ARC:conj', 'LEFT-ARC:dobj', 'LEFT-ARC:nn', 'LEFT-ARC:neg', 'LEFT-ARC:discourse', 'LEFT-ARC:mark', 'LEFT-ARC:auxpass', 'LEFT-ARC:infmod', 'LEFT-ARC:mwe', 'LEFT-ARC:advcl', 'LEFT-ARC:aux', 'LEFT-ARC:prep', 'LEFT-ARC:parataxis', 'LEFT-ARC:nsubj', 'LEFT-ARC:<null>', 'LEFT-ARC:rcmod', 'LEFT-ARC:advmod', 'LEFT-ARC:punct', 'LEFT-ARC:quantmod', 'LEFT-ARC:tmod', 'LEFT-ARC:acomp', 'LEFT-ARC:pcomp', 'LEFT-ARC:poss', 'LEFT-ARC:npadvmod', 'LEFT-ARC:xcomp', 'LEFT-ARC:cop', 'LEFT-ARC:partmod', 'LEFT-ARC:dep', 'LEFT-ARC:appos', 'LEFT-ARC:det', 'LEFT-ARC:amod', 'LEFT-ARC:pobj', 'LEFT-ARC:iobj', 'LEFT-ARC:expl', 'LEFT-ARC:predet', 'LEFT-ARC:preconj', 'LEFT-ARC:root', 'RIGHT-ARC:rroot', 'RIGHT-ARC:cc', 'RIGHT-ARC:number', 'RIGHT-ARC:ccomp', 'RIGHT-ARC:possessive', 'RIGHT-ARC:prt', 'RIGHT-ARC:num', 'RIGHT-ARC:nsubjpass', 'RIGHT-ARC:csubj', 'RIGHT-ARC:conj', 'RIGHT-ARC:dobj', 'RIGHT-ARC:nn', 'RIGHT-ARC:neg', 'RIGHT-ARC:discourse', 'RIGHT-ARC:mark', 'RIGHT-ARC:auxpass', 'RIGHT-ARC:infmod', 'RIGHT-ARC:mwe', 'RIGHT-ARC:advcl', 'RIGHT-ARC:aux', 'RIGHT-ARC:prep', 'RIGHT-ARC:parataxis', 'RIGHT-ARC:nsubj', 'RIGHT-ARC:<null>', 'RIGHT-ARC:rcmod', 'RIGHT-ARC:advmod', 'RIGHT-ARC:punct', 'RIGHT-ARC:quantmod', 'RIGHT-ARC:tmod', 'RIGHT-ARC:acomp', 'RIGHT-ARC:pcomp', 'RIGHT-ARC:poss', 'RIGHT-ARC:npadvmod', 'RIGHT-ARC:xcomp', 'RIGHT-ARC:cop', 'RIGHT-ARC:partmod', 'RIGHT-ARC:dep', 'RIGHT-ARC:appos', 'RIGHT-ARC:det', 'RIGHT-ARC:amod', 'RIGHT-ARC:pobj', 'RIGHT-ARC:iobj', 'RIGHT-ARC:expl', 'RIGHT-ARC:predet', 'RIGHT-ARC:preconj', 'RIGHT-ARC:root']
        # write your code here for additional parameters.
        # feel free to add more arguments to the initializer.

        self.word_num = word_num
        self.pos_num = pos_num
        self.dep_num = dep_num
        self.act_num = act_num
        self.word_emb_dim = word_emb_dim
        self.pos_emb_dim = pos_emb_dim
        self.dep_emb_dim = dep_emb_dim
        self.hid_1_dim = hid_1_dim
        self.hid_2_dim = hid_2_dim

        self.input_dim = 20 * self.word_emb_dim + 20 * self.pos_emb_dim + 12 * self.dep_emb_dim

        self.transfer = transfer
        if self.transfer == 'relu':
            init_positive = True
        else:
            init_positive = False

        self.l2_reg = l2_reg

        # embedding layer
        self.word_embed = EmbeddingLookup(self.word_num, self.word_emb_dim, name=self.name+'_word_embed', init_positive=init_positive)
        self.params += self.word_embed.params
        self.pos_embed = EmbeddingLookup(self.pos_num, self.pos_emb_dim, name=self.name+'_pos_embed', init_positive=init_positive)
        self.params += self.pos_embed.params
        self.dep_embed = EmbeddingLookup(self.dep_num, self.dep_emb_dim, name=self.name+'_dep_embed', init_positive=init_positive)
        self.params += self.dep_embed.params

        # 1st, 2nd hidden, output layer
        self.hid_1 = Linear(self.input_dim, self.hid_1_dim, name=self.name+'_hid_1', init_positive=init_positive)
        self.params += self.hid_1.params
        self.hid_2 = Linear(self.hid_1_dim, self.hid_2_dim, name=self.name+'_hid_2', init_positive=init_positive)
        self.params += self.hid_2.params
        self.out = Linear(self.hid_2_dim, self.act_num, name=self.name+'_out')
        self.params += self.out.params


    def build_graph(self):
        # input node
        self.x = TT.matrix('x', dtype='int64') # (batch_size, 20+20+12)
        self.y = TT.vector('y', dtype='int64') # (batch_size, )
        self.inputs = [self.x, self.y]

        self.batch_size = self.x.shape[0]

        # embedding layer
        self.word_emb = self.word_embed(self.x[:,:20]).reshape((self.batch_size,-1))
        self.pos_emb = self.pos_embed(self.x[:,20:40]).reshape((self.batch_size,-1))
        self.dep_emb = self.dep_embed(self.x[:,40:]).reshape((self.batch_size,-1))
        self.embedding = TT.concatenate([self.word_emb, self.pos_emb, self.dep_emb], axis=1) #(batch_size, input_dim)

        # 1st, 2nd hidden and output
        if self.transfer == 'relu':
            self.h1 = TT.nnet.relu(self.hid_1(self.embedding)) #(batch_size, hid_1_dim)
            self.h2 = TT.nnet.relu(self.hid_2(self.h1)) #(batch_size, hid_2_dim)
        elif self.transfer == 'sigmoid':
            self.h1 = TT.nnet.sigmoid(self.hid_1(self.embedding))
            self.h2 = TT.nnet.sigmoid(self.hid_2(self.h1))
        elif self.transfer == 'tanh':
            self.h1 = TT.tanh(self.hid_1(self.embedding))
            self.h2 = TT.tanh(self.hid_2(self.h1))
        elif self.transfer == 'cubic':
            self.h1 = self.hid_1(self.embedding) ** 3
            self.h2 = self.hid_2(self.h1) ** 3
        self.output = TT.nnet.softmax(self.out(self.h2)) #(batch_size, act_num)

        # cost
        y_idx = TT.arange(self.y.flatten().shape[0]) * self.act_num + self.y.flatten()
        costs = self.output.flatten()[y_idx]        
        costs = TT.mean(-TT.log(costs))
        if self.l2_reg: # l2 penalty
            for p in self.params:
            	costs += 1e-8 * (p**2).sum()
        self.costs = costs

        # scoring function
        self.test = theano.function(inputs=self.inputs, outputs=[self.output], on_unused_input='ignore')

    def get_vocabs(self, word, pos, label):
        read_vocab = lambda f: { entry[0]: int(entry[1])
            for entry in map(lambda l: l.split(), open(f,'r').readlines()) }
        self.word2id = defaultdict(lambda: 0)
        self.word2id.update(read_vocab(word))
        self.pos2id = defaultdict(lambda: 30) # default <null>
        self.pos2id.update(read_vocab(pos))
        self.label2id = read_vocab(label)

    def score(self, str_features):
        '''
        :param str_features: String features
        20 first: words, next 20: pos, next 12: dependency labels.
        DO NOT ADD ANY ARGUMENTS TO THIS FUNCTION.
        :return: list of scores
        '''
        # type of str_features: list of string (Zhuoran Liu, Dec 7 2017)
        # change this part of the code.
        _input = map(lambda w: self.word2id[w], str_features[:20]) + \
                map(lambda p: self.pos2id[p], str_features[20:40]) + \
                map(lambda l: self.label2id[l], str_features[40:])
        x = np.reshape(np.array(_input), (1,-1))
        dummy_y = np.array([1])

        scores = self.test(x, dummy_y)[0].flatten().tolist()
        return scores

        # return [0]*len(self.actions)


# if __name__=='__main__':
#     m = DepModel()
#     input_p = os.path.abspath(sys.argv[1])
#     output_p = os.path.abspath(sys.argv[2])
#     Decoder(m.score, m.actions).parse(input_p, output_p)