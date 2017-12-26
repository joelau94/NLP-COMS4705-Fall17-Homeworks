import os, sys
import json
import datetime

import numpy as np

from depModel import *
from decoder import *
from optim import *
from data import *

class Config(object):
	"""docstring for Config"""
	def __init__(self, name='default'):
		super(Config, self).__init__()
		self.name = name

		self.config = {
			'data_train': 'data/train.data',
			'word_path': 'data/vocabs.word',
			'pos_path': 'data/vocabs.pos',
			'labels_path': 'data/vocabs.labels',
			'actions_path': 'data/vocabs.actions',
			'trainer': 'Adam',
			'adam_alpha': 0.005,
            'l2_reg': False,
			'epochs': 7,
			'batch_size': 1000,
			'transfer': 'relu',
			'word_num': 4807,
			'pos_num': 45,
			'dep_num': 46,
			'act_num': 93,
			'word_emb_dim': 64,
			'pos_emb_dim': 32,
			'dep_emb_dim': 32,
			'hid_1_dim': 200,
			'hid_2_dim': 200
		}

	def save(self):
		with open('configs/'+self.name+'.json','w+') as f:
			json.dump(self.config, f)

	def load(self, cfg_name):
		with open('configs/'+cfg_name+'.json','r') as f:
			self.config = json.load(f)
		

class Parser(object):
    """docstring for Parser"""
    def __init__(self, cfg=Config().config):
        super(Parser, self).__init__()
        self.config = cfg.config
        self.name = cfg.name
    
    def train(self):
    	print('Loading data ... ({})'.format(datetime.datetime.now()))
    	data = Data(self.config['word_path'],self.config['pos_path'],self.config['labels_path'],self.config['actions_path'])
    	data.load_data(self.config['data_train'])

    	model = DepModel(self.config['word_num'], self.config['pos_num'], self.config['dep_num'], self.config['act_num'],
    					self.config['word_emb_dim'], self.config['pos_emb_dim'], self.config['dep_emb_dim'],
    					self.config['hid_1_dim'], self.config['hid_2_dim'], self.config['transfer'], self.config['l2_reg'],
                        name=self.name)
    	model.build_graph()

    	print('Compiling graph ... ({})'.format(datetime.datetime.now()))
    	print('Using optimizer {}'.format(self.config['trainer']))
    	if self.config['trainer'] == 'Adam':
    		trainer = Adam(model.inputs, model.costs, model.params, alpha=self.config['adam_alpha'])
    	elif self.config['trainer'] == 'AdaDelta':
    		trainer = AdaDelta(model.inputs, model.costs, model.params)
    	elif self.config['trainer'] == 'SGD':
    		trainer = SGD(model.inputs, model.costs, model.params)

    	print('Training begins ({})'.format(datetime.datetime.now()))
    	print('data_size = {}'.format(data.data_size))
    	for i in range(data.data_size * self.config['epochs'] / self.config['batch_size']):
    		X, Y = data.next(self.config['batch_size'])
    		costs, grads_norm = trainer.update_grads(X, Y)
    		if i % 10 == 0:
    			print('Training {}, costs = {} ({})'.format(i, costs, datetime.datetime.now()))
    		trainer.update_params()

    	model.save('models/'+str(self.name)+'.npz')


    def test(self, test_in, test_out):
    	model = DepModel(self.config['word_num'], self.config['pos_num'], self.config['dep_num'], self.config['act_num'],
    					self.config['word_emb_dim'], self.config['pos_emb_dim'], self.config['dep_emb_dim'],
    					self.config['hid_1_dim'], self.config['hid_2_dim'], self.config['transfer'], name=self.name)
    	model.build_graph()
    	model.load('models/'+str(self.name)+'.npz')

    	model.get_vocabs(self.config['word_path'],self.config['pos_path'],self.config['labels_path'])

    	Decoder(model.score, model.actions).parse(test_in, test_out)