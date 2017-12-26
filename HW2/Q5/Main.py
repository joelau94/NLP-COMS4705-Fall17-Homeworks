#!/usr/bin/python

'''
COMS 4705 NLP - Fall 2017: HW2

Prerequisite: parse_train.RARE.dat generated by Q4
'''

import os
import json
from collections import defaultdict
from collections import Counter
from math import log
import pdb

'''
Rule: ('lhs', 'rhs1'[, 'rhs2'])
BackPointer: (Rule, split)
'''

def preprocess(train_rare_file='parse_train.RARE.dat', count_file='cfg.rare.counts'):
	os.system('python count_cfg_freq.py {} > {}'.format(train_rare_file, count_file))
	unk_replace = defaultdict(lambda:'_RARE_') # used to replace rare words in test sentence

	non_term = set() # set of non-terminals

	# a rule (unary and binary) is a tuple ('lhs', 'rhs1'[, 'rhs2'])
	# u_rules: set of unary rules
	u_rules = set()
	# bin_rules: dictionary of sets of binary rules,
	# with the key being left-hand side symbol of rules in that set
	bin_rules = defaultdict(lambda:set())

	rule_count = Counter() # count of rules
	non_term_count = Counter() # count of non-terminals
	q_params = defaultdict(lambda: -float('inf')) # q parameters

	# process count file
	for line in open(count_file,'r'):
		entry = line.strip().split()
		if entry[1] == 'NONTERMINAL':
			non_term.add(entry[2])
			non_term_count[entry[2]] = int(entry[0])
		elif entry[1] == 'UNARYRULE':
			rule = (entry[2], entry[3])
			u_rules.add(rule)
			rule_count.update({rule: int(entry[0])})
			unk_replace.update({entry[3]: entry[3]})
		elif entry[1] == 'BINARYRULE':
			rule = (entry[2], entry[3], entry[4])
			bin_rules[entry[2]].add(rule)
			rule_count.update({rule: int(entry[0])})

	# calculate q parameters for each rule
	for rule in u_rules:
		q_params[rule] = log(float(rule_count[rule])) \
				- log(float(non_term_count[rule[0]]))
	for lhs_cluster in bin_rules.itervalues():
		for rule in lhs_cluster:
			q_params[rule] = log(float(rule_count[rule])) \
					- log(float(non_term_count[rule[0]]))

	return unk_replace, non_term, bin_rules, q_params


class CKY(object):
	'''CKY parser'''
	def __init__(self, unk_replace, non_term, bin_rules, q_params):
		super(CKY, self).__init__()
		self.unk_replace = unk_replace
		self.non_term = non_term
		self.bin_rules = bin_rules
		self.q_params = q_params

	def decode(self, sent):
		print 'Parsing:', ' '.join(sent)
		tokens = [ self.unk_replace[w] for w in sent ] # replace rare words with '_RARE_'
		n = len(tokens)
		# log_probs: \pi(i,j,X) for dynamic programming
		log_probs = [[ defaultdict(lambda: -float('inf')) for j in xrange(n) ] for i in xrange(n)]
		# bp: Back-pointers. Each back-pointer is a tuple (Rule, split).
		bp = [[ dict() for j in xrange(n) ] for i in xrange(n)] # bp[i][j][X] = (Rule, split)

		# CKY Algorithm
		# initialization
		for i in xrange(n):
			for X in self.non_term:
				log_probs[i][i][X] = self.q_params[(X, tokens[i])]
		# bottom-up search
		for l in xrange(1, n):
			for i in xrange(n-l):
				j = i + l
				for X in self.non_term:
					max_rule = None
					max_s = 0
					max_log_prob = -float('inf')
					for rule in self.bin_rules[X]: # all rules with left-hand side being X
						Y = rule[1]
						Z = rule[2]
						for s in xrange(i,j):
							log_prob = self.q_params[rule] + log_probs[i][s][Y] + log_probs[s+1][j][Z]
							if log_prob >= max_log_prob:
								max_rule = rule
								max_s = s
								max_log_prob = log_prob
					log_probs[i][j][X] = max_log_prob
					bp[i][j][X] = (max_rule, max_s)
		if log_probs[0][n-1]['S'] != -float('inf'):
			return self.__build_tree(bp, sent, 'S', 0, n-1)
		else:
			max_X = 'S'
			max_log_prob = -float('inf')
			for X in self.non_term:
				if log_probs[0][n-1][X] >= max_log_prob:
					max_X = X
					max_log_prob = log_probs[0][n-1][X]
			return self.__build_tree(bp, sent, max_X, 0, n-1)

	def __build_tree(self, bp, sent, X, i, j):
		'''Build parse tree recursively'''
		if i == j:
			return [X, sent[i]]
		else:
			rule = bp[i][j][X][0]
			split = bp[i][j][X][1]
			return [X,
				self.__build_tree(bp, sent, rule[1], i, split),
				self.__build_tree(bp, sent, rule[2], split+1, j)]

def run_parser(train_input, test_input, test_output):
	print 'Preprocessing ...'
	preprocess_results = preprocess(train_rare_file=train_input, count_file='cfg.rare.counts') # preprocess
	parser = CKY(*preprocess_results) # initialize parser
	test_corpus = map(lambda x: x.strip().split(), open(test_input, 'r').readlines()) # read test set
	test_results = map(CKY.decode, [parser]*len(test_corpus), test_corpus) # parse sentences in test set
	open(test_output, 'w+').write('\n'.join( map(json.dumps, test_results) )) # dump tree to json and write to file