#!/usr/bin/python

import os
import sys
from collections import defaultdict
from collections import Counter
from math import log

def get_e_param(count_file='ner.counts'): # compute emission parameters (as log-probabilities)
	raw = [line.split() for line in open(count_file,'r')] # read count file, split each line
	e_info = [ (int(a[0]), a[2], a[3]) for a in raw if a[1]=='WORDTAG' ] # extract (count, y, x)
	y_count = defaultdict(int)
	for e in e_info:
		y_count[e[1]] += e[0] # count y
	e_param = defaultdict(lambda:defaultdict(lambda:-float('inf'))) # store e(x|y) in e_param[x][y]
	for e in e_info:
		e_param[e[2]].update({ e[1]: log(float(e[0]))-log(float(y_count[e[1]])) }) # e[2]:x, e[1]:y, e[0]:count
	return e_param

def get_q_param(count_file='ner.counts'): # compute emission parameters (as log-probabilities)
	raw = [line.split() for line in open(count_file,'r')] # read count file, split each line
	bigram = { a[2]+' '+a[3] : int(a[0]) for a in raw if a[1]=='2-GRAM' } # extract bigram ('y_i-1 y_i': count)
	trigram = { a[2]+' '+a[3]+' '+a[4] : int(a[0]) for a in raw if a[1]=='3-GRAM' } # extract trigram ('y_i-2 y_i-1 y_i': count)
	q_param = defaultdict(lambda: -float('inf'))
	for k,v in trigram.iteritems(): # q_param['y_i-2 y_i-1 y_i'] is the log probability of trigram y_i-2,y_i-1,y_i
		q_param[k] = log(float(v)) - log(float( bigram[' '.join(k.split()[:-1])] ))
	return q_param

def rm_rare(unk_replace, in_file='ner_train.dat', out_file='ner_train.rare.dat'): # replace infrequent words with '_RARE_'
	count = Counter([line.split()[0] for line in open(in_file,'r') if line.strip()!='']) # count word frequency
	open(out_file,'w+').write('\n'.join([ unk_replace[line.split()[0]]+' '+line.split()[1]
		if line.strip()!='' else ''
		for line in open(in_file,'r') ])) # write data without low-frequency words into another file

def get_ner_dev_dat(key_file='ner_dev.key', dat_file='ner_dev.dat'): # remove tags from ner_dev.key
	open(dat_file,'w+').write('\n'.join([line.split()[0] if line.strip()!='' else '' for line in open(key_file,'r')]))

def get_unk_replace(corpus_file='ner_train.dat', min_freq=5, use_rules=False): # get a lookup table for replacing infrequent words
	count = Counter([line.split()[0] for line in open(corpus_file,'r') if line.strip()!=''])
	unk_replace = defaultdict(lambda:'_RARE_')
	for k in count.keys():
		if count[k]>=min_freq:
			unk_replace[k] = k
		elif use_rules:
			if any(i.isdigit() for i in k):
				unk_replace[k] = '_containsDigit_'
			elif k[0].isupper():
				unk_replace[k] = '_initCap_'
	return unk_replace

def load_test_data(in_file):
	data = []
	sentence = []
	for line in open(in_file,'r'):
		if line.strip() == '' and len(sentence)>0:
			data.append(sentence)
			sentence = []
		else:
			sentence.append(line.strip())
	if len(sentence) > 0:
		data.append(sentence)
	return data

def decode(e, q, sent, unk_replace):
	"""
	e: emission parameters returned by get_e_param()
	q: trigram log-probabilities returned by get_q_param()
	sent: a properly tokenized sentence
	"""
	def tag_set(idx):
		if idx == -1 or idx == 0:
			return ['*']
		elif idx > 0:
			return ['I-PER', 'I-ORG', 'B-ORG', 'I-LOC', 'B-LOC', 'I-MISC', 'B-MISC', 'O']
		else:
			return []
	x = ['*'] + [ unk_replace[w] for w in sent ]
	n = len(x) - 1 # actual sentence length: len(x)-1, because '*' prepended.
	log_probs = [ defaultdict(lambda:defaultdict(lambda:-float('inf'))) for i in xrange(n+1) ]
	log_probs[0]['*']['*'] = 0.
	bp = [ defaultdict(lambda:defaultdict(lambda:'_UNDEFINED_')) for i in xrange(n+1) ]
	for k in xrange(1, n+1):
		# print("====== k:{} ======\n".format(k))
		for y_pre in tag_set(k-1):
			for y in tag_set(k):
				lprob_max = -float('inf')
				tag_max = ''
				for y_pre_pre in tag_set(k-2):
					lprob = log_probs[ k-1 ][ y_pre_pre ][ y_pre ] \
							+ q[ y_pre_pre+' '+y_pre+' '+y ] \
							+ e[ x[k] ][ y ]
					# print("lp[{}][{}][{}]={}".format(k-1,y_pre_pre,y_pre,log_probs[ k-1 ][ y_pre_pre ][ y_pre ]))
					# print("q[{} {} {}]={}".format(y_pre_pre,y_pre,y,q[ y_pre_pre+' '+y_pre+' '+y ]))
					# print("e[{}][{}]={}\n".format(x[k],y,e[ x[k] ][ y ]))
					if lprob >= lprob_max:
						lprob_max = lprob
						tag_max = y_pre_pre
				log_probs[k][y_pre][y] = lprob_max
				bp[k][y_pre][y] = tag_max
				# print("lprob_max=log_probs[{}][{}][{}]={}, tag_max={}\n".format(k,y_pre,y,lprob_max,tag_max)) # debug
	y_n_max = y_n_pre_max = ''
	lprob_max = -float('inf')
	for y in tag_set(1):
		for y_pre in tag_set(1):
			lprob = log_probs[n][y_pre][y] + q[ y_pre+' '+y+' STOP' ]
			if lprob >= lprob_max:
				lprob_max = lprob
				y_n_max = y
				y_n_pre_max = y_pre
	y_result = [y_n_pre_max, y_n_max]
	lprobs_result = [lprob_max]
	for k in xrange(n-2, 0, -1):
		y_result = [ bp[k+2][y_result[0]][y_result[1]] ] + y_result
		lprobs_result = [ log_probs[k+2][y_result[0]][y_result[1]] ] + lprobs_result
	lprobs_result = [0] + lprobs_result
	return y_result, lprobs_result

if __name__ == '__main__':
	unk_replace = get_unk_replace(min_freq=2, use_rules=False)
	rm_rare(unk_replace) # replace infrequent words in training file with a set of rules write into 'ner_train.rare.dat'
	os.system('python count_freqs.py ner_train.rare.dat > ner.counts') # generate count file
	e = get_e_param()
	q = get_q_param()

	# test on development data
	get_ner_dev_dat() # remove tag from 'ner_dev.key', write into file 'ner_dev.dat'
	test_data = load_test_data('ner_dev.dat')
	test_output = open('6.txt','w+')
	for sent in test_data:
		tags, lprobs = decode(e, q, sent, unk_replace)
		for x, y, p in zip(sent, tags, lprobs):
			test_output.write("{} {} {}\n".format(x,y,p))
		test_output.write('\n')