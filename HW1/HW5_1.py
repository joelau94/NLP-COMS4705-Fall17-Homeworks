#!/usr/bin/python

import os
import sys
from collections import defaultdict
from collections import Counter
from math import log

def get_q_param(count_file='ner.counts'): # compute emission parameters (as log-probabilities)
	raw = [line.split() for line in open(count_file,'r')] # read count file, split each line
	bigram = { a[2]+' '+a[3] : int(a[0]) for a in raw if a[1]=='2-GRAM' } # extract bigram ('y_i-1 y_i': count)
	trigram = { a[2]+' '+a[3]+' '+a[4] : int(a[0]) for a in raw if a[1]=='3-GRAM' } # extract trigram ('y_i-2 y_i-1 y_i': count)
	q_param = defaultdict(lambda: -float('inf'))
	for k,v in trigram.iteritems(): # q_param['y_i-2 y_i-1 y_i'] is the log probability of trigram y_i-2,y_i-1,y_i
		q_param[k] = log(float(v)) - log(float( bigram[' '.join(k.split()[:-1])] ))
	return q_param

def rm_rare(in_file='ner_train.dat', out_file='ner_train.rare.dat', min_freq=5): # replace infrequent words with '_RARE_'
	count = Counter([line.split()[0] for line in open(in_file,'r') if line.strip()!='']) # count word frequency
	open(out_file,'w+').write('\n'.join([ line.strip()
		if line.strip()=='' or count[line.split()[0]]>=min_freq else '_RARE_ '+line.split()[1]
		for line in open(in_file,'r') ])) # write data without low-frequency words into another file

if __name__ == '__main__':
	rm_rare() # replace infrequent words in training file with '_RARE_', write into 'ner_train.rare.dat'
	os.system('python count_freqs.py ner_train.rare.dat > ner.counts') # generate count file
	q_param = get_q_param()
	out = open('5_1.txt','w+')
	for line in open('trigrams.txt','r'):
		trigram = line.strip()
		out.write("{} {}\n".format(trigram,q_param[trigram]))
