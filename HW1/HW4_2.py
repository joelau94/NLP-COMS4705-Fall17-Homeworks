#!/usr/bin/python

import os
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

def rm_rare(in_file='ner_train.dat', out_file='ner_train.dat.rare_processed', min_freq=5): # replace infrequent words with '_RARE_'
	count = Counter([line.split()[0] for line in open(in_file,'r') if line.strip()!='']) # count word frequency
	open(out_file,'w+').write('\n'.join([ line.strip()
		if line.strip()=='' or count[line.split()[0]]>=min_freq else '_RARE_ '+line.split()[1]
		for line in open(in_file,'r') ])) # write data without low-frequency words into another file

def get_ner_dev_dat(key_file='ner_dev.key', dat_file='ner_dev.dat'): # remove tags from ner_dev.key
	open(dat_file,'w+').write('\n'.join([line.split()[0] if line.strip()!='' else '' for line in open(key_file,'r')]))

def decode(in_file='ner_dev.dat', out_file='ner_dev.out'): # baseline tagger
	e_param = get_e_param() # calculate emission parameters (as log-probabilities)
	open(out_file,'w+').write( '\n'.join([ '{} {} {}'.format(line.strip(),
		*max(e_param[line.strip() if line.strip() in e_param else '_RARE_'].iteritems(), key=lambda e: e[1]))
		if line.strip()!='' else '' for line in open(in_file,'r') ] + ['\n']*2 ) ) # take argmax for each word
		# 2 empty lines added to the end of prediction file because eval_ne_tagger.py requires that
		# the number of lines in prediciton file and gold-standard file should be the same

if __name__ == '__main__':
	rm_rare() # replace infrequent words in training file with '_RARE_', write into 'ner_train.dat.rare_processed'
	os.system('python count_freqs.py ner_train.dat.rare_processed > ner.counts') # generate count file
	get_ner_dev_dat() # remove tag from 'ner_dev.key', write into file 'ner_dev.dat'
	decode(out_file='4_2.txt') # do tagging (using baseline tagger).
	# os.system('python eval_ne_tagger.py ner_dev.key ner_dev.out') # evaluate baseline tagger
