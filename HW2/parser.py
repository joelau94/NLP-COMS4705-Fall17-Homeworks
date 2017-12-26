#!/usr/bin/python

import sys
import Q4.Main
import Q5.Main

if __name__ == '__main__':
	q = str(sys.argv[1])
	if q == 'q4':
		trainData = str(sys.argv[2])
		trainRareData = str(sys.argv[3])
		min_word_freq = 5
		if len(sys.argv) >= 5:
			min_word_freq = int(sys.argv[4])
		Q4.Main.file_rm_rare(trainData, trainRareData, min_word_freq=min_word_freq)
	elif q == 'q5' or q == 'q6':
		train_input = str(sys.argv[2])
		test_input = str(sys.argv[3])
		test_output = str(sys.argv[4])
		Q5.Main.run_parser(train_input, test_input, test_output)
