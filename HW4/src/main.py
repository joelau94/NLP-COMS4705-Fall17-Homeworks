import os, sys
import json

from parser import *

def part_1():
	config_1 = Config(name='1')
	config_1.save()
	parser_1 = Parser(config_1)
	# parser_1.train()
	print('Testing on dev data...')
	parser_1.test('trees/dev.conll', 'output/dev_part1.conll')
	print('Testing on test data ...')
	parser_1.test('trees/test.conll', 'output/test_part1.conll')

def part_2():
	config_2 = Config(name='2')
	config_2.config['hid_1_dim'] = config_2.config['hid_2_dim'] = 400
	config_2.save()
	parser_2 = Parser(config_2)
	parser_2.train()
	print('Testing on dev data...')
	parser_2.test('trees/dev.conll', 'output/dev_part2.conll')
	print('Testing on test data ...')
	parser_2.test('trees/test.conll', 'output/test_part2.conll')

def part_3():
	config_3 = Config(name='3')
	config_3.config['transfer'] = 'cubic'
	config_3.config['l2_reg'] = 'True'
	config_3.config['epochs'] = 20
	config_3.save()
	parser_3 = Parser(config_3)
	parser_3.train()
	print('Testing on dev data...')
	parser_3.test('trees/dev.conll', 'output/dev_part3.conll')
	print('Testing on test data ...')
	parser_3.test('trees/test.conll', 'output/test_part3.conll')

if __name__ == '__main__':
	if str(sys.argv[1]) == '1':
		part_1()
	elif str(sys.argv[1]) == '2':
		part_2()
	elif str(sys.argv[1]) == '3':
		part_3()
	else:
		print('Argument must be 1, 2, or 3!')