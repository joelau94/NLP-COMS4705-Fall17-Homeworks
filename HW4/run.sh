#!/bin/sh

DEVICE=cpu

PYTHONPATH=./ THEANO_FLAGS=floatX=float32,device=$DEVICE,lib.cnmem=0.2 python ./src/main.py 1
python src/eval.py trees/dev.conll output/dev_part1.conll

PYTHONPATH=./ THEANO_FLAGS=floatX=float32,device=$DEVICE,lib.cnmem=0.2 python ./src/main.py 2
python src/eval.py trees/dev.conll output/dev_part2.conll

PYTHONPATH=./ THEANO_FLAGS=floatX=float32,device=$DEVICE,lib.cnmem=0.2 python ./src/main.py 3
python src/eval.py trees/dev.conll output/dev_part3.conll