#!/bin/bash

#DATASET_PATH='/home/skhalid/Documents/datalake/shakespeare.txt'
DATASET_PATH='/home/skhalid/Documents/datalake/cifar-10-python/cifar-10-batches-py'
rm -rf /tmp/haiku-transformer/checkpoint_0000000.pkl && python -W ignore train.py --dataset_path=$DATASET_PATH

