#!/bin/bash

DATASET_PATH='/home/skhalid/Documents/datalake/shakespeare.txt'
rm -rf /tmp/haiku-transformer/checkpoint_0000000.pkl && python train.py --dataset_path=$DATASET_PATH

