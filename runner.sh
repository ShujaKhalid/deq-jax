#!/bin/bash

#DATASET_PATH='/home/skhalid/Documents/datalake/shakespeare.txt'
# DATASET_PATH='/home/skhalid/Documents/datalake/cifar-10-python/cifar-10-batches-py'
rm -rf /tmp/haiku-transformer/checkpoint_* && python -W ignore train.py --job_id "./jobs/1.json"

