#!/bin/bash

#DATASET_PATH='/home/skhalid/Documents/datalake/shakespeare.txt'
# DATASET_PATH='/home/skhalid/Documents/datalake/cifar-10-python/cifar-10-batches-py'
#rm -rf /tmp/haiku-transformer/checkpoint_* && python -W ignore train.py --job_id "./jobs/1.json"

jobs=$(ls -lrt ./jobs | awk '{print $9}')

for job in ${jobs[@]}
do 
    echo $job
    sbatch ./_launch.sh $job
done	


