#!/bin/bash

#DATASET_PATH='/home/skhalid/Documents/datalake/shakespeare.txt'
# DATASET_PATH='/home/skhalid/Documents/datalake/cifar-10-python/cifar-10-batches-py'
#rm -rf /tmp/haiku-transformer/checkpoint_* && python -W ignore train.py --job_id "./jobs/base.json"
rm -rf /tmp/haiku-transformer/checkpoint_* && python -W ignore train.py --job_id "./jobs/base_test.json"

# jobs=$(ls -lrt ./jobs/vector/*.json | awk '{print $9}')
# jobs_dir=./jobs/vector/
# for job in ${jobs[@]}
# do
#     echo $job
#     sbatch ./_launch.sh $job
#     sleep 2
# done
