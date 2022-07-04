#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p rtx6000 
#SBATCH --cpus-per-task=3
#SBATCH --time=180:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=deq
#SBATCH --output=deq-job-%j.out

nvidia-smi
nvcc -V
. /etc/profile.d/lmod.sh
. ../jax.env
#conda activate /scratch/ssd001/home/skhalid/jax

CUDNN_VER=cudnn-11.0-v8.0.5.39
module use /pkgs/environment-modules/
module --ignore-cache load $CUDA_VER

PATH=/pkgs/$CUDA_VER/bin:$PATH
LD_LIBRARY_PATH=/pkgs/$CUDA_VER/lib64:/pkgs/$CUDNN_VER/lib64:/pkgs/nccl_2.8.4-1+cuda11.1_x86_64:$LD_LIBRARY_PATH


# Run the job 
python -W ignore train.py --job_id $1

# --- MISC ---
#(while true; do nvidia-smi; top -b -n 1 | head -20; sleep 10; done) &
#python /h/skhalid/pytorch.py
#wait
