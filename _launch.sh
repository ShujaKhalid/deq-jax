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

. /etc/profile.d/lmod.sh
. ../jax.env
module use /pkgs/environment-modules/
module load pytorch1.7.1-cuda11.0-python3.6
#./package_installer.sh

# Run the job 
python -W ignore train.py --job_id $1

# --- MISC ---
#(while true; do nvidia-smi; top -b -n 1 | head -20; sleep 10; done) &
#python /h/skhalid/pytorch.py
#wait
