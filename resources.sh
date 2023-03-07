#!/bin/bash

#SBATCH --account=gimli
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=1
#SBATCH --mem=1gb
#SBATCH --partition=beQuick
#SBATCH --nodelist=samson
#SBATCH --output=./logs/resources.log

# https://crc.pitt.edu/user-support/job-scheduling-policy/submitting-multiple-jobs-cluster 
# to train on multiple nodes with --array

cd /home/gimli/AlphaBSc/alpha_bsc_src

echo "############## NVIDIA-SMI #####################"
nvidia-smi
echo "############## FREE  #####################"
free -g

ps aux --sort=pcpu > logs/procs.txt






