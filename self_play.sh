#!/bin/bash

#SBATCH --account=gimli
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb                 
#SBATCH --partition=beQuick

# can use --array according to 
# https://crc.pitt.edu/user-support/job-scheduling-policy/submitting-multiple-jobs-cluster
# to train on multiple nodes


RUNPATH=/home/gimli/AlphaBSc/alpha_bsc_src/cpp_src

cd $RUNPATH

./self_play