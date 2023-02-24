#!/bin/bash

#SBATCH --account=gimli
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32gb                 
#SBATCH --partition=beQuick
#SBATCH --nodelist=samson

# https://crc.pitt.edu/user-support/job-scheduling-policy/submitting-multiple-jobs-cluster 
# to train on multiple nodes with --array


RUNPATH=/home/gimli/AlphaBSc/alpha_bsc_src/cpp_src

cd $RUNPATH

./self_play

