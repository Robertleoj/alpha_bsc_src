#!/bin/bash

#SBATCH --account=gimli
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32gb                 
#SBATCH --partition=beQuick


RUNPATH=/home/gimli/AlphaBSc/alpha_bsc_src/cpp_src

cd $RUNPATH

./eval_agent

