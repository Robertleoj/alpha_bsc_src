#!/bin/bash

#SBATCH --account=gimli
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb                 
#SBATCH --partition=beQuick


RUNPATH=/home/gimli/AlphaBSc/alpha_bsc_src/py_src

cd $RUNPATH

source ./.venv/bin/activate

python3 train.py