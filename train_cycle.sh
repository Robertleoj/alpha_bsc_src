#!/bin/bash

#SBATCH --account=staff
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32gb                 
#SBATCH --partition=beQuick
#SBATCH --output=logs/train_window.log


RUNPATH=/home/gimli/AlphaBSc/alpha_bsc_src

cd $RUNPATH

source ./py_src/.venv/bin/activate

python3 train_cycle.py $1 $2



