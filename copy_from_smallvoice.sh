#!/bin/bash

# Check if arguments are passed
if [ $# -eq 0 ]; then
    echo "Usage: $0 <run-name>"
    exit 1
fi

RUN_NAME=$1

mkdir -p ./saved_runs/$RUN_NAME
scp gimli@smallvoice.ru.is:/home/gimli/AlphaBSc/alpha_bsc_src/saved_runs/$RUN_NAME.7z ./saved_runs/
