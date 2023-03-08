#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <run-name>"
  exit 1
fi

scp -r gimli@smallvoice.ru.is:/home/gimli/AlphaBSc/alpha_bsc_src/saved_runs/$1 ./saved_runs/$1