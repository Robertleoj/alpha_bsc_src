#!/bin/bash

# Set the directory you want to check

if [ -f ./db/db.db ]; then
    echo "Database file exists. Exiting"
    exit 1
fi

if [ "$(ls -A ./models/*/)" ]; then
    echo "Models directory is not empty. Exiting"
    exit 1
fi

7z x ./saved_runs/$1/db.7z 
7z x ./saved_runs/$1/models.7z 

if [ "$(ls -A ./py_src/training_data/*/)" ]; then
    echo "Warning: cached training data has not been deleted!"
fi
