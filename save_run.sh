#!/bin/bash

mkdir -p ./saved_runs/$1

7z a ./saved_runs/$1/db.7z -r ./db/db.db
7z a ./saved_runs/$1/models.7z ./models

cp cpp_src/config_files/hyperparameters.json ./saved_runs/$1/hyperparameters.json
cp py_src/config.py ./saved_runs/$1/config.py

