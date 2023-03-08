#!/bin/bash

mkdir -p saved_runs

mkdir -p $1

7z a ./saved_runs/$1/db.7z -r ./db/db.db
7z a ./saved_runs/$1/models.7z -r ./models
