#!/bin/bash


rm ./db/db.db
rm ./models/*/*.pt
rm ./py_src/training_data/*/*.pt

source ./py_src/.venv/bin/activate

cd py_src
python3 init_conn4_net.py

cd ..
./init_db.sh



