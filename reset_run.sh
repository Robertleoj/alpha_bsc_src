#!/bin/bash

if [ "$USER" = "gimli" ]; then
  echo "You are Gimli. Are you sure you want to reset the run? (y/n)"
    read sure
    if [ "$sure" != "y" ]; then
        echo "Exiting."
        exit 1
    fi
fi


rm ./db/db.db 
rm ./models/*/*.pt 
rm ./py_src/training_data/*/*.pt 

source ./py_src/.venv/bin/activate

cd py_src
python3 init_conn4_net.py

cd ..
./init_db.sh



