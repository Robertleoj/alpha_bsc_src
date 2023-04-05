#!/bin/bash

# Check if arguments are given
if [ $# -eq 0 ]; then
  echo "Usage: $0 <run-name> [<game-name>]"
  exit 1
fi

RUN_NAME=$1

# check if second argument is given
if [ $# -eq 2 ]; then
  # check if second argument is a valid game name
    if [ "$2" != "connect4" ] && [ "$2" != "breakthrough" ]; then
        echo "\"$2\" is not a valid game name"
        exit 1
    fi
    GAME_NAME=$2
else
    GAME_NAME="connect4"
fi

RUN_DIR=./vault/$GAME_NAME/$RUN_NAME
DEFAULT_DIR=./vault/$GAME_NAME/defaults

mkdir -p $RUN_DIR

mkdir $RUN_DIR/cached_data
mkdir $RUN_DIR/models

# copy the default files
cp $DEFAULT_DIR/* $RUN_DIR

sqlite3 $RUN_DIR/db.db < ./db/configure_db.sql

cd ./py_src

source .venv/bin/activate

python3 init_nn.py ../$RUN_DIR/models/ $GAME_NAME

echo "Run $RUN_NAME has been creatd."
echo "REMEMBER TO EDIT THE CONFIG"



