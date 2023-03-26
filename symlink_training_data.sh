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

mkdir -p $RUN_DIR/training_data

SSD_DIR=/media/bigbrainman/stuff/alpha_bsc_dbs/$GAME_NAME/$RUN_NAME

mkdir -p $SSD_DIR

mv $RUN_DIR/training_data $SSD_DIR/training_data

ln -s $SSD_DIR/training_data `pwd`/vault/$GAME_NAME/$RUN_NAME/training_data

