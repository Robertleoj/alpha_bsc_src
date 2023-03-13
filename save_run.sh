#!/bin/bash


if [ $# -eq 0 ]; then
  echo "Usage: $0 <run-name> [<game-name>]"
  exit 1
fi

RUN_NAME=$1

GAME="connect4"
if [ $# -eq 2 ]; then
  # check if second argument is a valid game name
    if [ "$2" != "connect4" ] && [ "$2" != "breakthrough" ]; then
        echo "\"$2\" is not a valid game name"
        exit 1
    fi
    GAME=$2
fi


mkdir -p ./saved_runs

RUN_DIR=./vault/$GAME/$RUN_NAME

# make sure the run exists
if [ ! -d "$RUN_DIR" ]; then
    echo "Run \"$RUN_NAME\" does not exist"
    exit 1
fi

COMPRESSED_PATH=./saved_runs/$RUN_NAME.7z

7z a $COMPRESSED_PATH $RUN_DIR


