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

COMPRESSED_PATH=./saved_runs/$RUN_NAME.7z
OUTPUT_PATH=./vault/$GAME_NAME

# make sure the run exists
if [ ! -f "$COMPRESSED_PATH" ]; then
    echo "Run \"$RUN_NAME\" does not exist"
    exit 1
fi


7z x $COMPRESSED_PATH -o$OUTPUT_PATH
