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

CACHED_DIR=./vault/$GAME_NAME/$RUN_NAME/cached_data

find $CACHED_DIR -name '*.pt' | awk -F/ '{print $NF}' | awk -F. '{print $1}' | sort -n | tail -1

