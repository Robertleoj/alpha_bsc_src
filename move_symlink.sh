#!/bin/bash

# Check if arguments are given
if [ $# -eq 0 ]; then
  echo "Usage: $0 <run-name> [<game-name>]"
  exit 1
fi

RUN_NAME=$1
GAME_NAME="connect4"

SSD_DIR=/media/bigbrainman/stuff/alpha_bsc_dbs/$GAME_NAME/$RUN_NAME

RUN_DIR=./vault/$GAME_NAME/$RUN_NAME

rm $RUN_DIR/db.db
ln -s $SSD_DIR/db.db `pwd`/vault/$GAME_NAME/$RUN_NAME/db.db
