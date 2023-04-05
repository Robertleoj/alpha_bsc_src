#!/bin/bash

# check if the run name is given
if [ $# -eq 0 ]; then
    echo "Usage: $0 <run_name>"
    exit 1
fi

# get the run name
RUN_NAME=$1

# ask the user if they are sure
echo "Are you sure you want to delete run $RUN_NAME? (y/n)"
read -r CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "Aborting."
    exit 1
fi

GAME_NAME="connect4"

# get the optional game argument (second argument)
if [ $# -eq 2 ]; then
    GAME_NAME=$2
fi

# check if directory exists
if [ ! -d "./vault/$GAME_NAME/$RUN_NAME" ]; then
    echo "Run $RUN_NAME for $GAME_NAME does not exist."
    exit 1
fi


# delete the run
echo "Deleting run $RUN_NAME..."
rm -rf "./vault/$GAME_NAME/$RUN_NAME"

echo "Done."