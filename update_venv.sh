#!/bin/bash

{
VENV_DIR=~/venv/IRP

# REQUIREMENTS_FILE=../requirements.txt
BASE_DIR=$(dirname $(realpath $0))
REQUIREMENTS_FILE=$BASE_DIR/requirements.txt
HASH_FILE=$VENV_DIR/requirements.hash

# Create hash of current requirements file
NEW_HASH=$(md5sum $REQUIREMENTS_FILE | awk '{ print $1 }')

# Check if the hash file exists
if [ -f $HASH_FILE ]; then
    OLD_HASH=$(cat $HASH_FILE)
else
    OLD_HASH=""
fi

# Compare hashes
if [ "$NEW_HASH" != "$OLD_HASH" ]; then
    echo "Updating virtual environment"
    source $VENV_DIR/bin/activate
    pip install --upgrade -r $REQUIREMENTS_FILE
    echo $NEW_HASH > $HASH_FILE
else
    echo "No updates needed for virtual environment"
    echo "Packages: $(cat $REQUIREMENTS_FILE)"
fi
} # > /dev/null 2>&1