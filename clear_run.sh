#!/bin/bash

echo "This will delete the database, models, and cached training data. Are you sure? (y/n)"

read sure

if [ "$sure" != "y" ]; then
    echo "Exiting."
    exit 1
fi

rm ./db/db.db
rm ./models/*/*.pt
rm ./py_src/training_data/*/*.pt
