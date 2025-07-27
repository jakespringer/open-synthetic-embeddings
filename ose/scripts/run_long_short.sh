#!/bin/bash

set -e

echo "Running long_short data generation pipeline..."

echo "Step 1: Brainstorming..."
python -m ose.generate --config ose/configs/long_short/long_short_brainstorm.yaml
if [ $? -ne 0 ]; then
    echo "ERROR: Step 1 (Brainstorming) failed!"
    exit 1
fi

echo "Step 2: Collecting brainstorming ideas..."
python -m ose.collect --config ose/configs/long_short/long_short_brainstorm_collect.yaml
if [ $? -ne 0 ]; then
    echo "ERROR: Step 2 (Collecting brainstorming ideas) failed!"
    exit 1
fi

echo "Step 3: Main generation..."
python -m ose.generate --config ose/configs/long_short/long_short_generation.yaml
if [ $? -ne 0 ]; then
    echo "ERROR: Step 3 (Main generation) failed!"
    exit 1
fi

echo "Step 4: Collecting generated data..."
python -m ose.collect --config ose/configs/long_short/long_short_collect.yaml
if [ $? -ne 0 ]; then
    echo "ERROR: Step 4 (Collecting generated data) failed!"
    exit 1
fi

echo "long_short pipeline completed successfully!" 