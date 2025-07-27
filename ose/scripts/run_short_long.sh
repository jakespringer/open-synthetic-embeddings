#!/bin/bash

set -e

echo "Running short_long data generation pipeline..."

echo "Step 1: Brainstorming..."
python -m ose.generate --config ose/configs/short_long/short_long_brainstorm.yaml
if [ $? -ne 0 ]; then
    echo "ERROR: Step 1 (Brainstorming) failed!"
    exit 1
fi

echo "Step 2: Collecting brainstorming ideas..."
python -m ose.collect --config ose/configs/short_long/short_long_brainstorm_collect.yaml
if [ $? -ne 0 ]; then
    echo "ERROR: Step 2 (Collecting brainstorming ideas) failed!"
    exit 1
fi

echo "Step 3: Main generation..."
python -m ose.generate --config ose/configs/short_long/short_long_generation.yaml
if [ $? -ne 0 ]; then
    echo "ERROR: Step 3 (Main generation) failed!"
    exit 1
fi

echo "Step 4: Collecting generated data..."
python -m ose.collect --config ose/configs/short_long/short_long_collect.yaml
if [ $? -ne 0 ]; then
    echo "ERROR: Step 4 (Collecting generated data) failed!"
    exit 1
fi

echo "Step 5: Generating hard negatives..."
python -m ose.collect --config ose/configs/short_long/short_long_hard_negatives.yaml
if [ $? -ne 0 ]; then
    echo "ERROR: Step 5 (Generating hard negatives) failed!"
    exit 1
fi

echo "short_long pipeline completed successfully!"
