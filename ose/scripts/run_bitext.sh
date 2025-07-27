#!/bin/bash

set -e

echo "Running bitext data generation pipeline..."

echo "Step 1: Main generation..."
python -m ose.generate --config ose/configs/bitext/bitext_generation.yaml
if [ $? -ne 0 ]; then
    echo "ERROR: Step 1 (Main generation) failed!"
    exit 1
fi

echo "Step 2: Collecting generated data..."
python -m ose.collect --config ose/configs/bitext/bitext_collect.yaml
if [ $? -ne 0 ]; then
    echo "ERROR: Step 2 (Collecting generated data) failed!"
    exit 1
fi

echo "bitext pipeline completed successfully!" 