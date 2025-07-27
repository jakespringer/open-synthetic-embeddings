#!/bin/bash

set -e

echo "Starting synthetic data generation for all data types..."
echo "=============================================="

echo ""
echo "Running short_short pipeline..."
bash ose/scripts/run_short_short.sh
if [ $? -ne 0 ]; then
    echo "ERROR: short_short pipeline failed!"
    exit 1
fi

echo ""
echo "Running short_long pipeline..."
bash ose/scripts/run_short_long.sh
if [ $? -ne 0 ]; then
    echo "ERROR: short_long pipeline failed!"
    exit 1
fi

echo ""
echo "Running long_short pipeline..."
bash ose/scripts/run_long_short.sh
if [ $? -ne 0 ]; then
    echo "ERROR: long_short pipeline failed!"
    exit 1
fi

echo ""
echo "Running long_long pipeline..."
bash ose/scripts/run_long_long.sh
if [ $? -ne 0 ]; then
    echo "ERROR: long_long pipeline failed!"
    exit 1
fi

echo ""
echo "Running sts pipeline..."
bash ose/scripts/run_sts.sh
if [ $? -ne 0 ]; then
    echo "ERROR: sts pipeline failed!"
    exit 1
fi

echo ""
echo "Running bitext pipeline..."
bash ose/scripts/run_bitext.sh
if [ $? -ne 0 ]; then
    echo "ERROR: bitext pipeline failed!"
    exit 1
fi

echo ""
echo "=============================================="
echo "All synthetic data generation pipelines completed successfully!" 