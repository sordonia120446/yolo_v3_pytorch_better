#!/usr/bin/env bash

echo "Detecting cars..."

python detect.py \
    -c cfg/carpk.cfg \
    -w carpk.weights \
    -n data/carpk.names \
    -i 20161225_TPZ_00459.png carpk_sample.png

echo -e "\nAnalyzing results\n"

python throughput.py
