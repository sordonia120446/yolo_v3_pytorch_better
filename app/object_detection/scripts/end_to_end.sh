#!/usr/bin/env bash

echo "Detecting cars..."

mkdir -p throughput

python detect.py \
    -c cfg/carpk.cfg \
    -w carpk.weights \
    -n data/carpk.names \
    -z data/cars_same_view.zip

echo -e "\nAnalyzing results\n"

python throughput.py
