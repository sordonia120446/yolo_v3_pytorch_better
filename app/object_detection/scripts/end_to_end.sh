#!/usr/bin/env bash

echo "Detecting cars..."

mkdir -p throughput_data

# TODO change these weights to the actual ones
python detect.py \
    -c cfg/carpk_v3.cfg \
    -w weights/carpk_v3_half_size.weights \
    -n data/carpk.names \
    -z data/cars_same_view.zip

echo -e "\nAnalyzing results\n"

docker-compose run skynet python /app/app/object_detection/throughput.py
