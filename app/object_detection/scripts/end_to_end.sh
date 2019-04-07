#!/usr/bin/env bash

echo "Detecting cars..."

mkdir -p throughput

python detect.py \
    -c cfg/carpk.cfg \
    -w weights/yolov3.weights \
    -n data/carpk.names \
    -z data/cars_same_view.zip

echo -e "\nAnalyzing results\n"

docker-compose run skynet python /app/app/object_detection/throughput.py
