#!/usr/bin/env bash

echo "Detecting cars..."

mkdir -p throughput_data

if [ -z $TARGET ]; then
    echo ""
    echo "bruhhhhhhhhhh"
    echo "Specify a value for TARGET via make command. E.g.,"
    echo ""
    echo "make TARGET=<target-type> skynet"
    echo ""
    exit 1
fi

if [ $TARGET == "cnr" ]; then
    python detect.py \
        -c cfg/cnr_v3.cfg \
        -w weights/cnr_v3.weights \
        -t 0.25 \
        -n data/cnr.names \
        -z data/sunny_2015_11_12.zip
elif [ $TARGET == "carpk" ]; then
    python detect.py \
        -c cfg/carpk_v3.cfg \
        -w weights/carpk_v3_half_size.weights \
        -n data/carpk.names \
        -z data/carpk_samples.zip
elif [ $TARGET == "carpk-v2" ]; then
    python detect.py \
        -c cfg/carpk.cfg \
        -w weights/carpk.weights \
        -n data/carpk.names \
        -z data/carpk_samples.zip
elif [ $TARGET == "cnr-v2" ]; then
    python detect.py \
        -c cfg/cnr.cfg \
        -w weights/cnr.weights \
        -t 0.1 \
        -n data/cnr.names \
        -z data/sunny_2015_11_12.zip
elif [ $TARGET == "" ]; then
    echo ""
    echo "bruhhhhhhhhhh"
    echo "Specify a value for TARGET via make command. E.g.,"
    echo ""
    echo "make TARGET=<target-type> skynet"
    echo ""
    exit 1
else
    echo "Unrecognized target. Exiting..."
    exit 1
fi

echo -e "\nAnalyzing results\n"

docker-compose run skynet python /app/app/object_detection/throughput.py
