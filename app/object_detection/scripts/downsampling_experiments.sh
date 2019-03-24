echo "Make sure you only run one detector command per execution. Or just change the output image name, so you don't overwrite previous predictions like a nubcake."

python detect.py -c cfg/carpk_v3.cfg -w weights/carpk_v3_diff_anchors.weights -n data/carpk.names -t 0.3 -i 20161225_TPZ_00459.png carpk_sample.png

python detect.py -c cfg/carpk_v3.cfg -w weights/carpk_v3_half_size.weights -n data/carpk.names -t 0.3 -i 20161225_TPZ_00459.png carpk_sample.png

python detect.py -c cfg/carpk_v3.cfg -w weights/carpk_v3_quarter_size.weights -n data/carpk.names -t 0.3 -i 20161225_TPZ_00459.png carpk_sample.png

echo "Detection complete. Check the predictions folder."
