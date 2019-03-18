.PHONY: clean clean-tpt clean-voc detect detect-carpk detect-cnr detect-pklot detect-pklot-v3 detect-pucpr pascal purge skynet train train-carpk train-cnr train-pklot train-pklot-v3 train-pucpr weights

all:
	@echo "make clean - delete yolo_output image folder"
	@echo "make clean-voc - delete all VOC data"
	@echo "make detect - detect an image"
	@echo "make pascal - download and build the Pascal VOC dataset"
	@echo "make skynet - detect cars in list of images and print stats"
	@echo "make train - train on Pascal VOC data. Update training data pathing in the cfg/*.data files"
	@echo "make weights - get YOLO v3 weights"

clean: clean-detect clean-training

clean-training:
	@rm -rf training_logs/

clean-detect:
	@rm -rf throughput/

clean-voc:
	@bash data/pascal_data/clean.sh

detect:
	@mkdir -p predictions
	@echo "Detecting based on weights trained on COCO data"
	@python detect.py -c cfg/yolo_v3.cfg -w yolov3.weights -n data/coco.names -i data/dog-cycle-car.png

detect-carpk:
	@mkdir -p predictions
	@echo "Detecting based on weights trained on CARPK data"
	@python detect.py -c cfg/carpk.cfg -w carpk.weights -n data/carpk.names -i 20161225_TPZ_00459.png

detect-carpk-v3:
	@mkdir -p predictions
	@echo "Detecting based on weights trained on CARPK data and YOLO v3"
	@python detect.py -c cfg/carpk_v3.cfg -w carpk_v3.weights -n data/carpk.names -t 0.2 -i 20161225_TPZ_00459.png carpk_sample.png

detect-cnr:
	@mkdir -p predictions
	@echo "Detecting based on weights trained on CNR data"
	@python detect.py -c cfg/cnr.cfg -w cnr.weights -n data/cnr.names -t 0.05 -i cnr_sunny.jpg cnr_overcast.jpg cnr_rainy.jpg

detect-pklot:
	@mkdir -p predictions
	@echo "Detecting based on weights trained on PKLOT data"
	@python detect.py -c cfg/pklot.cfg -w pklot.weights -n data/pklot.names -i pklot_sample.jpg

detect-pklot-v3:
	@mkdir -p predictions
	@echo "Detecting based on weights trained on PKLOT data and YOLO v3"
	@python detect.py -c cfg/pklot_v3.cfg -w pklot_v3.weights -n data/pklot.names -t 0.05 -i 3_Sunny.jpg pklot_sample.jpg

detect-pucpr:
	@mkdir -p predictions
	@echo "Detecting based on weights trained on PUCPR data"
	@python detect.py -c cfg/pucpr.cfg -w pucpr.weights -n data/pucpr.names -t 0.1 -i 3_Sunny.jpg

pascal:
	@bash data/pascal_data/pascal.sh

purge: clean clean-voc

skynet:
	@bash scripts/end_to_end.sh

tensorboard:
	@tensorboard --logdir='./training_logs' --port=6006

train:
	@python train.py --noeval -d cfg/voc.data -c cfg/yolo_v3.cfg -w yolov3.weights

train-carpk:
	@python train.py --noeval -d cfg/carpk.data -c cfg/carpk.cfg -w yolov3.weights

train-carpk-v3:
	@python train.py --noeval -d cfg/carpk.data -c cfg/carpk_v3.cfg -w yolov3.weights -p checkpoint/carpk_v3.ckpt

train-cnr:
	@python train.py --noeval -d cfg/cnr.data -c cfg/cnr.cfg -p checkpoint/cnr.ckpt -w yolov3.weights

train-pklot:
	@python train.py --noeval -d cfg/pklot.data -c cfg/pklot.cfg -w yolov3.weights

train-pklot-v3:
	@echo 'Training on YOLO v3 config'
	@python train.py --noeval -d cfg/pklot.data -c cfg/pklot_v3.cfg -w yolov3.weights -p checkpoint/pklot_v3.ckpt

train-pucpr:
	@python train.py --noeval -d cfg/pucpr.data -c cfg/pucpr.cfg -w yolov3.weights

watch-gpu:
	@watch -n 0.3 nvidia-smi

weights:
	@echo "Getting YOLO v3 weights"
	@wget -N https://pjreddie.com/media/files/yolov3.weights .
	@wget -N https://pjreddie.com/media/files/darknet53.conv.74 .
