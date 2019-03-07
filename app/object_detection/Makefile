.PHONY: clean clean-voc detect detect-carpk detect-cnr detect-pklot detect-pucpr pascal purge train train-carpk train-cnr train-pklot train-pucpr weights

all:
	@echo "make clean - delete yolo_output image folder"
	@echo "make clean-voc - delete all VOC data"
	@echo "make detect - detect an image"
	@echo "make pascal - download and build the Pascal VOC dataset"
	@echo "make train - train on Pascal VOC data"
	@echo "make weights - get YOLO v3 weights"

clean:
	@rm -rf yolo_output/
	@rm -rf training_logs/

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

detect-cnr:
	@mkdir -p predictions
	@echo "Detecting based on weights trained on CARPK data"
	@python detect.py -c cfg/cnr.cfg -w cnr.weights -n data/cnr.names -t 0.05 -i cnr_sample2.jpg

detect-pklot:
	@mkdir -p predictions
	@echo "Detecting based on weights trained on PKLOT data"
	@python detect.py -c cfg/pklot.cfg -w pklot.weights -n data/pklot.names -i pklot_sample.jpg

detect-pucpr:
	@mkdir -p predictions
	@echo "Detecting based on weights trained on PUCPR data"
	@python detect.py -c cfg/pucpr.cfg -w pucpr.weights -n data/pucpr.names -t 0.1 -i 3_Sunny.jpg

pascal:
	@bash data/pascal_data/pascal.sh

purge: clean clean-voc

tensorboard:
	@tensorboard --logdir='./training_logs' --port=6006

train:
	@echo 'Update training data pathing in the cfg/*.data files'
	@python train.py --noeval -d cfg/voc.data -c cfg/yolo_v3.cfg -w yolov3.weights

train-carpk:
	@echo 'Update training data pathing in the cfg/*.data files'
	@python train.py --noeval -d cfg/carpk.data -c cfg/carpk.cfg -w yolov3.weights

train-cnr:
	@echo 'Update training data pathing in the cfg/*.data files'
	@python train.py --noeval -d cfg/cnr.data -c cfg/cnr.cfg -w yolov3.weights

train-pklot:
	@echo 'Update training data pathing in the cfg/*.data files'
	@python train.py --noeval -d cfg/pklot.data -c cfg/pklot.cfg -w yolov3.weights

train-pucpr:
	@echo 'Update training data pathing in the cfg/*.data files'
	@python train.py --noeval -d cfg/pucpr.data -c cfg/pucpr.cfg -w yolov3.weights

weights:
	@echo "Getting YOLO v3 weights" && wget https://pjreddie.com/media/files/yolov3.weights yolo_v3/yolov3.weights
