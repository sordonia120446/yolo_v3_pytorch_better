.PHONY: clean clean-voc detect detect-carpk pascal purge train train-carpk train-pklot weights

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
	@python detect.py cfg/yolo_v3.cfg yolov3.weights data/dog-cycle-car.png data/coco.names

detect-carpk:
	@echo "Detecting based on weights trained on CARPK data"
	@python detect.py cfg/carpk.cfg carpk.weights 20161225_TPZ_00459.png data/carpk.names

detect-pklot:
	@echo "Detecting based on weights trained on PKLOT data"
	@python detect.py cfg/pklot.cfg pklot.weights pklot_sample.jpg data/pklot.names

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

train-pklot:
	@echo 'Update training data pathing in the cfg/*.data files'
	@python train.py --noeval -d cfg/pklot.data -c cfg/pklot.cfg -w yolov3.weights

weights:
	@echo "Getting YOLO v3 weights" && wget https://pjreddie.com/media/files/yolov3.weights yolo_v3/yolov3.weights
