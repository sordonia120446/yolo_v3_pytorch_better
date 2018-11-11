.PHONY: clean pascal train

all: build test_video opencv

build:
	@docker-compose build

build-fresh:
	@docker-compose build --pull --no-cache

clean: clean-data

clean-data:
	@rm -rf opencv_data/
	@rm -rf test_video/
	@rm -rf yolo_output/

clean-voc:
	@bash data/pascal_data/clean.sh

clean-docker:
	# Only run this when the current Docker is running
	@docker system prune -af

detect:
	@docker-compose run -e DEBUG vision python detect.py cfg/yolo_v3.cfg yolov3.weights data/dog-cycle-car.png data/coco.names

detect-naked:
	@python detect.py cfg/yolo_v3.cfg yolov3.weights data/dog-cycle-car.png data/coco.names

opencv:
	@docker-compose run -e DEBUG vision python opencv.py -h

pascal:
	@bash data/pascal_data/pascal.sh

purge: clean clean-voc

status:
	@docker stats --no-stream

test_video:
	@docker-compose run -e DEBUG vision bash ./app/opencv/get_test_video.sh
	@docker-compose run -e DEBUG vision python opencv.py -v test_video/big_buck_bunny.mp4

train:
	@echo 'Update training data pathing in the cfg/*.data files'
	@python train.py -d cfg/voc.data -c cfg/yolo_v3.cfg -w yolov3.weights

weights:
	@echo "Getting YOLO v3 weights" && wget https://pjreddie.com/media/files/yolov3.weights yolo_v3/yolov3.weights
