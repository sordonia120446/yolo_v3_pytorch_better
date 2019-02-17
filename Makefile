.PHONY: build clean clean-data opencv status test_video

all: build test_video opencv

build:
	@docker-compose build

clean: clean-data

clean-data:
	@rm -rf opencv_data/
	@rm -rf test_video/

detect-carpk:
	@echo "Detecting based on weights trained on CARPK data"
	@python detect.py cfg/sam.cfg yolov3.weights data/dog-cycle-car.png data/sam.names

opencv:
	@docker-compose run -e DEBUG vision python opencv.py -h

status:
	@docker stats --no-stream

tensorboard:
	make -C app/object_detection tensorboard

test:
	@echo "Running unit tests."
	@pytest -p no:warnings

test_video:
	@docker-compose run -e DEBUG vision bash ./app/opencv/get_test_video.sh
	@docker-compose run -e DEBUG vision python opencv.py -v test_video/big_buck_bunny.mp4
