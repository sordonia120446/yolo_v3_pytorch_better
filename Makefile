.PHONY: build clean clean-data clean-tpt opencv status test_video

all: build test_video opencv

build:
	@docker-compose build

clean: clean-data clean-tpt

clean-data:
	@rm -rf opencv_data/
	@rm -rf test_video/

clean-tpt:
	@rm -rf throughput/
	make -C app/object_detection clean-tpt

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

watch-gpu:
	make -C app/object_detection watch-gpu
