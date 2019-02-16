.PHONY: build clean clean-data opencv status test_video

all: build test_video opencv

build:
	@docker-compose build

clean: clean-data

clean-data:
	@rm -rf opencv_data/
	@rm -rf test_video/

opencv:
	@docker-compose run -e DEBUG vision python opencv.py -h

status:
	@docker stats --no-stream

test:
	@echo "Running unit tests."
	@pytest -p no:warnings

test_video:
	@docker-compose run -e DEBUG vision bash ./app/opencv/get_test_video.sh
	@docker-compose run -e DEBUG vision python opencv.py -v test_video/big_buck_bunny.mp4
