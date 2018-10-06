all: build test_video opencv

build:
	@docker-compose build

build-fresh:
	@docker-compose build --pull --no-cache

clean:
	@rm -rf opencv_data/
	@rm -rf test_video/
	@rm -rf yolo_output/

clean-docker:
	@docker system prune -af

opencv:
	@docker-compose run -e DEBUG web python opencv.py -h

purge: clean clean-docker

status:
	@docker stats --no-stream

test_video:
	@docker-compose run -e DEBUG web bash ./app/opencv/get_test_video.sh
	@docker-compose run -e DEBUG web python opencv.py -v test_video/big_buck_bunny.mp4
