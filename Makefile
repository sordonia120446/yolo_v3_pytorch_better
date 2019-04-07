.PHONY: build clean clean-data clean-tpt opencv status test_video

all: build test_video opencv

build:
	@docker-compose build
	@echo "Container built. To install Pytorch, follow README.md instructions."

clean: clean-data clean-tpt

clean-data:
	@rm -rf opencv_data/
	@rm -rf test_video/

clean-tpt:
	make -C app/object_detection clean-tpt

shell:
	@docker-compose run skynet bash

status:
	@docker stats --no-stream

tensorboard:
	make -C app/object_detection tensorboard

test:
	@mkdir -p app/object_detection/throughput_data/
	@pytest -p no:warnings

watch-gpu:
	make -C app/object_detection watch-gpu
