# YOLO v3 on Pytorch

Pytorch implementation of the YOLO v3 SSD. Detects and trains.

## Setup

### Virtual Environment

The best way to get this running is to set up a virtualenv. Upon creation and
activation, run below cmd to set up Pytorch and other relevant dependencies.

```
pip install -r requirements.txt
```

### Docker

Currently, Docker support for Pytorch is a no-go. There's some weird CUDA issue when attempting to run on GPU.

However, to support the throughput analysis of cars in a parking lot, a Docker container is provided for summarizing, plotting, and saving car counts data. This application is called `Skynet`.

To set up this container, run the following. This will pull from Vision and set up anything additional to run this within this repo.

```
make build
```

To execute the application, run `make skynet` in `app/object_detection`. Check the `throughput_data` folder in that same subdirectory for the csv results and the `time_series.png` plot. Note that this command needs to run within the `venv`, since its first part is to execute the Pytorch detection model.

### Getting Training Data

So far, only VOC data collection is set up. Execute below command to download
and compile the VOC data into the appropriate file structure.

```
make pascal
```

Additionally, the corresponding `voc.data` file should be updated.

The structure should be like below.

```
train  = data/pascal_data/voc_train.txt
valid  = data/pascal_data/2007_test.txt
names = data/voc.names
backup = backup
gpus  = 0,1,2,3
```

You might want to remove the training data. Maybe something just went awry or whatever. To do so run, below.

```
make clean-voc
```

## Training

Be sure to define `max_epochs` in the `.cfg` file of the model being trained. There is also the option to add it as an environment variable `MAX_EPOCHS`.

## Tests

Yes, there are tests. To run them locally, run `make test` at the top-level (be sure to run within the venv, of course). They are also run on every push to the repo, courtesy of Travis.
