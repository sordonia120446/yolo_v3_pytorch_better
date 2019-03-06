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

Currently, Docker support is a no-go. There's some weird CUDA issue when
attempting to run on GPU.

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
