# Hacky YOLO V3 with Pytorch

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

## Additional Notes from Alex Guy

### Difference between this repository and marvis original version.
* some programs are re-structured for windows environments.
(for example \_\_name\_\_ == '\_\_main\_\_' (variable in python program) is checked for multiple threads).
* load and save weights are modified to compatible to yolov2 and yolov3 versions
(means that this repository works for yolov2 and yolov3 configuration without source modification.)
* fully support yolov3 detection and training
   * region_loss.py is renamed to region_layer.py.
   * outputs of region_layer.py and yolo_layer.py are enclosed to dictionary variables.
* codes are modified to work on pytorch 0.4 and python3
* some codes are modified to speed up and easy readings. (I'm not sure.. T_T)
* in training mode, check nan value and use gradient clipping.

#### If you want to know the training and detect procedures, please refer to https://github.com/marvis/pytorch-yolo2 for the detail information.

### Train your own data or coco, voc data as follows:
```
python train.py -d cfg/coco.data -c cfg/yolo_v3.cfg -w yolov3.weights
```

* new weights are saved in backup directory along to epoch numbers (last 5 weights are saved, you control the number of backups in train.py)

* The above command shows the example of training process. I didn't execute the above command. But, I did successully train my own data with the pretrained yolov3.weights.

* You __should__ notice that the anchor information is different when it used in yolov2 or yolov3 model.

* If you want to use the pretrained weight as the initial weights, add -r option in the training command

```
python train.py -d cfg/my.data -c cfg/my.cfg -w yolov3.weights -r
```

* maximum epochs option, which is automatically calculated, somestimes is too small, then you can set the max_epochs in your configuration.

### Detect the objects in dog image using pretrained weights

#### yolov2 models
```
wget http://pjreddie.com/media/files/yolo.weights
python detect.py cfg/yolo.cfg yolo.weights data/dog.jpg data/coco.names
```

#### yolov3 models
```
wget https://pjreddie.com/media/files/yolov3.weights
python detect.py cfg/yolo_v3.cfg yolov3.weights data/dog.jpg data/coco.names
```

### validation and get evaluation results

```
valid.py data/yourown.data cfg/yourown.cfg yourown_weights
scripts/sketch_eval.py prefix test_files class_names
```

Heres are results:
```
AP for button = 0.9593
AP for checkbutton = 0.9202
AP for edittext = 0.8424
AP for imageview = 0.8356
AP for radiobutton = 0.8827
AP for spinner = 0.9539
AP for textview = 0.7971
Mean AP = 0.8845
~~~~~~~~~~~~~
   Results:
-------------
button          0.959
checkbutton     0.920
edittext        0.842
imageview       0.836
radiobutton     0.883
spinner         0.954
textview        0.797
=============
 Average        0.884
~~~~~~~~~~~~~
```

### License

MIT License (see LICENSE file).

## References

This repository is forked from great work pytorch-yolo2 of @github/marvis,
but I couldn't upload or modify directly to marvis source files because many files were changed even filenames.
