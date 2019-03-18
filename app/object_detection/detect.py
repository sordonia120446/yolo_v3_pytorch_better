import argparse
import os
import time
import zipfile

import torch
from PIL import Image

from utils import get_all_boxes, image2torch, load_class_names, nms, plot_boxes
from darknet import Darknet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='path to cfg file')
    parser.add_argument('--weights', '-w', help='path to weights file')
    parser.add_argument('--names', '-n', help='path to namesfile')
    parser.add_argument('--imgfiles', '-i', nargs='+', help='path to image')
    parser.add_argument('--zip-imgs', '-z', help='zip file containing image input(s)')
    parser.add_argument('--conf-thresh', '-t', default=0.25, type=float, help='Confidence threshold of object detection')
    return parser.parse_args()


def _traverse_image_list(imgfiles):
    inputs = []
    for imgfile in args.imgfiles:
        inputs.append({
            'filename': imgfile,
            'image': Image.open(imgfile),
        })
    return inputs


def _traverse_zip(zip_filepath):
    inputs = []
    with zipfile.ZipFile(zip_filepath) as z_f_in:
        z_f_in.extractall('data')  # hacky but whatever
    data_dir = os.path.splitext(zip_filepath)[0]
    for filename in os.listdir(data_dir):
        basename, ext = os.path.splitext(filename)
        if ext not in {'.png', '.jpg', '.jpeg', '.JPEG'}:
            continue
        img_path = os.path.join(data_dir, filename)
        inputs.append({
            'filename': filename,
            'image': Image.open(img_path),
        })
    return inputs


def detect(model, img, conf_thresh, nms_thresh, use_cuda):
    img = image2torch(img)
    img = img.to(torch.device('cuda' if use_cuda else 'cpu'))
    out_boxes = model(img)
    boxes = get_all_boxes(out_boxes, conf_thresh, model.num_classes, use_cuda=use_cuda)[0]
    return nms(boxes, nms_thresh)


def main(args):
    model = Darknet(args.config)

    model.print_network()
    model.load_weights(args.weights)
    print(f'Loading weights from {args.weights}... Done!')

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()

    size = (model.width, model.height)

    inputs = []
    if args.zip_imgs:
        inputs = _traverse_zip(args.zip_imgs)
    else:
        inputs = _traverse_image_list(args.imgfiles)

    model.eval()
    start = time.time()
    for input_ in inputs:
        img = input_['image'].convert('RGB').resize(size)
        imgfile = input_['filename']
        # used to be higher confidence threshold and nms threshold
        # boxes = detect(model, img, 0.5, 0.4, use_cuda)
        boxes = detect(model, img, args.conf_thresh, 0.4, use_cuda)
        class_names = load_class_names(args.names)
        savename = f'predicted_{os.path.basename(imgfile)}'
        plot_boxes(img, boxes, savename, class_names)

    finish = time.time()
    print(f'{args.imgfiles}: Predicted in {finish - start} seconds.')


if __name__ == '__main__':
    args = parse_args()
    main(args)
