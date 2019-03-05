import argparse
import os
import time

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
    return parser.parse_args()


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

    model.eval()
    start = time.time()
    for imgfile in args.imgfiles:
        img = Image.open(imgfile).convert('RGB').resize(size)
        # used to be higher confidence threshold and nms threshold
        # boxes = detect(model, img, 0.5, 0.4, use_cuda)
        boxes = detect(model, img, 0.25, 0.2, use_cuda)
        class_names = load_class_names(args.names)
        savename = f'predicted_{os.path.basename(imgfile)}'
        plot_boxes(img, boxes, savename, class_names)

    finish = time.time()
    print(f'{args.imgfiles}: Predicted in {finish - start} seconds.')


if __name__ == '__main__':
    args = parse_args()
    main(args)
