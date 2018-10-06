import argparse
import logging
import sys

from webapp.opencv import image_processing as ImageProcessing
from webapp.opencv import image_manipulation as ImageManipulation


# Config
logging.getLogger(__name__).setLevel(logging.INFO)
FOLDER = 'opencv_data'


"""CLARGS"""
parser = argparse.ArgumentParser(
    description='Open CV utility commands for image operations.',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='For further questions, please consult the README.',
)

# Image Processing
parser.add_argument(
    'input_path',
    help='Path to input file.',
)
parser.add_argument(
    '-v',
    '--video',
    action='store_true',
    help='Specify this to convert a video into frame-by-frame images.',
)

# Image Transformations
parser.add_argument(
    '-at',
    '--affine-transform',
    action='store_true',
    help='Rotate input image by some number of degrees.',
)
parser.add_argument(
    '-c',
    '--concatenate',
    action='store_true',
    help='Rotate input image by some number of degrees.',
)
parser.add_argument(
    '-r',
    '--resize',
    default=1.0,
    type=float,
    help='Specify this to resize an image by some input factor.',
)
parser.add_argument(
    '-s',
    '--spin',
    default=0,
    type=int,
    help='Rotate input image by some number of degrees.',
)


if __name__ == '__main__':
    args = parser.parse_args()
    print('Open CV version: {}'.format(ImageProcessing.show_info()))

    if args.video:
        # TODO add video file type validation
        ImageProcessing.vid2img(args.input_path, folder=FOLDER)
        sys.exit(0)

    img = ImageProcessing.read_image(args.input_path)
    if args.resize and args.resize != 1.0:
        factor = args.resize
        output_img = ImageManipulation.resize(img, factor)
        dest_path = f'resize_{factor}_{args.input_path}'
    elif args.spin and args.spin != 0:
        degrees = args.spin
        output_img = ImageManipulation.rotate(img, degrees)
        dest_path = f'rotate_{degrees}_{args.input_path}'
    elif args.affine_transform:
        output_img = ImageManipulation.affine_transform(img)
        dest_path = f'affine_transform_{args.input_path}'
    elif args.concatenate:
        output_img = ImageManipulation.concatenate(img)
        dest_path = f'concatenate_transform_{args.input_path}'
    ImageProcessing.write_image(dest_path, output_img, folder=FOLDER)
