import base64
import os
from logging import getLogger

import cv2


LOGGER = getLogger(__name__)


def show_info():
    return cv2.__version__


def vid2img(file_path, output_filetype='jpg', folder=None):
    filename = os.path.splitext(file_path)[0]
    LOGGER.info(f'Reading in file: {filename}')
    vidcap = cv2.VideoCapture(file_path)
    count = 0
    ret = True
    while ret:
        ret, frame = vidcap.read()
        dest_path = f'{filename}_{count}.{output_filetype}'
        write_image(dest_path, frame, folder=folder)
        if ret:
            LOGGER.info(f'[{count}] Saved another frame')
            count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def read_image(file_path):
    LOGGER.info(f'Reading in file from {file_path}')
    return cv2.imread(file_path, cv2.IMREAD_COLOR)


def write_image(dest_path, img, folder=None):
    # It might be advantageous to keep the full filepath until writing
    # to an output file.  So only strip the path upon output write.
    dest_path = os.path.basename(dest_path)
    if folder:
        if not os.path.isdir(folder):
            os.mkdir(folder)
        dest_path = os.path.join(folder, dest_path)
    return cv2.imwrite(dest_path, img)


def _html_img_frame(frame):
    retval, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    html = "<img src='data:image/jpg;base64, " + jpg_as_text + "'>"
    return html
