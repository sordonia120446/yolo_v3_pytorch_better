from unittest import TestCase

from app.object_detection.throughput import track_throughput


class ThroughputTestCase(TestCase):
    def test_make_throughput_folder(self):
        boxes = ['i', 'am', 'a', 'fake', 'box', 'list']
        predictions_save_path = 'predictions/test_img.jpg'
        track_throughput(boxes, predictions_save_path)
