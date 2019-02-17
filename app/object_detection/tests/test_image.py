from unittest import TestCase

from app.object_detection.image import _get_label_path


class ImageTestCase(TestCase):
    def test_get_label_path(self):
        label_path1 = _get_label_path('VOC/data/Images/jackie_chan.jpg')
        label_path2 = _get_label_path('VOC/data/JPEGImages/jet_li.jpg')

        self.assertEquals('VOC/data/labels/jackie_chan.txt', label_path1)
        self.assertEquals('VOC/data/labels/jet_li.txt', label_path2)

        with self.assertRaises(KeyError):
            _get_label_path('VOC/data/docs/whatever.docx')
