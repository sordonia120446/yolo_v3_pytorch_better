from unittest import TestCase, mock

from app.object_detection.train import curmodel


class TrainTestCase(TestCase):
    def test_cur_model(self):
        mock_model = mock.MagicMock()
        mock_model.module = 'fake_layer'
        model = curmodel(mock_model, ngpus=1)
        self.assertEquals(mock_model, model)

        model = curmodel(mock_model, ngpus=2)
        self.assertEquals(mock_model.module, model)
