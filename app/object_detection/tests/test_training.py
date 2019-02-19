from unittest import TestCase, mock

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from app.object_detection.tests import fixtures as Fixtures

from app.object_detection.train import curmodel
from app.object_detection import darknet as Darknet


class TrainHelperFncsTestCase(TestCase):
    def test_cur_model(self):
        mock_model = mock.MagicMock()
        mock_model.module = 'fake_layer'
        model = curmodel(mock_model, ngpus=1)
        self.assertEquals(mock_model, model)

        model = curmodel(mock_model, ngpus=2)
        self.assertEquals(mock_model.module, model)


class LayerTestCase(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        inputs = Variable(torch.randn(20, 20))
        targets = Variable(torch.randint(0, 2, (20,))).long()
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda:
            inputs.cuda()
            targets.cuda()
        self.batch = [inputs, targets]

    def test_linear_layer(self):
        model = nn.Linear(20, 2)
        if self.is_cuda:
            model.cuda()
        initial_params, params = Fixtures.var_change_helper(
            model, F.cross_entropy, torch.optim.Adam(model.parameters()), self.batch)

        for (_, p0), (name, p1) in zip(initial_params, params):
            fail_msg = 'did not change'
            assert not torch.equal(p0, p1), f'{name} {fail_msg}'

    def test_empty_layer(self):
        model = Darknet.EmptyModule()
        if self.is_cuda:
            model.cuda()
        assert torch.equal(self.batch[0], model.forward(self.batch[0]))

    def test_global_average_pool_2d_layer(self):
        inputs = Variable(torch.randn(2, 2, 2, 2))
        if self.is_cuda:
            inputs.cuda()
        model = Darknet.GlobalAvgPool2d()
        if self.is_cuda:
            model.cuda()
        assert not torch.equal(inputs, model.forward(inputs))

    def test_max_pool_stride(self):
        inputs = Variable(torch.randn(2, 2, 2, 2))
        if self.is_cuda:
            inputs.cuda()
        model = Darknet.MaxPoolStride1()
        if self.is_cuda:
            model.cuda()
        assert not torch.equal(inputs, model.forward(inputs))
