import argparse
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import gc

try:
    from app.object_detection import dataset
    from app.object_detection.utils import (
        multi_bbox_ious, nms, get_all_boxes,
        read_data_cfg, file_lines,
    )
    from app.object_detection.nn_logging import logging, savelog, Logger
    from app.object_detection.cfg import parse_cfg
    from app.object_detection.darknet import Darknet
except ModuleNotFoundError:
    import dataset
    from utils import (
        multi_bbox_ious, nms, get_all_boxes,
        read_data_cfg, file_lines,
    )
    from nn_logging import logging, savelog, Logger
    from cfg import parse_cfg
    from darknet import Darknet


FLAGS = None
unparsed = None
device = None

# global variables
# Training settings
# Train parameters
use_cuda      = None
eps           = 1e-5
keep_backup   = 5
save_interval = 5  # epoches
test_interval = 10  # epoches
tensorboard_logger = Logger(log_dir='./training_logs')
CHECKPOINT_PATH = './checkpoint/params.ckpt'

# Test parameters
evaluate = False
conf_thresh   = 0.25
nms_thresh    = 0.4
iou_thresh    = 0.5

# no test evalulation
no_eval = False


# Training settings
def load_testlist(testlist, model):
    init_width = model.width
    init_height = model.height

    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    loader = torch.utils.data.DataLoader(
        dataset.listDataset(
            testlist,
            shape=(init_width, init_height),
            shuffle=False,
            transform=transforms.Compose([transforms.ToTensor(),]),
            train=False),
        batch_size=batch_size,
        shuffle=False,
        **kwargs)
    return loader


def main():
    datacfg    = FLAGS.data
    cfgfile    = FLAGS.config
    weightfile = FLAGS.weights
    no_eval    = FLAGS.no_eval

    data_options  = read_data_cfg(datacfg)
    net_options   = parse_cfg(cfgfile)[0]

    global use_cuda
    use_cuda = torch.cuda.is_available()

    globals()["trainlist"]     = data_options['train']
    globals()["testlist"]      = data_options['valid']
    globals()["backupdir"]     = data_options['backup']  # where we store weights after training
    globals()["gpus"]          = data_options['gpus']  # e.g. 0,1,2,3
    globals()["ngpus"]         = len(gpus.split(','))
    globals()["num_workers"]   = int(data_options['num_workers'])

    globals()["batch_size"]    = int(net_options['batch'])
    globals()["max_batches"]   = int(net_options['max_batches'])
    globals()["learning_rate"] = float(net_options['learning_rate'])
    globals()["momentum"]      = float(net_options['momentum'])
    globals()["decay"]         = float(net_options['decay'])
    globals()["steps"]         = [float(step) for step in net_options['steps'].split(',')]
    globals()["scales"]        = [float(scale) for scale in net_options['scales'].split(',')]

    # Train parameters
    nsamples = file_lines(trainlist)
    global max_epochs
    try:
        max_epochs = int(net_options['max_epochs'])
    except KeyError:
        default_max_epochs = (max_batches*batch_size)//nsamples+1
        max_epochs = int(os.getenv('MAX_EPOCHS', default_max_epochs))

    seed = int(time.time())
    torch.manual_seed(seed)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)
    global device
    device = torch.device("cuda" if use_cuda else "cpu")

    # instantiate model
    checkpoint = load_checkpoint()
    global model
    model = Darknet(cfgfile, use_cuda=use_cuda)
    model.load_weights(weightfile)
    if checkpoint.get('model_state_dict'):
        model.load_state_dict(checkpoint.get('model_state_dict'))
    model.print_network()

    # initialize the model
    if FLAGS.reset:
        model.seen = 0
        init_epoch = 0
    else:
        curr_epoch = checkpoint.get('epoch')
        init_epoch = curr_epoch if curr_epoch else model.seen//nsamples

    global loss_layers
    loss_layers = model.loss_layers
    for l in loss_layers:
        l.seen = model.seen

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if key.find('.bn') >= 0 or key.find('.bias') >= 0:
            params += [{'params': [value], 'weight_decay': 0.0}]
        else:
            params += [{'params': [value], 'weight_decay': decay*batch_size}]

    global optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate/batch_size,
        momentum=momentum,
        dampening=0,
        weight_decay=decay * batch_size
    )
    # if checkpoint.get('optimizer_state_dict'):
        # optimizer.load_state_dict(checkpoint.get('optimizer_state_dict'))

    # need to get model height and width before transferring model to GPU
    globals()["test_loader"] = load_testlist(testlist, model)

    # load model onto CUDA
    if use_cuda:
        if ngpus > 1:
            model = torch.nn.DataParallel(model).to(device)
        else:
            model = model.to(device)

    if evaluate:
        logging('evaluating ...')
        test(0)
    else:
        try:
            print("Training for ({:d},{:d})".format(init_epoch, max_epochs))
            fscore = 0
            if not no_eval and init_epoch > test_interval:
                print('>> initial evaluating ...')
                mfscore = test(init_epoch)
                print('>> done evaluation.')
            else:
                mfscore = 0.5
            for epoch in range(init_epoch+1, max_epochs):
                epoch_start_time = time.time()

                # train or intermittent eval
                nsamples = train(epoch)
                if not no_eval and epoch > test_interval and (epoch%test_interval) == 0:
                    print('>> intermittent evaluating ...')
                    fscore = test(epoch)
                    print('>> done evaluation.')
                if epoch % save_interval == 0:
                    savemodel(epoch, nsamples)
                if FLAGS.localmax and fscore > mfscore:
                    mfscore = fscore
                    savemodel(epoch, nsamples, True)

                epoch_end_time = time.time()
                elapsed_time = round((epoch_end_time - epoch_start_time) / 60, 2)
                print('Epoch training duration: {} minutes'.format(elapsed_time))
                print('-'*90)
        except KeyboardInterrupt:
            print('='*80)
            print('Exiting from training by interrupt')


def adjust_learning_rate(optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr


def curmodel(model, ngpus=1):
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model
    return cur_model


def train(epoch):
    global processed_batches
    t0 = time.time()
    cur_model = curmodel(model, ngpus=ngpus)
    init_width = cur_model.width
    init_height = cur_model.height
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(dataset.listDataset(
        trainlist, shape=(init_width, init_height),
        shuffle=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
        train=True,
        seen=cur_model.seen,
        batch_size=batch_size,
        num_workers=num_workers
    ), batch_size=batch_size, shuffle=False, **kwargs)

    processed_batches = cur_model.seen//batch_size
    lr = adjust_learning_rate(optimizer, processed_batches)
    logging('[%03d] processed %d samples, lr %e' % (epoch, epoch * len(train_loader.dataset), lr))
    model.train()
    t1 = time.time()
    avg_time = torch.zeros(9)
    for batch_idx, (data, target) in enumerate(train_loader):
        t2 = time.time()
        adjust_learning_rate(optimizer, processed_batches)
        processed_batches = processed_batches + 1

        t3 = time.time()
        data, target = data.to(device), target.to(device)

        t4 = time.time()
        optimizer.zero_grad()

        t5 = time.time()
        output = model(data)

        t6 = time.time()
        org_loss = []
        org_loss_info = {}
        for i, l in enumerate(loss_layers):
            l.seen = l.seen + data.data.size(0)
            ol = l(output[i]['x'], target)
            org_loss.append(ol)
            log_key = f'total_layer_{i}_loss'
            org_loss_info[log_key] = ol.item()
        t7 = time.time()

        #for i, l in enumerate(reversed(org_loss)):
        #    l.backward(retain_graph=True if i < len(org_loss)-1 else False)
        # org_loss.reverse()
        sum(org_loss).backward()
        org_loss_info['total_model_loss'] = sum(org_loss)
        org_loss_info['epoch'] = epoch

        nn.utils.clip_grad_norm_(model.parameters(), 10000)
        #for p in model.parameters():
        #    p.data.add_(-lr, p.grad.data)

        t8 = time.time()
        optimizer.step()

        t9 = time.time()

        if (batch_idx + 1) % 100 == 0:
            tensorboard_logger.step += 1
            tensorboard_logger.log_scalars(**org_loss_info)
            tensorboard_logger.log_named_parameters(model.named_parameters())

        # change to True?
        if False and batch_idx > 1:
            avg_time[0] = avg_time[0] + (t2-t1)
            avg_time[1] = avg_time[1] + (t3-t2)
            avg_time[2] = avg_time[2] + (t4-t3)
            avg_time[3] = avg_time[3] + (t5-t4)
            avg_time[4] = avg_time[4] + (t6-t5)
            avg_time[5] = avg_time[5] + (t7-t6)
            avg_time[6] = avg_time[6] + (t8-t7)
            avg_time[7] = avg_time[7] + (t9-t8)
            avg_time[8] = avg_time[8] + (t9-t1)
            print('-------------------------------')
            print('       load data : %f' % (avg_time[0]/(batch_idx)))
            print('     cpu to cuda : %f' % (avg_time[1]/(batch_idx)))
            print('cuda to variable : %f' % (avg_time[2]/(batch_idx)))
            print('       zero_grad : %f' % (avg_time[3]/(batch_idx)))
            print(' forward feature : %f' % (avg_time[4]/(batch_idx)))
            print('    forward loss : %f' % (avg_time[5]/(batch_idx)))
            print('        backward : %f' % (avg_time[6]/(batch_idx)))
            print('            step : %f' % (avg_time[7]/(batch_idx)))
            print('           total : %f' % (avg_time[8]/(batch_idx)))
        t1 = time.time()
        del data, target
        org_loss.clear()
        gc.collect()

    print('')
    t1 = time.time()
    nsamples = len(train_loader.dataset)
    logging('training with %f samples/s' % (nsamples/(t1-t0)))
    return nsamples


def load_checkpoint():
    """Mainly useful for resuming training. The weights are the end goal."""
    checkpoint_path = os.path.join(os.getcwd(), FLAGS.checkpoint)
    if not os.path.exists(checkpoint_path):
        logging('no model checkpoint detected')
        return {}
    checkpoint = torch.load(checkpoint_path)
    assert 'epoch' in checkpoint.keys()
    assert 'model_state_dict' in checkpoint.keys()
    # assert 'optimizer_state_dict' in checkpoint.keys()
    # You probably saved the model using nn.DataParallel, which stores the model in module,
    # and now you are trying to load it without DataParallel.
    # You can either add a nn.DataParallel temporarily in your network for loading purposes,
    # or you can load the weights file, create a new ordered dict without the module prefix,
    # and load it back.
    valid_model_state_dict = OrderedDict()
    for key, value in checkpoint['model_state_dict'].items():
        new_key = '.'.join(key.split('.')[1:]) if 'module.models' in key else key
        assert new_key.startswith('models')
        valid_model_state_dict[new_key] = value
    checkpoint['model_state_dict_original'] = checkpoint['model_state_dict']
    checkpoint['model_state_dict'] = valid_model_state_dict
    logging('model checkpoint loaded')
    return checkpoint


def savemodel(epoch, nsamples, curmax=False):
    """Saves weights and the checkpoint."""
    cur_model = curmodel(model, ngpus=ngpus)
    if curmax:
        logging('save local maximum weights to %s/localmax.weights' % (backupdir))
    else:
        logging('save weights to %s/%06d.weights' % (backupdir, epoch))
    cur_model.seen = epoch * nsamples
    if curmax:
        cur_model.save_weights('%s/localmax.weights' % (backupdir))
    else:
        cur_model.save_weights('%s/%06d.weights' % (backupdir, epoch))
        old_wgts = '%s/%06d.weights' % (backupdir, epoch-keep_backup*save_interval)
        try: #  it avoids the unnecessary call to os.path.exists()
            os.remove(old_wgts)
        except OSError:
            pass
    logging('saving model checkpoint')
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                }, FLAGS.checkpoint)


def test(epoch):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i
        return 50

    model.eval()
    cur_model = curmodel(model, ngpus=ngpus)
    num_classes = cur_model.num_classes
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            all_boxes = get_all_boxes(output, conf_thresh, num_classes, use_cuda=use_cuda)

            for k in range(len(all_boxes)):
                boxes = all_boxes[k]
                boxes = np.array(nms(boxes, nms_thresh))
                truths = target[k].view(-1, 5)
                num_gts = truths_length(truths)
                total = total + num_gts
                num_pred = len(boxes)
                if num_pred == 0:
                    continue

                proposals += int((boxes[:,4]>conf_thresh).sum())
                for i in range(num_gts):
                    gt_boxes = torch.FloatTensor([truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]])
                    gt_boxes = gt_boxes.repeat(num_pred,1).t()
                    pred_boxes = torch.FloatTensor(boxes).t()
                    best_iou, best_j = torch.max(multi_bbox_ious(gt_boxes, pred_boxes, x1y1x2y2=False),0)
                    # pred_boxes and gt_boxes are transposed for torch.max
                    if best_iou > iou_thresh and pred_boxes[6][best_j] == gt_boxes[6][0]:
                        correct += 1

    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = 2.0*precision*recall/(precision+recall+eps)
    savelog("[%03d] correct: %d, precision: %f, recall: %f, fscore: %f" % (epoch, correct, precision, recall, fscore))
    return fscore


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d',
        type=str, default='cfg/sketch.data', help='data definition file')
    parser.add_argument('--config', '-c',
        type=str, default='cfg/sketch.cfg', help='network configuration file')
    parser.add_argument('--weights', '-w',
        type=str, default='weights/yolov3.weights', help='initial weights file')
    parser.add_argument('--noeval', '-n', dest='no_eval', action='store_true',
        help='prohibit test evalulation')
    parser.add_argument('--reset', '-r',
        action="store_true", default=False, help='initialize the epoch and model seen value')
    parser.add_argument('--localmax', '-l',
        action="store_true", default=False, help='save net weights for local maximum fscore')
    parser.add_argument('--checkpoint', '-p',
        default=CHECKPOINT_PATH, help='path to save and load model params')

    FLAGS, _ = parser.parse_known_args()
    main()
