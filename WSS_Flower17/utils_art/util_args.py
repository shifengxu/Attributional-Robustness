import os
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}.")


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--datadir', default=None, type=str)
    parser.add_argument('--datamaskdir', default=None, type=str)
    parser.add_argument('--gpu-ids', default=None, type=int, nargs='+', help='GPU IDs to use.')
    parser.add_argument('--batch-size', default=None, type=int, metavar='N')
    parser.add_argument('--epochs', default=None, type=int, metavar='N')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
    parser.add_argument('--enable-exemplar', default=None, type=str2bool)

    parser.add_argument('--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')

    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet',
                        help='model architecture: default: resnet18)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=30, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    # CASE NAME
    parser.add_argument('--name', type=str, default='test_case')
    parser.add_argument('--task', type=str, default='cls')

    # path
    parser.add_argument('--save-dir', type=str, default='checkpoints/')
    parser.add_argument('--image-save', action='store_true')

    # basic hyperparameters
    parser.add_argument('--lr', '--learning-rate', default=None, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-decay', type=int, default=None, help='Reducing lr frequency')
    parser.add_argument('--lr-decay-ratio', type=float, default=None, help='Reducing lr ratio')
    parser.add_argument('--lr-ratio', type=float, default=10)
    parser.add_argument('--nest', action='store_true')

    parser.add_argument('--grad', type=str, default=None, help="gradient option: g|ig")
    parser.add_argument('--cam-thr', type=float, default=0.1, help='cam threshold value')
    parser.add_argument('--grad-thr', type=float, default=0.2, help='grad threshold value')
    parser.add_argument('--cam-curve', action='store_true')
    parser.add_argument('--grad-curve', action='store_true')

    # bbox
    # data transform
    parser.add_argument('--resize-size', type=int, default=256, help='validation resize size')
    parser.add_argument('--crop-size', type=int, default=224, help='validation crop size')

    args = parser.parse_args()
    return args
