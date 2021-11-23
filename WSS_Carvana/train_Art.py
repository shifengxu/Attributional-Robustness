import torch.optim
from torch.autograd import Variable
import network.unet as unet
from utils_art.util_args import get_args
from utils_art.util_acc import adjust_learning_rate, \
    AverageEpochMeter, SumEpochMeter
from utils_art.util_loader import data_loader
from utils_art.util_eval import *
from utils_art.util import *
import os
import sys
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if parent_dir not in sys.path: sys.path.insert(0, parent_dir)
# from shared.util_normalize import normalize
from art_attack import * 
import loss_art

LEARNING_RATE = 1e-6


def main(args):
    model = unet.UNET(in_channels=3, out_channels=1)
    if len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    model = model.to(args.device)

    # define loss function (criterion) and optimizer
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # param_features = []
    # param_classifiers = []
    # for name, parameter in model.named_parameters():
    #     if 'layer4.' in name or 'fc.' in name:
    #         param_classifiers.append(parameter)
    #     else:
    #         param_features.append(parameter)
    # optimizer = torch.optim.SGD([
    #         {'params': param_features,    'lr': args.lr},
    #         {'params': param_classifiers, 'lr': args.lr * args.lr_ratio}],
    #         momentum=args.momentum,
    #         weight_decay=args.weight_decay,
    #         nesterov=args.nest)

    train_loader, val_loader = data_loader(args)

    print("Batch Size: %d" % args.batch_size)
    print("saliency training started")

    check_accuracy(val_loader, model, device=args.device)
    for epoch in range(args.start_epoch, args.epochs):
        print("Start Epoch %d/%d ..." % (epoch+1, args.epochs))
        adjust_learning_rate(optimizer, epoch, args)
        train(train_loader, model, criterion, optimizer, epoch, args)
        check_accuracy(val_loader, model, device=args.device)
    # for


def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()  # switch to train mode
    b_cnt = len(train_loader)
    for i, (x, y) in enumerate(train_loader):
        # A PyTorch Variable is a wrapper around a PyTorch Tensor, and represents a node in a
        # computational graph. If x is a Variable then x.data is a Tensor giving its value,
        # and x.grad is another Variable holding the gradient of x with respect to some scalar value.
        y = Variable(y.cuda(args.device))
        x = Variable(x.cuda(args.device))
        y = y.unsqueeze(dim=1)  # y is mask map, containing only 0 and 1. 1 means masked

        if args.enable_art:
            exemplar_loss, scores = calc_exemplar_loss(x, y, model)
        else:
            exemplar_loss, scores = 0, model(x)
        loss = criterion(scores, y)
        optimizer.zero_grad()
        total_loss = loss + 0.5*exemplar_loss
        total_loss.backward()       
        optimizer.step()
        print(f"E{epoch}.B{i}/{b_cnt} loss:{loss:.4f}; exemp:{exemplar_loss:.4f}; total:{total_loss:.4f}")
    # for


def calc_exemplar_loss(x, y, model):
    x.requires_grad = True
    scores = model(x)
    g = torch.autograd.grad(scores.mean(), x, retain_graph=True, create_graph=True)[0]
    # g.shape: [40, 3, 160, 240]. 40 is batch size
    y_rgb = torch.cat((y, y, y), dim=1)  # y is singe channel, extend it to RGB 3 channels
    y2 = 1 - y_rgb
    g1 = g * y_rgb  # the gradient inside the mask
    g2 = g * y2  # the gradient outside the mask

    # g1.shape: [40, 3, 160, 240]. 40 is batch size
    # g1_adp.shape: [40, 160, 240]
    g1_adp = torch.mean(torch.abs(g1), 1)
    g2_adp = torch.mean(torch.abs(g2), 1)
    x_conv = torch.mean(torch.abs(x.detach()), 1)
    # Tensor.detach() Returns a new Tensor, detached from the current graph.
    # The result will never require gradient.
    # Returned Tensor shared the same storage with the original one.
    # In-place modification will trigger an error, or affect the original Tensor.

    x.requires_grad = False
    batch_size = x.size(0)
    x_r = x_conv.reshape(batch_size, -1)
    g1_r = g1_adp.reshape(batch_size, -1)
    g2_r = g2_adp.reshape(batch_size, -1)
    exemplar_loss = loss_art.exemplar_loss_fn(x_r, g1_r, g2_r)
    return exemplar_loss


if __name__ == '__main__':
    print("process id:", os.getpid())
    args = get_args()
    args.device = f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() and args.gpu_ids else "cpu"
    args.train_img_dir = "data/train_images/"
    args.train_msk_dir = "data/train_masks/"
    args.val_img_dir = "data/val_images/"
    args.val_msk_dir = "data/val_masks/"
    args.enable_art = False
    print(args)
    main(args)
