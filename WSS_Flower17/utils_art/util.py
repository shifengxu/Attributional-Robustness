# -*- coding: utf-8 -*-
import datetime
import os
import torch
import torch.optim
import torchvision


def load_model(model, optimizer, args):
    if not os.path.isfile(args.resume):
        log_info("=> no checkpoint found at '{}'".format(args.resume))
        return model, optimizer

    log_info("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    log_info("=> loading checkpoint '{}' done".format(args.resume))
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def log_info(msg):
    dtstr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"{dtstr} {msg}")


def check_pred_accuracy_save_attr_map(val_loader, model, device, mask_reader, epoch=-1):
    numerator_sum = 0
    denominator_sum = 0
    model.eval()  # set the module in evaluation mode.
    for b_i, (x, y, ids) in enumerate(val_loader):
        x = x.to(device)
        y = y.to(device)
        denominator_sum += x.shape[0]
        x.requires_grad = True
        scores = model({0: x, 1: 0})
        _check_attr_map(b_i, scores, x, y, ids, mask_reader)
        preds = torch.argmax(scores, dim=1)
        numerator_sum += (preds == y).sum().item()
    # for
    accu = float(numerator_sum) / denominator_sum
    log_info(f"===> E{epoch:03d} Got {numerator_sum}/{denominator_sum} with accu {accu:.4f}")
    model.train()  # set the module in training mode
    return accu


def _check_attr_map(b_i, scores, x, y, ids, mask_reader):
    top_scores = scores.gather(1, index=y.unsqueeze(1))
    grad = torch.autograd.grad(top_scores.mean(), x, retain_graph=True)[0]
    # x.shape    : [1, 3, 256, 256] # batch size is 1.
    # grad.shape : [1, 3, 256, 256]
    x = x.detach()
    attr = torch.clone(grad)
    a_dx = torch.mean(attr, dim=1)  # attribution. dx, means differential of x
    # attr.shape : [1, 3, 256, 256]
    # a_dx.shape : [1, 256, 256]
    _apply_value_to_attr(a_dx, attr)

    attrx = grad * x
    ax_dx = torch.mean(attrx, dim=1)  # attribution. dx, means differential of x
    _apply_value_to_attr(ax_dx, attrx)

    masks_tensor = _load_mask_img(x, ids, mask_reader)

    x_ = torch.cat((x, attr, attrx, masks_tensor), 0)

    f_path = f"datalist/batch_{b_i:03d}.jpg"
    log_info(f_path)
    b_size = x.size(0)
    torchvision.utils.save_image(x_, f_path, nrow=b_size)


def _apply_value_to_attr(val, attr, threshold=20, bg_color='white'):
    val[val < 0] = 0  # ignore the minus elements
    for i, a in enumerate(val):  # iterate each attribution map in the batch
        max_g = torch.max(a)  # now a shape is [256, 256]
        if max_g == 0: max_g = 1
        val[i] = a * 255 / max_g
    val[val > threshold] = 255
    val[val <= threshold] = 0
    if bg_color == 'white':
        attr[attr != 255] = 255  # clear to 255
        for i, a in enumerate(attr):  # for each image in the batch
            attr[i, 1] = 255 - val[i]
            attr[i, 2] = 255 - val[i]
    elif bg_color == 'black':
        attr[attr != 0] = 0  # clear to 0
        for i, a in enumerate(attr):  # for each image in the batch
            attr[i, 0] = val[i]
    else:
        raise ValueError(f"Not supported bg_colr: {bg_color}.")
# def _apply_value_to_attr


def _load_mask_img(x, ids, mask_reader):
    masks_tensor = torch.zeros_like(x)
    for i, img_id in enumerate(ids):
        mask = mask_reader.get_image(img_id)
        if mask is not None:
            masks_tensor[i] = mask
    # for
    return masks_tensor


def check_pred_accuracy(val_loader, model, device, epoch):
    numerator_sum = 0
    denominator_sum = 0
    model.eval()  # set the module in evaluation mode. equivalent with model2.train(False)
    with torch.no_grad():  # context-manager that disable gradient calculation
        for x, y, _ in val_loader:
            x = x.to(device)
            y = y.to(device)
            denominator_sum += x.shape[0]
            scores = model({0: x, 1: 0})
            preds = torch.argmax(scores, dim=1)
            numerator_sum += (preds == y).sum().item()
        # for
    # with
    accu = float(numerator_sum) / denominator_sum
    log_info(f"===> E{epoch:03d} Got {numerator_sum}/{denominator_sum} with accu {accu:.4f}")
    model.train()  # set the module in training mode
    return accu
