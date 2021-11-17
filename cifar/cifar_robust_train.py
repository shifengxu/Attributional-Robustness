import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import os
import backbone
from io_utils import model_dict, parse_args, get_resume_file ,get_assigned_file
from pgd import *
from saliency_pgd import *
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision
from PIL import Image
import numpy as np
import loss_art
from torch.utils.data import DataLoader
import datetime


def lr_lambda(steps):
    if steps < 50:
        return 1.0
    elif steps in range(50, 80):
        return 0.1
    elif steps in range(80, 150):
        return 0.01
    elif steps in range(150, 200):
        return 0.005
    else:
        return 0.005


def saliency_train(train_loader, val_loader, model, start_epoch, stop_epoch, params):
    print("saliency_train() started...")
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    stdd = torch.tensor([0.2023, 0.1994, 0.2010])
    new_stdd = stdd[..., None, None].to(params.device)
    new_mean = mean[..., None, None].to(params.device)
    def normalize(m): return (m - new_mean) / new_stdd

    # config = {'epsilon': 8.0 / 255.0, 'num_steps': 3, 'step_size': 5. / 255.0, 'k_top': 1000}
    config = {'epsilon': 8.0 / 255.0, 'num_steps': 1, 'step_size': 10. / 255.0, 'k_top': 1000}
    print("ART config:", config)
    attack_art = AttackSaliency(config, normalize, device=params.device)
    
    config1 = {'epsilon': 8.0 / 255.0, 'num_steps': 40, 'step_size': 2.0 / 255.0}
    print("PGD config:", config1)
    attack_pgd = AttackPGD(config1, normalize)
    
    ce_loss = nn.CrossEntropyLoss()
    # ranking_loss = nn.SoftMarginLoss()
    # softmax_F = nn.Softmax(dim=1)
    
    params_optimize = list(model.parameters())
    optimizer = torch.optim.SGD(params_optimize, lr=0.1, nesterov=True, momentum=0.9, weight_decay=2e-4)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"start_epoch:{start_epoch}; stop_epoch:{stop_epoch}")
    for initial_step in range(start_epoch):
        if initial_step == 0:
            print("restoring step size scheduler")
        optimizer.zero_grad()
        optimizer.step()
        scheduler.step()

    for epoch in range(start_epoch, stop_epoch):
        model.train()
        batch_cnt = len(train_loader)
        for i, (x, y) in enumerate(train_loader):
            y = Variable(y).to(params.device)
            x = Variable(x).to(params.device)
            train_epoch_batch(epoch, i, batch_cnt, model, optimizer, ce_loss, attack_art, normalize, x, y)
        print("lr:", optimizer.state_dict()['param_groups'][0]['lr'])
        
        scheduler.step()
        model.eval()
        correct = correct_adv = total = total_adv = 0
        
        for i, (x, y) in enumerate(val_loader):
            x = x.to(params.device)
            y = y.to(params.device)
            _, scores = model.forward({0: normalize(x), 1: 0})
            p1 = torch.argmax(scores, 1)
            correct += (p1 == y).sum().item()
            total += p1.size(0)
            if epoch % 10 == 0:
                optimizer.zero_grad()
                xadv = attack_pgd.attack(model, x, y, 0., 1.)
                _, scores_adv = model.forward({0: normalize(xadv), 1: 0})
                p1_adv = torch.argmax(scores_adv, 1)
                correct_adv += (p1_adv == y).sum().item()
                total_adv += p1_adv.size(0)
        # for
        accu = float(correct)/total
        msg = f"E{epoch}: Accu={accu:6f}"
        if total_adv > 0:
            accu_adv = float(correct_adv) / total_adv
            msg += f", AdvAccu={accu_adv:6f}"
        print(msg)

        if epoch % 10 == 0 or epoch == stop_epoch - 1:
            outfile = os.path.join(params.checkpoint_dir, '{:03d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
    return model


# train on a single epoch, a single batch.
# Please see the "Algorithm 1" in the page 7 of the original paper
def train_epoch_batch(epoch, b_idx, b_cnt, model, optimizer, ce_loss, attack, normalize, x, y):
    # optimizer.zero_grad()
    # this function is time-consuming.
    # if batch_size=75, attack_art.num_steps=3, and single GPU, it may take 10 seconds.
    xadv = attack.topk_alignment_saliency_attack(model, x, y, 0., 1.)  # adversarial image
    x_ = normalize(torch.cat([x, xadv], 0))
    y_ = y.repeat(2)
    a_ = torch.cat((torch.arange(0, x.size(0)), torch.arange(0, x.size(0))), 0).long().to(params.device)

    x_.requires_grad = True  # if gradients need to be computed for this Tensor
    _, scores = model({0: x_, 1: 1})

    # https://pytorch.org/docs/stable/generated/torch.gather.html
    # torch.gather(input, dim, index, *, sparse_grad=False, out=None) -> Tensor.
    # Gathers values along an axis specified by dim.
    # For a 3-D tensor the output is specified by:
    #   given x = index[i][j][k]
    #   output[i][j][k] = input[x][j][k] # if dim == 0
    #   output[i][j][k] = input[i][x][k] # if dim == 1
    #   output[i][j][k] = input[i][j][x] # if dim == 2
    # The output has the same dimension as index;
    # For an element of output tensor, its value is from input, but special subscript
    #
    # Assuming batch size is 75, the scores shape is [150, 10], and top_scores will be [150, 1]
    # And other_scores shape is [150, 9].
    top_scores = scores.gather(1, index=y_.unsqueeze(1))
    non_target_indices = np.array([[k for k in range(10) if k != y_[j]] for j in range(y_.size(0))])
    other_scores = scores.gather(1, index=torch.tensor(non_target_indices).to(params.device))
    # https://pytorch.org/docs/stable/generated/torch.max.html
    #  torch.max(input, dim, keepdim=False, *, out=None)
    #  Returns a namedtuple (values, indices) where values is the maximum
    #  value of each row of the input tensor in the given dimension dim
    snd_scores = other_scores.max(dim=1)[0]  # second scores

    # https://pytorch.org/docs/stable/generated/torch.autograd.grad.html
    # torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None,
    #   create_graph=False, only_inputs=True, allow_unused=False)
    # Computes and returns the sum of gradients of outputs with respect to the inputs.
    g1 = torch.autograd.grad(top_scores.mean(), x_, retain_graph=True, create_graph=True)[0]
    g2 = torch.autograd.grad(snd_scores.mean(), x_, retain_graph=True, create_graph=True)[0]

    b_size = x_.size(0)
    x_.requires_grad = False
    x_r, g1_r, g2_r = x_.reshape(b_size, -1), g1.reshape(b_size, -1), g2.reshape(b_size, -1)
    exemplar_loss = loss_art.exemplar_loss_fn(x_r, g1_r, g2_r, y_, a_)
    _, scores_adv = model({0: normalize(xadv), 1: 0})
    # scores_adv looks like this:
    # [[ 1.8173e-01, -2.7396e-02,  1.5750e-01, -2.3128e-01, -3.4252e-02,
    #    1.2717e-01, -2.2079e-01, -1.1641e-02,  2.4980e-01,  3.9312e-01], , , ,

    # CrossEntropyLoss: The input is expected to contain raw, unnormalized scores for each class
    # On single GPU, the following loss and BP take about 4 seconds
    loss = ce_loss(scores_adv, y)
    optimizer.zero_grad()
    total_loss = loss + 0.5 * exemplar_loss  # 0.5: lambda for coefficient of Loss_attr
    total_loss.backward()
    optimizer.step()

    if b_idx % 2 == 0:
        dtstr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f_loss = loss.data.item()
        f_e_loss = exemplar_loss.data.item()
        print(f"{dtstr} E{epoch:02d}.B{b_idx:03d}/{b_cnt:03d} Loss:{f_loss:5f}, SalLoss:{f_e_loss:5f}")


class TransformsC10:

    def __init__(self):
        
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25,.25,.25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()
        ])
        

    def __call__(self, inp):
        out1 = self.train_transform(inp)
        return out1


def main(params):
    train_transform = TransformsC10()  # Data Loading
    test_transform = train_transform.test_transform

    dset_dir = './dataset/'
    print(f"CIFAR10: {dset.CIFAR10.url}")
    print(f"Local: {os.path.join(os.getcwd(), dset_dir, dset.CIFAR10.base_folder)}")
    print("Load training data...")
    trainset = dset.CIFAR10(root=dset_dir, train=True, download=True, transform=train_transform)
    print("Load test data...")
    testset = dset.CIFAR10(root=dset_dir, train=False, download=True, transform=test_transform)
    trainloader = DataLoader(trainset, batch_size=params.bs, shuffle=True, num_workers=8)
    testloader = DataLoader(testset, batch_size=params.bs, shuffle=False, num_workers=8)

    if not os.path.isdir(params.checkpoint_dir):
        print(f"make dirs: {params.checkpoint_dir}")
        os.makedirs(params.checkpoint_dir)

    start_epoch = 0
    model = backbone.WideResNet28_10(flatten=True, beta_value=params.hyper_beta)
    if len(params.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=params.gpu_ids)
    if params.resume:
        print("resuming", params.checkpoint_dir)
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            print("resume_file", resume_file)
            state_dict = torch.load(resume_file)
            start_epoch = state_dict['epoch'] + 1
            model.load_state_dict(state_dict['state'])

    model.to(params.device)
    # model = model.cuda()
    saliency_train(trainloader, testloader,  model, start_epoch, 101, params)


if __name__ == '__main__':
    print("Process id", os.getpid())
    params = parse_args('train')
    params.model = 'WideResNet28_10'
    params.dataset = 'cifar'
    params.method = 'art'
    params.checkpoint_dir = './checkpoints/%s/%s_%s_%s' % (params.dataset, params.model, params.method, 'cifar')
    params.gpu_ids = [2]
    params.device = f"cuda:{params.gpu_ids[0]}" if torch.cuda.is_available() else 'cpu'
    params.bs = 75 * len(params.gpu_ids)
    params.hyper_beta = 50.0   # hyperparameter: beta value in softplus
    print(params)
    main(params)
