import torch.optim
import torchvision.utils

from network import resnet
from utils_art.util_args import get_args
from utils_art.util_loader import data_loader
from utils_art.util_loader import MaskReader
from utils_art.util_acc import *
from utils_art.util_eval import *
from utils_art.util import *
from art_attack import *
import loss_art
from utils_art.integrated_gradients import *

best_epoch = 0
best_acc1 = 0
best_loc1 = 0
loc1_at_best_acc1 = 0
acc1_at_best_loc1 = 0


def main(args):
    global best_acc1, best_loc1, best_epoch, loc1_at_best_acc1, acc1_at_best_loc1

    mean = torch.tensor([.485, .456, .406])
    stdd = torch.tensor([.229, .224, .225])
    new_stdd = stdd[..., None, None]
    new_mean = mean[..., None, None]

    def normalize(x):
        return (x - new_mean.cuda(args.device)) / new_stdd.cuda(args.device)

    log_folder = os.path.join('train_log', args.name)
    if not os.path.isdir(log_folder):
        print(f"make dirs: {log_folder}")
        os.makedirs(log_folder)

    num_classes = args.num_classes
    model = resnet.wide_resnet50_2(pretrained=True, beta_value=args.beta, num_classes=num_classes)
    # model = resnet.resnet50(pretrained=True, num_classes=num_classes)
    # model = backbone.WideResNet28_10(flatten=True, num_classes=num_classes)

    # if len(args.gpu_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    model = model.to(args.device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()  # .cuda(args.device)
    param_features = []
    param_classifiers = []
    for name, parameter in model.named_parameters():
        if 'layer4.' in name or 'fc.' in name:
            param_classifiers.append(parameter)
        else:
            param_features.append(parameter)
    optimizer = torch.optim.SGD(
        [
            {'params': param_features,    'lr': args.lr},
            {'params': param_classifiers, 'lr': args.lr * args.lr_ratio}
        ],
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nest
    )

    # optionally resume from a checkpoint
    if args.resume:
        print(f"resume: {args.resume}")
        model, optimizer = load_model(model, optimizer, args)

    if args.enable_exemplar:
        config = {
                'epsilon': 2.0 / 255.0,
                'num_steps': 1,  # was 3
                'step_size': 1.5 / 255.0,
                'k_top': 1500,  # was 15000
                'img_size': 224,
                'n_classes': num_classes
                }
        print(f"AttackSaliency: {config}")
        attack = AttackSaliency(config, normalize, args.device)
    else:
        attack = None

    if args.grad == 'ig':
        print(f"grad_fn=integrated_gradients")
        grad_fn = integrated_gradients
    elif args.grad == 'g':
        print(f"grad_fn=normal_gradients")
        grad_fn = normal_gradients
    else:
        raise ValueError(f"invalid args.grad:{args.grad}")

    train_loader, val_loader = data_loader(args, num_classes)
    mask_reader = MaskReader(args)
    check_pred_accuracy(val_loader, model, args.device, -1)
    # check_pred_accuracy_save_attr_map(val_loader, model, args.device, mask_reader)

    for e in range(args.start_epoch, args.epochs):
        log_info("Start Epoch %d/%d ..." % (e, args.epochs))
        adjust_learning_rate(optimizer, e, args)

        train(train_loader, model, criterion, optimizer, e, args, attack, num_classes, grad_fn)

        save_freq = 10
        if e % save_freq == 0 or e == args.epochs-1:
            saving_dir = os.path.join(log_folder)
            state_dict = {
                'epoch': e + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(state_dict, False, saving_dir, filename=f"checkpoint_{e}.pth.tar")
        # if
        check_pred_accuracy(val_loader, model, args.device, e)
    # for


def train(train_loader, model, criterion, optimizer, epoch, args, attack, num_classes, grad_fn):
    b_cnt = len(train_loader)
    for i, (x, y, _) in enumerate(train_loader):
        y = Variable(y.cuda(args.device))
        x = Variable(x.cuda(args.device))

        if args.enable_exemplar:
            model.eval()  # This switch is very important. else, model will mess up and accuracy not increase.
            xadv = attack.topk_sal_attack(model, x, y, 0., 1., grad_fn)
            exemplar_loss = calc_exemplar_loss(x, xadv, y, model, num_classes, grad_fn)
        else:
            exemplar_loss = 0

        model.train()  # switch to train mode
        scores = model({0: x, 1: 0})  # changed from xadv to x

        optimizer.zero_grad()
        loss = criterion(scores, y)
        if args.enable_exemplar:
            total_loss = loss + 0.5*exemplar_loss
            total_loss.backward()
            msg = f"E{epoch}.B{i:03d}/{b_cnt} loss:{loss:.4f}; exemp:{exemplar_loss:.4f}; total:{total_loss:.4f}"
        else:
            loss.backward()
            msg = f"E{epoch}.B{i:03d}/{b_cnt} loss:{loss:.4f}"
        optimizer.step()
        log_info(msg)
    # for


def calc_exemplar_loss(x, xadv, y, model, num_classes, grad_fn):
    x_ = torch.cat([x, xadv], 0)

    y_ = y.repeat(2)
    a_ = torch.cat((torch.arange(0, x.size(0)), torch.arange(0, x.size(0))), 0).long()
    a_ = Variable(a_).cuda(args.device)

    x_.requires_grad = True
    scores = model({0: x_, 1: 1})

    top_scores = scores.gather(1, index=y_.unsqueeze(1))
    tmp_arr = np.array([[k for k in range(num_classes) if k != y_[j]] for j in range(y_.size(0))])
    non_target_indices = torch.from_numpy(tmp_arr).cuda(args.device)

    btm_scores = scores.gather(1, index=non_target_indices)
    btm_scores = btm_scores.max(dim=1)[0]

    # https://pytorch.org/docs/stable/generated/torch.autograd.grad.html
    # torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None,
    #   create_graph=False, only_inputs=True, allow_unused=False)
    # Computes and returns the sum of gradients of outputs with respect to the inputs.
    # g1 = torch.autograd.grad(top_scores.mean(), x_, retain_graph=True, create_graph=True)[0]
    # g2 = torch.autograd.grad(btm_scores.mean(), x_, retain_graph=True, create_graph=True)[0]

    g1 = grad_fn(top_scores.mean(), x_, y_, model, args.device)
    g2 = grad_fn(btm_scores.mean(), x_, y_, model, args.device)

    g1_adp = torch.mean(torch.abs(g1), 1)
    g2_adp = torch.mean(torch.abs(g2), 1)
    x_conv = torch.mean(torch.abs(x_.detach()), 1)

    x_.requires_grad = False
    batch_size = x_.size(0)
    x_r = x_conv.reshape(batch_size, -1)
    g1_r = g1_adp.reshape(batch_size, -1)
    g2_r = g2_adp.reshape(batch_size, -1)
    exemplar_loss = loss_art.exemplar_loss_fn(x_r, g1_r, g2_r, y_, a_)
    return exemplar_loss


if __name__ == '__main__':
    print("process id:", os.getpid())
    args = get_args()
    args.epochs = 10
    # args.resume = 'runs/train_log/test_case/checkpoint_98.pth.tar'
    # args.resume = 'runs_ig/train_log/test_case/checkpoint_99.pth.tar'
    # args.resume = 'runs_pure_classify/train_log/test_case/checkpoint_99.pth.tar'

    if args.datadir is None: args.datadir = 'data/jpg'
    if args.datamaskdir is None: args.datamaskdir = args.datadir.replace('/jpg', '/trimaps')
    args.train_list = [(1, 71)]   # 1360 images. image ID start from 1.
    args.valdt_list = [(71, 81)]
    args.beta = 1
    args.num_classes = 17
    if args.gpu_ids is None: args.gpu_ids = [0]
    args.device = f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() and args.gpu_ids else "cpu"

    if args.enable_exemplar is None: args.enable_exemplar = True

    # For one GPU: if enable_exemplar=False, then batch size can be 128; but if True, can have 16 only.
    # the following code eat much memory, causing the 16 size issue:
    # g1 = torch.autograd.grad(top_scores.mean(), x_, retain_graph=True, create_graph=True)[0]
    # g2 = torch.autograd.grad(btm_scores.mean(), x_, retain_graph=True, create_graph=True)[0]
    if args.batch_size is None: args.batch_size = 8 * len(args.gpu_ids)
    if args.epochs is None: args.epochs = 100
    if args.lr is None: args.lr = 5e-5
    if args.lr_decay is None: args.lr_decay = 10
    if args.lr_decay_ratio is None: args.lr_decay_ratio = 1

    if args.grad is None: args.grad = 'g'
    print(args)
    print(f"batch_size:{args.batch_size}; gpu_ids:{args.gpu_ids}; enable_exemplar:{args.enable_exemplar}")
    print(f"lr:{args.lr}; lr_ratio:{args.lr_ratio}; lr_decay:{args.lr_decay}; lr_decay_ratio:{args.lr_decay_ratio}; ")
    print(f"grad:{args.grad}")
    print("")

    main(args)  # ======================================================
