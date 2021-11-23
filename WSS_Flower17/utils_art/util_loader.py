import os

from torchvision import transforms as T
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

from .util import log_info
from .dataset_flower17 import Flower17Dataset

import torch


def data_loader(args, num_classes):
    mean_vals = [0.485, 0.456, 0.406]
    stdd_vals = [0.229, 0.224, 0.225]

    tsfm_train = T.Compose([
        T.Resize((args.resize_size, args.resize_size)),
        T.RandomCrop(args.crop_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        # T.Normalize(mean_vals, stdd_vals),
    ])

    tsfm_valdt = T.Compose([
        T.Resize((args.resize_size, args.resize_size)),
        # T.CenterCrop(args.crop_size),
        T.ToTensor(),
        # T.Normalize(mean_vals, stdd_vals),
    ])

    img_train = Flower17Dataset(args.datadir, args.train_list, tsfm_train, num_classes=num_classes)
    img_valdt = Flower17Dataset(args.datadir, args.valdt_list, tsfm_valdt, num_classes=num_classes)

    train_loader = DataLoader(img_train, args.batch_size, shuffle=True, num_workers=args.workers)
    valdt_loader = DataLoader(img_valdt, args.batch_size, shuffle=False, num_workers=args.workers)
    return train_loader, valdt_loader


class MaskReader:
    def __init__(self, args):
        self.transform = T.Compose([T.Resize((args.resize_size, args.resize_size))])
        self.datamaskdir = args.datamaskdir

    def get_image(self, img_id):
        mf_path = os.path.join(self.datamaskdir, f"image_{img_id:04d}.png")  # mask file path
        if not os.path.isfile(mf_path):
            log_info(f"file not exist: {mf_path}")
            return None
        image = Image.open(mf_path).convert('RGB')  # image.size : (666, 500) may other values
        image = self.transform(image)               # image.size : (256, 256)
        narr = np.asarray(image)                    # narr.shape : (256, 256, 3)
        narr2 = np.transpose(narr, (2, 0, 1))       # narr2.shape: (3, 256, 256)
        t2 = torch.tensor(narr2)
        return t2
# class
