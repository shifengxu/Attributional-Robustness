from torchvision import transforms
from torch.utils.data import (Dataset, DataLoader)
from .dataset_cub import CUBClsDataset, CUBCamDataset
from PIL import Image
from albumentations.pytorch import ToTensorV2
import torch
import os
import numpy as np
import albumentations as A


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # PASCAL dataset has thumb.db file in dataset dir
        self.images = [fn for fn in os.listdir(image_dir) if ".jpg" in fn]
        print(f"Dataset: Find {len(self.images)} images in {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        if "pascal" in self.mask_dir:
            mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))
            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            mask[mask != 0] = 1.0
        else:
            mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally


def data_loader(args):
    # Albumentations ensures that the input image and the output mask will
    # receive the same set of augmentations with the same parameters.
    train_transform = A.Compose( # create an instance of Compose class.
        [ # define a list of augmentations
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize( # img = (img - mean * max_pixel_value) / (std * max_pixel_value)
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(), # it is a class. To convert image and mask to torch.Tensor
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    train_ds = CarvanaDataset(
        image_dir=args.train_img_dir,
        mask_dir=args.train_msk_dir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=6,    # how many subprocesses to use for data loading. default 0.
        pin_memory=True,  # the data loader will copy Tensors into CUDA pinned memory before return them.
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=args.val_img_dir,
        mask_dir=args.val_msk_dir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=6,
        pin_memory=True,
        shuffle=False,
    )

    return train_loader, val_loader





    if args.beta:
        tsfm_train = transforms.Compose([transforms.Resize((args.resize_size, args.resize_size)),
                                     transforms.RandomCrop(args.crop_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor()
                                     ])
        
    else:
        tsfm_train = transforms.Compose([transforms.Resize((args.resize_size, args.resize_size)),
                                     transforms.RandomCrop(args.crop_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)
                                     ])

    if args.tencrop == 'True':
        func_transforms = [transforms.Resize(args.resize_size),
                           transforms.TenCrop(args.crop_size),
                           transforms.Lambda(
                               lambda crops: torch.stack(
                                   [transforms.Normalize(mean_vals, std_vals)(transforms.ToTensor()(crop)) for crop in crops])),
                           ]
    else:
        func_transforms = []

        # print input_size, crop_size
        if args.resize_size == 0 or args.crop_size == 0:
            pass
        else:
            func_transforms.append(transforms.Resize((args.resize_size, args.resize_size)))
            func_transforms.append(transforms.CenterCrop(args.crop_size))

        func_transforms.append(transforms.ToTensor())
        if args.normalize:
            func_transforms.append(transforms.Normalize(mean_vals, std_vals))

    tsfm_test = transforms.Compose(func_transforms)

    img_train = CUBClsDataset(root=args.data, datalist=args.train_list, transform=tsfm_train)
    img_test = CUBCamDataset(root=args.data, datalist=args.test_list, transform=tsfm_test)

    if args.beta:  
        train_sampler = None
    else:
        if hasattr(args, 'distributed') and args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(img_train)
        else:
            train_sampler = None

    train_loader = DataLoader(img_train,
                              batch_size=args.batch_size,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=args.workers)

    val_loader = DataLoader(img_test,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.workers)

    return train_loader, val_loader, train_sampler
