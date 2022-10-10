import torch
import os
import PIL
from typing import *

import torch
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class AddNoise:
    def __init__(self, sigma=0.5):
        self.sigma = sigma

    def __call__(self, x):
        noise = torch.randn_like(x) * self.sigma
        return x + noise

class DatasetWithInterval(torch.utils.data.Dataset):
    '''
    sampling data with interval from a given dataset 
    '''
    def __init__(self, dataset, interval):
        self.dataset = dataset
        self.interval = interval
    
    def __getitem__(self, index):
        return self.dataset[index * self.interval]
    
    def __len__(self):
        return len(self.dataset) // self.interval

def build_dataset_with_interval(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    print(dataset)
    dataset = DatasetWithInterval(dataset, args.sample_interval)
    print('With a interval of {}'.format(args.sample_interval))

    return dataset

def build_dataset(is_train, args):
    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    transform = build_transform(is_train, args)
    dataset = datasets.ImageFolder(root, transform=transform)
    print(dataset)

    return dataset


def build_transform(is_train, args):
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        # primary_tfl: reszie and clip
        # secondary: randaug
        # final_tfl: normalization
        primary_tfl, secondary_tfl, _ = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa if args.aa != "None" else None,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            separate=True
        )
        # noise augmentation
        transform = transforms.Compose([
            primary_tfl,
            secondary_tfl,
            transforms.ToTensor(),
            # transforms.Lambda(AddNoise(args.sigma)),
            # transforms.Normalize(mean, std),
        ])
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    # t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
