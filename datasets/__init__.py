
import torch.utils.data
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

import datasets.transforms as T
from .data_loader import LPVADataset
from .data_loader_new import *

def make_transforms(args, image_set, is_onestage=False):
    if is_onestage:
        normalize = Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return normalize

    imsize = args.imsize

    return T.Compose([
            T.RandomResize([640,640]),
            T.ToTensor(),
            T.NormalizeAndPad(size=640, aug_translate=False)
        ])


def build_dataset(split, args):
    return LPVADataset(data_root=args.data_root,
                        split_root=args.split_root,
                        dataset=args.dataset,
                        split=split,
                        transform=make_transforms(args, split),
                        max_query_len=args.max_query_len)
