# -*- coding: utf-8 -*-

import os
import argparse
import PIL.Image
from tqdm import tqdm
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import architecture as models
from utils.util import save_result_image
from utils.util_loc import get_cam_target_class, resize_threshold_cam


IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
IMAGE_STD_VALUE = [0.229, 0.224, 0.225]


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--log-folder', type=str, default='train_log')
    parser.add_argument('--data', help='path to dataset')
    parser.add_argument('--arch', choices=model_names)
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

    parser.add_argument('--name', type=str, default='test_case')

    parser.add_argument('--dataset', type=str, default='PASCAL', )
    parser.add_argument('--test-list', type=str, default='./datalist/PascalVOC/val.txt')

    parser.add_argument('--resize-size', type=int, default=321, help='input resize size')
    args = parser.parse_args()

    return args



def main():
    args = get_args()
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    num_classes = 20

    # Select Model & Method
    model = models.__dict__[args.arch](False, num_classes=num_classes)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # optionally resume from a checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume)['state_dict']
        model.load_state_dict(checkpoint, strict=True)

    test_loader = data_loader(args)
    evaluate_test(test_loader, model, args)


def evaluate_test(val_loader, model, args):

    model.eval()
    with torch.no_grad():
        for i, (images, image_id, image_sizes) in enumerate(tqdm(val_loader, desc='Evaluate')):
            images = images.cuda(args.gpu, non_blocking=True)

            output = model(images)
            cam = get_cam_target_class(model)
            cam = cam.cpu().numpy().transpose(0, 2, 3, 1)

            for j in range(cam.shape[0]):
                cam_ = resize_threshold_cam(cam[j],
                                            size=(image_sizes[j][0].item(), image_sizes[j][1].item()),
                                            thresh=0.3)
                cam_max = np.argmax(cam_, axis=2)

                save_result_image('final_map', cam_max, image_id[j], args)


def data_loader(args):

    # transforms for validation dataset
    transforms_val = transforms.Compose([
        transforms.Resize((args.resize_size, args.resize_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN_VALUE, IMAGE_STD_VALUE),
    ])
    test_loader = DataLoader(
        VOCTestDataset(root=args.data, datalist=args.test_list, transform=transforms_val),
        batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    return test_loader


class VOCTestDataset(Dataset):

    def __init__(self, root=None, datalist=None, transform=None):
        self.root = root
        datalist = open(datalist).read().splitlines()
        self.image_names = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in datalist]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]

        image_path = os.path.join(self.root, "JPEGImages", name + '.jpg')
        img = PIL.Image.open(image_path).convert("RGB")
        img_size = img.size
        if self.transform:
            img = self.transform(img)

        return img, name, img_size


if __name__ == '__main__':
    main()













