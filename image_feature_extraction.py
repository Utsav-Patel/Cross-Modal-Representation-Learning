import torch
import os
import numpy as np
import argparse

from models import ResnetEncoder
from dataset import Recipe1MDataset
from torch.utils.data import DataLoader
from utils import pickler
from tqdm import tqdm
from torchvision import transforms
from PIL import Image


def get_train_transforms(mean, std):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.ColorJitter(hue=.05, saturation=.05),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20, resample=Image.BILINEAR),
        transforms.RandomCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def get_val_transforms(mean, std):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model parameters')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lmdb_file', default='/common/home/as3503/as3503/courses/cs536/dataset/Recipe1M.lmdb')
    parser.add_argument('--save_dir', default='/common/home/as3503/as3503/courses/cs536/embeddings/')
    args = parser.parse_args()
    device = args.device
    model = ResnetEncoder().to(device)
    model.eval()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = get_train_transforms(mean, std)
    val_transform = get_val_transforms(mean, std)
    transform_parts = {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform
    }



    for part in ('train', 'test', 'val'):
        dataset = Recipe1MDataset(part=part, transform=transform_parts[part])
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        save_path = os.path.join(args.save_dir, f'image_{part}_embeddings.pkl')
        with torch.no_grad():
            for _, images in tqdm(dataloader):
                embeddings = model(images.to(device))
                pickler(obj=embeddings.cpu().numpy(), fpath=save_path, mode='ab')





 




