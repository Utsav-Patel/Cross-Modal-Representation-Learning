import torch
import torchvision
import os
import numpy as np
import argparse
import time
import pdb
import sys
import utils
import torch.multiprocessing

from torch import nn
from tqdm import tqdm
from types import SimpleNamespace
from transformers import BertTokenizer
from models import Embedder
from dataset import EmbeddingDataset
from datetime import datetime
from triplet_loss import global_loss, TripletLoss

# https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
# torch.multiprocessing.set_sharing_strategy('file_system')


def create_retrieval_model(args, device='cuda', image_encoder='resnet'):
    txt_encoder = Embedder(args.latent_size).to(device)
    img_encoder = Embedder(args.latent_size).to(device)
    optimizer = torch.optim.Adam(
        [
            {'params': txt_encoder.parameters()},
            {'params': img_encoder.parameters()},
        ], 
        lr=args.lr
    )
    return txt_encoder, img_encoder, optimizer

def load_retrieval_model(ckpt_path, device='cuda', image_encoder='resnet'):
    print(f'load retrieval model from {ckpt_path}')
    ckpt = torch.load(ckpt_path)
    
    ckpt_args = ckpt['args']
    if 'lr' not in ckpt_args.__dict__:
        ckpt_args.lr = 0.0001

    txt_encoder, img_encoder, optimizer = create_retrieval_model(ckpt_args, device, image_encoder)
    epoch_start = ckpt['epoch']+1
    txt_encoder.load_state_dict(ckpt['txt_encoder'])
    img_encoder.load_state_dict(ckpt['img_encoder'])
    optimizer.load_state_dict(ckpt['optimizer'])
    
    return ckpt_args, epoch_start, txt_encoder, img_encoder, optimizer

def save_retrieval_model(args, txt_module, img_module, optimizer, epoch, ckpt_path):
    print(f'save retrieval model to {ckpt_path}')
    ckpt = {
        'args': args,
        'epoch': epoch,
        'txt_encoder': txt_module.state_dict(),
        'img_encoder': img_module.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(ckpt, ckpt_path)

def train(
    args, train_loader, val_loader, 
    txt_encoder, img_encoder, optimizer, 
    n_train, n_val):
    
    save_dir = args.save_dir
    device = args.device
    if device == 'cuda':
        txt_module = txt_encoder.module
        img_module = img_encoder.module
    else:
        txt_module = txt_encoder
        img_module = img_encoder
    
    ckpt_path = os.path.join(save_dir, f'model.pt')
    triplet_loss = TripletLoss(args.margin)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
    min_val_loss = float('inf')
    for epoch in range(args.epochs):
        time.sleep(0.5)
        print('#' * 40)
        print(f'Epoch = {epoch}')
        
        print('train')
        txt_encoder.train()
        img_encoder.train()
        train_loss = 0.0
        time.sleep(0.5)
        iteration = 1
        for img, txt in tqdm(train_loader):
            txt_output = txt_encoder(txt.to(device))
            img_output = img_encoder(img.to(device))

            bs = img.shape[0]
            label = list(range(0, bs))
            label.extend(label)
            label = np.array(label)
            label = torch.tensor(label).long().to(device)
            loss = global_loss(triplet_loss, torch.cat((img_output, txt_output)), label)[0]
            train_loss += loss.item() * bs
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iteration += 1

        train_loss /= n_train
        
        time.sleep(0.5)
        print('val')
        txt_encoder.eval()
        img_encoder.eval()
        txt_outputs = []
        img_outputs = []
        val_loss = 0.0

        time.sleep(0.5)
        with torch.no_grad():
            iteration = 0
            for txt, img in tqdm(val_loader):

                txt_output = txt_encoder(txt.to(device))
                img_output = img_encoder(img.to(device))

                txt_outputs.append(txt_output.detach().cpu())
                img_outputs.append(img_output.detach().cpu())

                bs = img.shape[0]
                label = list(range(0, bs))
                label.extend(label)
                label = np.array(label)
                label = torch.tensor(label).long().to(device)
                loss = global_loss(triplet_loss, torch.cat((img_output, txt_output)), label)[0]
                val_loss += loss.item() * bs

                iteration += 1

        val_loss /= n_val
        txt_outputs = torch.cat(txt_outputs, dim=0).numpy()
        img_outputs = torch.cat(img_outputs, dim=0).numpy()
        retrieved_range = min(txt_outputs.shape[0], args.retrieved_range)
        medR, medR_std, recalls = utils.rank(
            txt_outputs, img_outputs, epoch=epoch, retrieved_type=args.retrieved_type, 
            retrieved_range=retrieved_range, verbose=True)

        ckpt_path = os.path.join(save_dir, f'model.pt')
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            print(f'save to {ckpt_path}')
            ckpt = {
                'args': args,
                'epoch': epoch,
                'txt_encoder': txt_module.state_dict(),
                'img_encoder': img_module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss
            }
            torch.save(ckpt, ckpt_path)

        log = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'medR': medR,
            'medR_std': medR_std
        }
        for k, v in recalls.items():
            log["Recall"+str(k)] = v

        print(log)
        with open(os.path.join(save_dir, 'metrics.txt'), 'a') as f:
            f.write('\n' + str(log) + '\n')
        scheduler.step(recalls[1])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Triplet loss trainer')
    parser.add_argument('--device', default='cuda', type=str, choices=['cuda', 'cpu'])
    parser.add_argument('--seed', default=8, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--margin', default=0.3, type=float)
    parser.add_argument('--retrieved_type', default='image', choices=['recipe', 'image'])
    parser.add_argument('--retrieved_range', default=1000, type=int)
    parser.add_argument('--ckpt_path', default='')
    parser.add_argument('--save_dir', default='')
    parser.add_argument('--image_pkl_path', required=True)
    parser.add_argument('--text_pkl_path', default='')
    parser.add_argument('--part', default=None)
    parser.add_argument('--latent_size', type=int, required=True)
    
    # in debug mode
    parser.add_argument("--debug", type=utils.str2bool, default=False, help="in debug mode or not")
    args = parser.parse_args()
    if not args.save_dir:
        args.save_dir = f'latent_{args.latent_size}_{args.part}'

    if args.save_dir and not args.ckpt_path:
        args.ckpt_path = os.path.join(args.save_dir, 'model.pt')
        
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = args.device

    train_set = EmbeddingDataset(image_path=args.image_pkl_path, text_path=args.text_pkl_path, part=args.part)
    val_set = EmbeddingDataset(image_path=args.image_pkl_path.replace('train', 'val'), text_path=args.text_pkl_path.replace('train', 'val'), part=args.part)
    if args.debug:
        print('in DEBUG mode')
        train_set = torch.utils.data.Subset(train_set, indices=range(min(len(train_set), 1000)))
        val_set = torch.utils.data.Subset(train_set, indices=range(min(len(val_set), 1000)))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    
    if args.ckpt_path:
        print('Loading model')
        ckpt_args, epoch_start, txt_encoder, img_encoder, optimizer = load_retrieval_model(
                                                                                    args.ckpt_path, device
                                                                        )
    else:
        print('Creating model')
        txt_encoder, img_encoder, optimizer = create_retrieval_model(args, device)
        epoch_start = 0
    
    if device == 'cuda':
        txt_encoder = nn.DataParallel(txt_encoder)
        img_encoder = nn.DataParallel(img_encoder)
        
    os.makedirs(args.save_dir, exist_ok=True)
    train(
        args, train_loader, val_loader, 
        txt_encoder, img_encoder, optimizer, 
        len(train_set), len(val_set))