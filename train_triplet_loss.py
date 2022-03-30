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
from models import ResnetEncoder, BertEncoder
from dataset import Recipe1MDataset
from datetime import datetime
from triplet_loss import global_loss, TripletLoss

# https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
torch.multiprocessing.set_sharing_strategy('file_system')


def create_retrieval_model(args, device='cuda', image_encoder='resnet'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    txt_encoder = BertEncoder().to(device)
    img_encoder = ResnetEncoder().to(device)
    optimizer = torch.optim.Adam(
        [
            {'params': txt_encoder.parameters()},
            {'params': img_encoder.parameters()},
        ], 
        lr=args.lr
    )
    return tokenizer, txt_encoder, img_encoder, optimizer

def load_retrieval_model(ckpt_path, device='cuda', image_encoder='resnet'):
    print(f'load retrieval model from {ckpt_path}')
    ckpt = torch.load(ckpt_path)
    
    ckpt_args = ckpt['args']
    if 'lr' not in ckpt_args.__dict__:
        ckpt_args.lr = 0.0001

    tokenizer, txt_encoder, img_encoder, optimizer = create_retrieval_model(ckpt_args, device, image_encoder)
    epoch_start = ckpt['epoch']+1
    txt_encoder.load_state_dict(ckpt['txt_encoder'])
    img_encoder.load_state_dict(ckpt['img_encoder'])
    optimizer.load_state_dict(ckpt['optimizer'])
    
    return ckpt_args, epoch_start, tokenizer, txt_encoder, img_encoder, optimizer

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

def compute_txt_feat(txt, tokenizer, txt_encoder, device='cuda'):
    txt_input = tokenizer(txt, truncation=True, padding=True, return_tensors="pt").to(device)
    txt_feat = txt_encoder(**txt_input)
    return txt_feat

def compute_img_feat(img, img_encoder, device='cuda'):
    img_input = img.to(device)
    img_feat = img_encoder(img_input)
    return img_feat

def train(
    args, train_loader, val_loader, 
    tokenizer, txt_encoder, img_encoder, optimizer, 
    n_train, n_val):
    
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    device = args.device
    if args.device == 'cuda':
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
        for txt, img in tqdm(train_loader):
            # print('Starting first iteration')
            txt_output = compute_txt_feat(txt, tokenizer, txt_encoder, device=device)
            img_output = compute_img_feat(img, img_encoder, device=device)

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

            if iteration % 500 == 0:
                ckpt = {
                    'args': args,
                    'epoch': epoch,
                    'txt_encoder': txt_module.state_dict(),
                    'img_encoder': img_module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                ckpt_path = os.path.join(save_dir, f'model_{epoch}_{iteration}.pt')
                torch.save(ckpt, ckpt_path)


            iteration += 1

        train_loss /= n_train
        
        time.sleep(0.5)
        print('val')
        txt_encoder.eval()
        img_encoder.eval()
        txt_outputs = []
        img_outputs = []
        val_loss = 0.0

        # show bad predictions
        captions = []
        img_inputs = []
        time.sleep(0.5)
        with torch.no_grad():
            iteration = 0
            for txt, img in tqdm(val_loader):
                # show bad predictions
                captions.extend([x[:50] for x in txt])
                img_inputs.append(img.detach().cpu())

                txt_output = compute_txt_feat(txt, tokenizer, txt_encoder, device=device)
                img_output = compute_img_feat(img, img_encoder, device=device)
                # print(img_output.shape)

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
        # print(args.retrieved_type)
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

        # show bad predictions
        img_inputs = torch.cat(img_inputs, dim=0) # [N, 3, 224, 224]
        scheduler.step(recalls[1])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='retrieval model parameters')
    parser.add_argument('--device', default='cuda', type=str, choices=['cuda', 'cpu'])
    parser.add_argument('--seed', default=8, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--margin', default=0.3, type=float)
    parser.add_argument('--lmdb_file', default='/common/home/as3503/as3503/courses/cs536/dataset/Recipe1M.lmdb')
    parser.add_argument('--retrieved_type', default='image', choices=['recipe', 'image'])
    parser.add_argument('--retrieved_range', default=1000, type=int)
    parser.add_argument('--ckpt_path', default='')
    # in debug mode
    parser.add_argument("--debug", type=utils.str2bool, default=False, help="in debug mode or not")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = args.device


    # Standard imagenet resnet mean and std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = utils.get_train_transforms(mean, std)
    val_transform = utils.get_val_transforms(mean, std)

    train_set = Recipe1MDataset(lmdb_file=args.lmdb_file, resolution=256, transform=train_transform, part='train')
    val_set = Recipe1MDataset(lmdb_file=args.lmdb_file, resolution=256, transform=val_transform, part='val')
    if args.debug:
        print('in DEBUG mode')
        train_set = torch.utils.data.Subset(train_set, indices=range(min(len(train_set), 1000)))
        val_set = torch.utils.data.Subset(train_set, indices=range(min(len(val_set), 1000)))


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=2*args.batch_size, num_workers=args.workers, shuffle=False)
            
    print(len(train_set), len(train_loader))
    print(len(val_set), len(val_loader))
    # pdb.set_trace()

    if args.ckpt_path:
        ckpt_args, epoch_start, tokenizer, txt_encoder, img_encoder, optimizer = load_retrieval_model(
                                                                                    args.ckpt_path, device
                                                                                 )
    else:
        tokenizer, txt_encoder, img_encoder, optimizer = create_retrieval_model(args, device)
        epoch_start = 0
    
    if device == 'cuda':
        txt_encoder = nn.DataParallel(txt_encoder)
        img_encoder = nn.DataParallel(img_encoder)
        
    dateTimeObj = datetime.now()
    time_stamp = dateTimeObj.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(os.path.dirname(__file__), 'runs', time_stamp)
    os.makedirs(save_dir, exist_ok=True)
    
    args.save_dir = save_dir

    train(
        args, train_loader, val_loader, tokenizer, 
        txt_encoder, img_encoder, optimizer, 
        len(train_set), len(val_set))