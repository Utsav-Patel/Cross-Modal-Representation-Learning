import torch
import numpy as np
import argparse
import os
import utils

from tqdm import tqdm
from torch import nn
from train_triplet_loss import load_retrieval_model
from dataset import EmbeddingDataset
from models import Embedder
from glob import glob

# # https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

@torch.no_grad()
def validate(args, val_loader, txt_encoder, img_encoder, device):
    print('val')
    txt_encoder.eval()
    img_encoder.eval()

    txt_outputs = []
    img_outputs = []
    for txt, img in tqdm(val_loader):
        txt_output = txt_encoder(txt.to(device))
        img_output = img_encoder(img.to(device))

        txt_outputs.append(txt_output.detach().cpu())
        img_outputs.append(img_output.detach().cpu())

    txt_outputs = torch.cat(txt_outputs, dim=0).numpy()
    img_outputs = torch.cat(img_outputs, dim=0).numpy()

    retrieved_range = min(txt_outputs.shape[0], args.retrieved_range)
    medR, _, recalls = utils.rank(
        txt_outputs, img_outputs, retrieved_type=args.retrieved_type, 
        retrieved_range=retrieved_range, verbose=True)
    
    return medR, recalls


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='retrieval model parameters')
    parser.add_argument('--seed', default=8, type=int)
    parser.add_argument('--device', default='cuda', type=str, choices=['cuda', 'cpu'])
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--retrieved_range', default=1000, type=int)
    parser.add_argument('--retrieved_type', default='recipe', choices=['recipe', 'image'])
    parser.add_argument('--pkl_path', default='/common/home/as3503/as3503/courses/cs536/dataset/prof_embeddings')
    parser.add_argument('--model_dir', default='/common/home/as3503/as3503/courses/cs536/final_project/Classical Multi-View Representation Learning/saved_models')
    parser.add_argument('--save_dir', default='/common/home/as3503/as3503/courses/cs536/final_project/Classical Multi-View Representation Learning/recipe_retrieval_metrics')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = args.device
    latent_dims = [64, 128, 256]
    parts = ['all', 'title', 'ingredients', 'instructions']

    for part in parts:
        text_pkl_path = os.path.join(args.pkl_path, f'{part}_embeddings_val.pkl')
        image_pkl_path = os.path.join(args.pkl_path, 'embeddings_val1.pkl')
        val_set = EmbeddingDataset(image_path=image_pkl_path, text_path=text_pkl_path, part=part)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=512, shuffle=False)

        for latent_size in latent_dims:
            print(f'Validating {part}, {latent_size}')
            dir_name = f'notebook_latent_{latent_size}_{part}'
            ckpt_path = os.path.join(args.model_dir, dir_name, 'model.pt')
            print('Loading model')
            ckpt_args, epoch_start, txt_encoder, img_encoder, optimizer = load_retrieval_model(ckpt_path, device)
            txt_encoder = nn.DataParallel(txt_encoder)
            img_encoder = nn.DataParallel(img_encoder)
            medR, recalls = validate(args, val_loader, txt_encoder, img_encoder, device)
            print('Saving metrics')
            torch.save({'medR': medR, 'recalls': recalls}, os.path.join(args.save_dir, f'{latent_size}_{part}_metrics.pth'))

        # Making space    
        for fpath in glob('./*.pkl'):
            os.remove(fpath)
