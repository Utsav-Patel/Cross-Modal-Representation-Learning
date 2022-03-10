import torch
import os
import numpy as np
import argparse

from models import BertEncoder
from dataset import RecipeTextDataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from utils import pickler
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model parameters')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lmdb_file', default='/common/home/as3503/as3503/courses/cs536/dataset/Recipe1M.lmdb')
    parser.add_argument('--save_dir', default='/common/home/as3503/as3503/courses/cs536/embeddings/')
    args = parser.parse_args()
    device = args.device
    model = BertEncoder().to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")



    # for part in ('train', 'test', 'val'):
        # for recipe_part in ('title', 'ingredients', 'instructions'):
    for part in ('test', 'val'):
        for recipe_part in ('title', 'ingredients'):    
            dataset = RecipeTextDataset(part=part, recipe_part=recipe_part)
            dataloader = DataLoader(dataset, batch_size=args.batch_size)
            save_path = os.path.join(args.save_dir, f'text_{recipe_part}_{part}_embeddings.pkl')
            with torch.no_grad():
                for recipe in tqdm(dataloader):
                    model_inputs = tokenizer(recipe, return_tensors='pt', truncation=True, padding=True).to(device)
                    embeddings = model(**model_inputs)
                    pickler(obj=embeddings.cpu().numpy(), fpath=save_path, mode='ab')










