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
    parser.add_argument('--save_dir', default='/common/home/as3503/as3503/courses/cs536/embeddings_all/')
    parser.add_argument('--means', default=False, type=bool)
    parser.add_argument('--part', default=['train', 'test', 'val'], nargs='+')
    parser.add_argument('--recipe_part', default=['title', 'ingredients', 'instructions', 'all'], nargs='+')
    args = parser.parse_args()
    device = args.device
    model = BertEncoder(means=args.means).to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    os.makedirs(args.save_dir, exist_ok=True)


    for part in args.part:
        for recipe_part in args.recipe_part:
            dataset = RecipeTextDataset(part=part, recipe_part=recipe_part)
            dataloader = DataLoader(dataset, batch_size=args.batch_size)
            save_path = os.path.join(args.save_dir, f'text_{recipe_part}_{part}_embeddings.pkl')
            with torch.no_grad():
                for recipe in tqdm(dataloader):
                    model_inputs = tokenizer(recipe, return_tensors='pt', truncation=True, padding=True).to(device)
                    embeddings = model(**model_inputs)
                    pickler(obj=embeddings.cpu().numpy(), fpath=save_path, mode='ab')










