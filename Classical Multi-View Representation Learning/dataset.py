import torch
import os
import numpy as np
import json
import lmdb
import random

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from io import BytesIO


class Recipe1MDataset(Dataset):
    """
    Dataset class for Recipe1M
    Attributes:
    part: train/val/test
    transform: image transforms

    Returns:
    text/image pair
    """
    def __init__(
        self, 
        lmdb_file=f'/common/home/as3503/as3503/courses/cs536/dataset/Recipe1M.lmdb',
        part='', transform=None, resolution=256):

        assert part in ['', 'train', 'val', 'test'], "part has to be in ['', 'train', 'val', 'test']"
        assert transform!=None, 'transform can not be None!'

        self.transform = transform
        self.resolution = resolution

        dirname = os.path.dirname(lmdb_file)
        path = os.path.join(dirname, 'keys.json')

        with open(path, 'r') as f:
            self.keys = json.load(f)

        self.keys = [x for x in self.keys if x['with_image']]

        if part:
            self.keys = [x for x in self.keys if x['partition']==part]

        self.env = lmdb.open(
            lmdb_file,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', lmdb_file)

    def __len__(self):
        return len(self.keys)

    def _load_recipe(self, rcp):
        rcp_id = rcp['id']

        with self.env.begin(write=False) as txn:
            # print("Loading recipe")
            key = f'title-{rcp_id}'.encode('utf-8')
            title = txn.get(key).decode('utf-8')

            key = f'ingredients-{rcp_id}'.encode('utf-8')
            ingredients = txn.get(key).decode('utf-8')

            key = f'instructions-{rcp_id}'.encode('utf-8')
            instructions = txn.get(key).decode('utf-8')

            key = f'{self.resolution}-{rcp_id}'.encode('utf-8')
            img_bytes = txn.get(key)

        txt = '\n'.join([title, ingredients, instructions])

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)
        return txt, img

    def __getitem__(self, index):
        rcp_key = self.keys[index]
        txt, img = self._load_recipe(rcp_key)
        return txt, img


class RecipeTextDataset(Dataset):
    """
    Dataset class for Recipe1M
    Attributes:
    part: train/val/test
    recipe_part: ingredients/title/instructions/all
    transform: image transforms

    Returns:
    recipe text
    """
    def __init__(
        self, 
        lmdb_file=f'/common/home/as3503/as3503/courses/cs536/dataset/Recipe1M.lmdb',
        part='', recipe_part=''):

        assert part in ['', 'train', 'val', 'test'], "part has to be in ['', 'train', 'val', 'test']"

        dirname = os.path.dirname(lmdb_file)
        path = os.path.join(dirname, 'keys.json')

        with open(path, 'r') as f:
            self.keys = json.load(f)

        self.keys = [x for x in self.keys if x['with_image']]

        if part:
            self.keys = [x for x in self.keys if x['partition']==part]

        self.env = lmdb.open(
            lmdb_file,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.recipe_part = recipe_part
        if not self.env:
            raise IOError('Cannot open lmdb dataset', lmdb_file)

    def __len__(self):
        return len(self.keys)

    def _load_recipe(self, rcp):
        rcp_id = rcp['id']

        with self.env.begin(write=False) as txn:
            # print("Loading recipe")
            key = f'title-{rcp_id}'.encode('utf-8')
            title = txn.get(key).decode('utf-8')

            key = f'ingredients-{rcp_id}'.encode('utf-8')
            ingredients = txn.get(key).decode('utf-8')

            key = f'instructions-{rcp_id}'.encode('utf-8')
            instructions = txn.get(key).decode('utf-8')

        return {
            'title': title,
            'ingredients': ingredients,
            'instructions': instructions,
            'all': '\n'.join([title, ingredients, instructions])
        }

    def __getitem__(self, index):
        rcp_key = self.keys[index]
        txt = self._load_recipe(rcp_key)
        return txt[self.recipe_part]