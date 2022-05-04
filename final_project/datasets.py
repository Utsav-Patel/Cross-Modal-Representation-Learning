import os
import json
import lmdb

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from io import BytesIO


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
        lmdb_file=f'/common/users/upp10/cs536/Recipe1M/Recipe1M.lmdb',
        part='', transform=None, resolution=256):

        assert part in ['', 'train', 'val', 'test'], "part has to be in ['', 'train', 'val', 'test']"

        if transform is None:
            if part == 'val':
                self.transform = get_train_transforms([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            else:
                self.transform = get_val_transforms([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

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