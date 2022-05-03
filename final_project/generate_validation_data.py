import torch
from tqdm import tqdm
import pickle

from datasets import Recipe1MDataset
from models import TextEncoder, ImageEncoder
from helper import freeze_params
from torch.utils.data import DataLoader
from transformers import BertTokenizer


if __name__ == '__main__':

    saved_model_path = 'saved_models/model.pt'
    saved_weights = torch.load(saved_model_path, map_location='cpu')

    device = 'cuda:1'
    text_encoder = TextEncoder(2, 2)
    text_encoder.load_state_dict(saved_weights['txt_encoder'])
    text_encoder = text_encoder.to(device)
    image_encoder = ImageEncoder()
    image_encoder.load_state_dict(saved_weights['img_encoder'])
    image_encoder = image_encoder.to(device)

    val_dataset = Recipe1MDataset(part='val')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    save_dir = 'saved_models/'

    freeze_params(text_encoder)
    freeze_params(image_encoder)

    batch_size = 1
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    image_encoder.eval()
    text_encoder.eval()

    text_embeddings = list()
    image_features = list()
    attention_masks = list()

    with torch.no_grad():
        for text, image in tqdm(val_loader):
            text_inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
            text_outputs = text_encoder(**text_inputs)
            image_outputs = image_encoder(image.to(device))

            for text_output, image_feature, attention_mask in zip(text_outputs, image_outputs,
                                                                  text_inputs.attention_mask):
                text_embeddings.append(text_output.cpu())
                image_features.append(image_feature.cpu())
                attention_masks.append(attention_mask.cpu())

        with open('/common/users/upp10/cs536/validation/text_embeddings.pkl', 'wb') as f:
            pickle.dump(text_embeddings, f)

        with open('/common/users/upp10/cs536/validation/image_features.pkl', 'wb') as f:
            pickle.dump(text_embeddings, f)

        with open('/common/users/upp10/cs536/validation/attention_masks.pkl', 'wb') as f:
            pickle.dump(text_embeddings, f)
