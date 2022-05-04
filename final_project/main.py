import torch

from datasets import Recipe1MDataset
from models import TextEncoder, ImageEncoder, CrossModalAttention
from trainer import train
from helper import freeze_params
from torch.utils.data import DataLoader
from transformers import BertTokenizer


if __name__ == '__main__':

    saved_model_path = 'saved_models/model.pt'
    saved_weights = torch.load(saved_model_path, map_location='cpu')
    transformer_model_path = '/common/home/upp10/Desktop/Cross-Modal-Representation-Learning/final_project/saved_models/temp_model.pt'

    transformer_weights = torch.load(transformer_model_path, map_location='cpu')
    device = 'cuda:0'
    text_encoder = TextEncoder(2, 2)
    text_encoder.load_state_dict(saved_weights['txt_encoder'])
    text_encoder = text_encoder.to(device)
    image_encoder = ImageEncoder()
    image_encoder.load_state_dict(saved_weights['img_encoder'])
    image_encoder = image_encoder.to(device)

    cm_transformer = CrossModalAttention()
    cm_transformer.load_state_dict(transformer_weights['cm_transformer'])
    cm_transformer = cm_transformer.to(device)

    train_dataset = Recipe1MDataset(part='train')
    val_dataset = Recipe1MDataset(part='val')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    save_dir = 'saved_models/'

    freeze_params(text_encoder)
    freeze_params(image_encoder)

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    train(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        cm_transformer=cm_transformer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        tokenizer=tokenizer,
        save_dir=save_dir,
        train_encoders=False,
        device=device,
        lr=1e-2
    )
