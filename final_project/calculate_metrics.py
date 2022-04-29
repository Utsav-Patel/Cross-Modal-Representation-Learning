import torch

from datasets import Recipe1MDataset
from models import TextEncoder, ImageEncoder, CrossModalAttention
from helper import calculate_metrics
from torch.utils.data import DataLoader
from transformers import BertTokenizer


if __name__ == '__main__':

    # Change paths here.
    saved_model_path = 'saved_models/model.pt'
    transformer_model_path = 'saved_models/'

    saved_weights = torch.load(saved_model_path, map_location='cpu')
    transformer_weights = torch.load(transformer_model_path, map_location='cpu')

    device = 'cuda:7'
    text_encoder = TextEncoder(2, 2)
    text_encoder.load_state_dict(saved_weights['txt_encoder'])
    text_encoder = text_encoder.to(device)

    image_encoder = ImageEncoder()
    image_encoder.load_state_dict(saved_weights['img_encoder'])
    image_encoder = image_encoder.to(device)

    cm_transformer = CrossModalAttention().to(device)
    cm_transformer.load_state_dict(transformer_weights)
    cm_transformer = cm_transformer.to(device)

    train_dataset = Recipe1MDataset(part='train')
    val_dataset = Recipe1MDataset(part='val')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    save_dir = 'saved_models/'

    # freeze_params(text_encoder)
    # freeze_params(image_encoder)

    batch_size = 8
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    calculate_metrics(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        cm_transformer=cm_transformer,
        dataloader=val_loader,
        tokenizer=tokenizer,
        device=device
    )
