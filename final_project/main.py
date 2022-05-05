import torch
import argparse

from datasets import Recipe1MDataset
from models import TextEncoder, ImageEncoder, CrossModalAttention
from trainer import train
from helper import freeze_params
from torch.utils.data import DataLoader
from transformers import BertTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='retrieval model parameters')
    parser.add_argument('--wandb', default=1, type=int, choices=[0,1])
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--num_hidden_layers', default=2, type=int)
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--cm_ckpt_path')
    parser.add_argument('--pretrained_ckpt_path', default='saved_models/model.pt', type=str)
    parser.add_argument('--save_dir', default='saved_models', type=str)
    parser.add_argument('--train_encoders', default=False, type=bool)
    args = parser.parse_args()

    device = args.device

    saved_model_path = args.pretrained_ckpt_path
    saved_weights = torch.load(saved_model_path, map_location='cpu')

    text_encoder = TextEncoder(2, 2)
    text_encoder.load_state_dict(saved_weights['txt_encoder'])
    text_encoder = text_encoder.to(device)

    image_encoder = ImageEncoder()
    image_encoder.load_state_dict(saved_weights['img_encoder'])
    image_encoder = image_encoder.to(device)

    cm_transformer = CrossModalAttention(n_heads=args.num_attention_heads, n_layers=args.num_hidden_layers)
    # cm_transformer.apply(init_weights)
    transformer_model_path = args.cm_ckpt_path
    if transformer_model_path:
        transformer_weights = torch.load(transformer_model_path, map_location='cpu')
        cm_transformer.load_state_dict(transformer_weights['cm_transformer'])
    cm_transformer = cm_transformer.to(device)

    train_dataset = Recipe1MDataset(part='train')
    val_dataset = Recipe1MDataset(part='val')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    save_dir = args.save_dir

    if not args.train_encoders:
        freeze_params(text_encoder)
        freeze_params(image_encoder)
    else:
        print("Training encoders!")

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    train(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        cm_transformer=cm_transformer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        tokenizer=tokenizer,
        save_dir=save_dir,
        train_encoders=args.train_encoders,
        device=device,
        lr=args.lr
    )
