import torch

from datasets import Recipe1MDataset
from models import TextEncoder, ImageEncoder, CrossModalAttention
from helper import calculate_metrics
from torch.utils.data import DataLoader
from transformers import BertTokenizer


if __name__ == '__main__':

    # Change paths here.
    saved_model_path = '/common/home/as3503/as3503/courses/cs536/final_project/final_project/saved_models/model.pt'
    transformer_model_path = '/common/home/as3503/as3503/courses/cs536/final_project/final_project/saved_models/3shrex3f/model_train_encoders_False_epoch_1.pt'

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
    cm_transformer.load_state_dict(transformer_weights['cm_transformer'])
    cm_transformer = cm_transformer.to(device)

    val_dataset = Recipe1MDataset(part='val')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    batch_size = 8
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    medR, medR_std, glob_recall = calculate_metrics(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        cm_transformer=cm_transformer,
        dataloader=val_loader,
        tokenizer=tokenizer,
        device=device
    )
    with open(f"{'.'.join(saved_model_path.split('.')[:-1])}_logs.txt", 'w') as f:
        f.write(f"""MedR={medR:.4f}({medR_std:.4f})
Global recall: 1: {glob_recall[1]:.4f}, 5: {glob_recall[5]:.4f}, 10: {glob_recall[10]:.4f}
""")
