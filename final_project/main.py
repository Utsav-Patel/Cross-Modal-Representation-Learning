from datasets import Recipe1MDataset
from models import TextEncoder, ImageEncoder, CrossModalAttention
from trainer import train
from torch.utils.data import DataLoader
from transformers import BertTokenizer


if __name__ == '__main__':
    device = 'cuda'
    text_encoder = TextEncoder(2, 2).to(device)
    image_encoder = ImageEncoder().to(device)
    cm_transformer = CrossModalAttention().to(device)
    train_dataset = Recipe1MDataset(part='train')
    val_dataset = Recipe1MDataset(part='val')
    train_loader = DataLoader(train_dataset, batch_size=4)
    val_loader = DataLoader(val_dataset, batch_size=4)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    save_dir = 'saved_models/'

    print('Starting training')
    train(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        cm_transformer=cm_transformer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        tokenizer=tokenizer,
        save_dir=save_dir,
        train_encoders=False,
        # device='cuda:7'
    )

