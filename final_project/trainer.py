import torch
import os

from torch import nn
from helper import get_transformer_input, save_model
from tqdm import tqdm


def train_one_epoch(image_encoder, text_encoder, cm_transformer, dataloader, tokenizer, criterion, optimizer, train_encoders=False, device='cuda'):
    print('New epoch!')
    if train_encoders:
        image_encoder.train()
        text_encoder.train()
    cm_transformer.train()


    train_loss, total_samples = 0, 0
    for text, image in tqdm(dataloader):
        text_inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
        text_outputs = text_encoder(**text_inputs)
        image_outputs = image_encoder(image.to(device))
        transformer_image_inputs, transformer_text_inputs, ground_truth = get_transformer_input(image_outputs, text_outputs)
        text_padding_mask = ~text_inputs.attention_mask.bool()
        outputs = cm_transformer(transformer_image_inputs.to(device), transformer_text_inputs.to(device), text_padding_mask.to(device))
        loss = criterion(outputs, ground_truth.to(device))
        optimizer.zero_grad()
        loss.backward()
        
        train_loss += loss.item() * image.shape[0]
        total_samples += image.shape[0]

    return train_loss / total_samples

def evaluate(image_encoder, text_encoder, cm_transformer, dataloader, tokenizer, criterion, device='cuda'):
    print('Evaluating')
    image_encoder.eval()
    text_encoder.eval()
    cm_transformer.eval()

    val_loss, total_samples = 0, 0
    with torch.no_grad():
        for text, image in dataloader:
            text_inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
            text_outputs = text_encoder(**text_inputs)
            image_outputs = image_encoder(image.to(device))
            transformer_image_inputs, transformer_text_inputs, ground_truth = get_transformer_input(image_outputs, text_outputs)
            text_padding_mask = ~text_inputs.attention_mask.bool()
            outputs = cm_transformer(transformer_image_inputs, transformer_text_inputs, text_padding_mask)
            loss = criterion(outputs, ground_truth.to(device))
            
            val_loss += loss.item() * image.shape[0]
            total_samples += image.shape[0]

    return val_loss / total_samples


def train(image_encoder, text_encoder, cm_transformer, train_dataloader, val_dataloader, tokenizer, save_dir, train_encoders=False, device='cuda', num_epochs=100, lr=1e-4):
    min_val_loss = float('inf')
    criterion = nn.CrossEntropyLoss()
    if train_encoders:
        optimizer = torch.optim.Adam(
            [
                {'params': image_encoder.parameters()},
                {'params': text_encoder.parameters()},
                {'params': cm_transformer.parameters()}
            ],
            lr=lr
        )
    else:
        optimizer = torch.optim.Adam(cm_transformer.parameters(), lr=lr)


    for epoch in range(num_epochs):
        train_loss = train_one_epoch(image_encoder, text_encoder, cm_transformer, train_dataloader, 
                                    tokenizer, criterion, optimizer, train_encoders, device)
        val_loss = evaluate(image_encoder, text_encoder, cm_transformer, 
                            val_dataloader, tokenizer, criterion, device)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            if train_encoders:
                save_dict = {
                    'image_encoder': image_encoder.state_dict(),
                    'text_encoder': text_encoder.state_dict(),
                    'cm_transformer': cm_transformer.state_dict()
                }
            else:
                save_dict = {
                    'cm_transformer': cm_transformer.state_dict()
                }

            save_model(save_dict, fpath=os.path.join(save_dir, f'model_train_encoders_{train_encoders}.pt'))

        print(f'Epoch: {epoch}, Train loss: {train_loss}, Val loss: {val_loss}')

