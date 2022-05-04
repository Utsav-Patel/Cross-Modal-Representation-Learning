import torch
import os
import wandb

from torch import nn
from helper import get_transformer_input, save_model, rank
from tqdm import tqdm

num_its = 0
val_its = 0


def train_one_epoch(image_encoder, text_encoder, cm_transformer, dataloader, tokenizer, criterion, optimizer,
                    train_encoders=False, device='cuda', save_dir=None):
    global num_its
    print('New epoch!')
    if train_encoders:
        image_encoder.train()
        text_encoder.train()
    cm_transformer.train()

    train_loss, total_samples = 0, 0
    for text, image in tqdm(dataloader):
        num_its += 1
        text_inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
        text_outputs = text_encoder(**text_inputs)
        image_outputs = image_encoder(image.to(device))
        transformer_image_inputs, transformer_text_inputs, output_attention_mask, ground_truth = \
            get_transformer_input(image_outputs, text_outputs, text_inputs.attention_mask)
        text_padding_mask = ~output_attention_mask.bool()
        outputs = cm_transformer(transformer_image_inputs.to(device), transformer_text_inputs.to(device),
                                 text_padding_mask.to(device))
        loss = criterion(outputs.to(device), ground_truth.long().to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * image.shape[0]
        total_samples += image.shape[0]

        if num_its % 1000 == 0:
            save_dict = {
                'cm_transformer': cm_transformer.state_dict()
            }
            save_model(save_dict, fpath=os.path.join(save_dir, f'model_train_encoders_{train_encoders}_num_its_{num_its}.pt'))

        if num_its % 10 == 0:
            wandb.log({'train_loss': round(train_loss / total_samples, 4)})

    return train_loss / total_samples


def evaluate(image_encoder, text_encoder, cm_transformer, dataloader, tokenizer, criterion, device='cuda'):
    global val_its
    print('Evaluating')
    image_encoder.eval()
    text_encoder.eval()
    cm_transformer.eval()

    val_loss, total_samples = 0, 0
    for text, image in tqdm(dataloader):
        val_its += 1
        text_inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
        text_outputs = text_encoder(**text_inputs)
        image_outputs = image_encoder(image.to(device))
        transformer_image_inputs, transformer_text_inputs, output_attention_mask, ground_truth = \
            get_transformer_input(image_outputs, text_outputs, text_inputs.attention_mask)
        text_padding_mask = ~output_attention_mask.bool()
        outputs = cm_transformer(transformer_image_inputs.to(device), transformer_text_inputs.to(device),
                                 text_padding_mask.to(device))
        loss = criterion(outputs, ground_truth.to(device).long())

        val_loss += loss.item() * image.shape[0]
        total_samples += image.shape[0]

        if val_its % 10 == 0:
            wandb.log({'val_loss': round(val_loss / total_samples, 4)})

    return val_loss / total_samples


def train(image_encoder, text_encoder, cm_transformer, train_dataloader, val_dataloader, tokenizer, save_dir,
          train_encoders=False, device='cuda', num_epochs=100, lr=2e-5):
    min_val_loss = float('inf')
    project_name = 'cross_modal_attention'
    wandb.init(project=project_name, entity='cs536')
    save_dir = os.path.join(save_dir, wandb.run.id)
    os.makedirs(save_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 2]), reduction='mean')
    if train_encoders:
        optimizer = torch.optim.SGD(
            [
                {'params': image_encoder.parameters()},
                {'params': text_encoder.parameters()},
                {'params': cm_transformer.parameters()}
            ],
            lr=lr
        )
    else:
        optimizer = torch.optim.SGD(cm_transformer.parameters(), lr=lr)

    for epoch in range(num_epochs):

        train_loss = train_one_epoch(image_encoder, text_encoder, cm_transformer, train_dataloader, 
                                    tokenizer, criterion, optimizer, train_encoders, device, save_dir)
        val_loss = evaluate(image_encoder, text_encoder, cm_transformer,
                            val_dataloader, tokenizer, criterion, device)

        # if val_loss < min_val_loss:
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

        save_dict['train_loss'] = train_loss
        save_dict['val_loss'] = val_loss

        save_model(save_dict, fpath=os.path.join(save_dir, f'model_train_encoders_{train_encoders}_epoch_{epoch}.pt'))

        print(f'Epoch: {epoch}, Train loss: {train_loss}, Val loss: {val_loss}')

