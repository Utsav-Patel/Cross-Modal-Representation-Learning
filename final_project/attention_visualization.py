import torch
from datasets import Recipe1MDataset
from models import TextEncoder, ImageEncoder, CrossModalAttention
from transformers import BertTokenizer
from torch import nn


def extract_self_attention_maps(transformer_encoder, x, mask, src_key_padding_mask):
    attention_maps = []
    num_layers = transformer_encoder.num_layers
    num_heads = transformer_encoder.layers[0].self_attn.num_heads
    norm_first = transformer_encoder.layers[0].norm_first
    with torch.no_grad():
        for i in range(num_layers):
            # compute attention of layer i
            h = x.clone()
            if norm_first:
                h = transformer_encoder.layers[i].norm1(h)
            attn = transformer_encoder.layers[i].self_attn(h, h, h, attn_mask=mask,
                                                           key_padding_mask=src_key_padding_mask,
                                                           need_weights=True)[1]
            attention_maps.append(attn)
            # forward of layer i
            x = transformer_encoder.layers[i](x,src_mask=mask,src_key_padding_mask=src_key_padding_mask)
    return attention_maps


if __name__ == '__main__':

    # Change paths here.
    saved_model_path = 'saved_models/model.pt'
    transformer_model_path = 'saved_models/model_train_encoders_False_epoch_1.pt'

    saved_weights = torch.load(saved_model_path, map_location='cpu')
    transformer_weights = torch.load(transformer_model_path, map_location='cpu')

    device = 'cuda'
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



    # batch_size = 8
    # seq_len = 25
    # d_model = 512
    # x = torch.randn((batch_size,seq_len,d_model))
    #
    # src_mask = torch.zeros((seq_len,seq_len)).bool()
    # src_key_padding_mask = torch.zeros((batch_size,seq_len)).bool()
    #
    # attention_maps = extract_self_attention_maps(cm_transformer,x,src_mask,src_key_padding_mask)

