import torch
from datasets import Recipe1MDataset
from models import TextEncoder, ImageEncoder, CrossModalAttention
from transformers import BertTokenizer
from torch import nn


def compute_selfattention(transformer_encoder, x, src_key_padding_mask, i_layer, d_model, num_heads):
    h = F.linear(x, transformer_encoder.layers[i_layer].self_attn.in_proj_weight, bias=transformer_encoder.layers[i_layer].self_attn.in_proj_bias)
    qkv = h.reshape(x.shape[0], x.shape[1], num_heads, 3 * d_model//num_heads)
    qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
    q, k, v = qkv.chunk(3, dim=-1) # [Batch, Head, SeqLen, d_head=d_model//num_heads]
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) # [Batch, Head, SeqLen, SeqLen]
    d_k = q.size()[-1]
    attn_probs = attn_logits / math.sqrt(d_k)
    # combining src_mask e.g. upper triangular with src_key_padding_mask e.g. columns over each padding position
    combined_mask = torch.zeros_like(attn_probs)
    if mask is not None:
        combined_mask += mask.float() # assume mask of shape (seq_len,seq_len)
    if src_key_padding_mask is not None:
        combined_mask += src_key_padding_mask.float().unsqueeze(1).unsqueeze(1).repeat(1,num_heads,x.shape[1],1)
        # assume shape (batch_size,seq_len), repeating along head and line dimensions == "column" mask
    combined_mask = torch.where(combined_mask>0,torch.zeros_like(combined_mask)-float("inf"),torch.zeros_like(combined_mask))
    # setting masked logits to -inf before softmax
    attn_probs += combined_mask
    attn_probs = F.softmax(attn_probs, dim=-1)
    return attn_logits,attn_probs


def extract_selfattention_maps(transformer_encoder, x, src_key_padding_mask):
    attn_logits_maps = []
    attn_probs_maps = []
    num_layers = transformer_encoder.num_layers
    d_model = transformer_encoder.layers[0].self_attn.embed_dim
    num_heads = transformer_encoder.layers[0].self_attn.num_heads
    norm_first = transformer_encoder.layers[0].norm_first
    with torch.no_grad():
        for i in range(num_layers):
            # compute attention of layer i
            h = x.clone()
            if norm_first:
                h = transformer_encoder.layers[i].norm1(h)
            # attn = transformer_encoder.layers[i].self_attn(h, h, h,attn_mask=mask,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
            # attention_maps.append(attn) # of shape [batch_size,seq_len,seq_len]
            attn_logits,attn_probs = compute_selfattention(transformer_encoder,h,src_key_padding_mask,i,d_model,num_heads)
            attn_logits_maps.append(attn_logits) # of shape [batch_size,num_heads,seq_len,seq_len]
            attn_probs_maps.append(attn_probs)
            # forward of layer i
            x = transformer_encoder.layers[i](x, src_key_padding_mask=src_key_padding_mask)
    return attn_logits_maps,attn_probs_maps


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

