import torch
from torch import nn
import math
from transformers import BertConfig, BertModel, ViTModel
import pdb


class TextEncoder(nn.Module):
    def __init__(self, num_heads, num_hidden_layers):
        super().__init__()

        config = BertConfig(num_attention_heads=num_heads, num_hidden_layers=num_hidden_layers)
        self.main = BertModel(config)
        bert_model_state_dict = BertModel.from_pretrained('bert-base-uncased').state_dict()
        embedding_weights = {x:bert_model_state_dict[x] for x in bert_model_state_dict if 'embedding' in x}
        self.main.load_state_dict(embedding_weights, strict=False)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_outputs = self.main(input_ids, attention_mask, token_type_ids, output_attentions=True)
        return bert_outputs.last_hidden_state


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    def forward(self, img):
        outputs = self.main(img)
        return outputs.last_hidden_state


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = torch.transpose(pe, 0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[0, :x.size(1), :]
        return self.dropout(x)


class CrossModalAttention(nn.Module):
    def __init__(self, model_dim=768, n_heads=2, n_layers=2, num_image_patches=197, num_classes=2, drop_rate=0.1):
        super().__init__()
        self.d_model = model_dim
        self.text_pos_embed = SinusoidalPositionalEncoding(model_dim, dropout=drop_rate)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.image_pos_embed = nn.Parameter(torch.zeros(1, num_image_patches + 1, model_dim))
        self.image_pos_drop = nn.Dropout(p=drop_rate)
        layers = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            batch_first=True,
            dropout=drop_rate
        )
        self.encoder = nn.TransformerEncoder(layers, num_layers=n_layers)
        self.cls_projection = nn.Linear(model_dim, num_classes)

    def forward(self, image_features, text_features, src_key_padding_mask=None):
        image_features *= math.sqrt(self.d_model)
        text_features *= math.sqrt(self.d_model)

        batch_size = image_features.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        image_features = torch.cat((cls_token, image_features), dim=1)
        image_features = image_features + self.image_pos_embed
        image_features = self.image_pos_drop(image_features)

        text_features = self.text_pos_embed(text_features)

        sep_token = self.sep_token.expand(batch_size, -1, -1)
        transformer_input = torch.cat((image_features, sep_token, text_features), dim=1)
        if src_key_padding_mask is not None:
            src_key_padding_mask = torch.cat((torch.zeros(batch_size, image_features.shape[1] + 1).to(
                transformer_input.device), src_key_padding_mask), dim=1)
        transformer_outputs = self.encoder(transformer_input, src_key_padding_mask=src_key_padding_mask)
        cls_outputs = transformer_outputs[:, 0, :]
        return self.cls_projection(cls_outputs)
