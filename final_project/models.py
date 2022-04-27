import torch
from torch import nn
import math
from torchvision import models
from transformers import BertConfig, BertModel, BertTokenizer, ViTModel
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