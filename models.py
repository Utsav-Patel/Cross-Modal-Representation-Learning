from torch import nn
from torchvision import models
from transformers import BertModel


class BertEncoder(nn.Module):
    def __init__(self, means=False):
        super().__init__()
        self.main = BertModel.from_pretrained("bert-base-uncased")
        self.means = means

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.main(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.means:
            return output.last_hidden_state[:, 1:].mean(dim=1)
        return output.last_hidden_state[:, 0]


class ResnetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = models.resnet50(pretrained=True)
        self.main.fc = nn.Identity()

    def forward(self, img):
        return self.main(img)
