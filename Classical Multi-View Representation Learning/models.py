from torch import nn
from torchvision import models
from transformers import BertModel


class BertEncoder(nn.Module):
    """
    means: boolean which decides whether to take mean of all token 
    representations or to take cls token representation
    """
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
        # Removing classification layer
        num_feat = self.main.fc.in_features
        self.main.fc = nn.Linear(num_feat, 768)

    def forward(self, img):
        return self.main(img)


class Embedder(nn.Module):
    def __init__(self, output_size, input_size=1024):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Linear(512, output_size)

    def forward(self, x):
        x = self.layer1(x)
        return self.layer2(x)