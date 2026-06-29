import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import HuberLoss
from torch.optim import AdamW, NAdam


def hard_sigmoid(x):
    return torch.clamp((x + 2.5) / 5, min=0, max=1)


class LapSimModelDense(nn.Module):

    def __init__(self, hidden_size=400):
        super().__init__()

        self.d1 = nn.Linear(739, 450)
        self.d2 = nn.Linear(450, hidden_size)

        self.v1 = nn.Linear(hidden_size, 200)
        self.v2 = nn.Linear(200, 9)

        self.p1 = nn.Linear(hidden_size, 200)
        self.p2 = nn.Linear(200, 9)

    def forward(self, windows, vehicles):
        x = torch.concatenate((vehicles, windows), axis=1)

        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        # x = F.relu(self.d3(x)) + x

        p = F.relu(self.p1(x))
        v = F.relu(self.v1(x))

        pos = hard_sigmoid(self.p2(p))
        vel = self.v2(v)

        return pos, vel


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.c1 = nn.Conv1d(in_channels, out_channels, 5)
        self.b1 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.b1(F.gelu(F.max_pool1d(self.c1(x), 2)))


class LapSimModelCNN(nn.Module):

    def __init__(self):
        super().__init__()

        d_h = 200

        self.c1 = ConvBlock(3, 16)
        self.c2 = ConvBlock(16, 24)
        self.c3 = ConvBlock(24, 32)
        self.c4 = ConvBlock(32, 48)
        # self.c6 = ConvBlock(64, 72)

        self.d1 = nn.Linear(640, d_h)
        self.d2 = nn.Linear(d_h, d_h)
        # self.d3 = nn.Linear(100, 100)

        self.p1 = nn.Linear(d_h, d_h)
        self.p2 = nn.Linear(d_h, 9)
        self.v1 = nn.Linear(d_h, d_h)
        self.v2 = nn.Linear(d_h, 9)

        self.loss = HuberLoss()
        self.optimiser = AdamW(self.parameters(), weight_decay=0.01)

    def forward(self, windows, vehicles):
        t = self.c1(windows)
        t = self.c2(t)
        t = self.c3(t)
        t = self.c4(t)
        # t = self.c5(t)
        # print(t.shape)
        t = torch.flatten(t, 1)

        x = torch.concatenate((vehicles, t), axis=1)

        x = F.sigmoid(self.d1(x))
        x = F.sigmoid(self.d2(x)) + x
        # x = F.gelu(self.d3(x)) + x
        # x = F.gelu(self.d4(x)) + x

        p = F.sigmoid(self.p1(x)) + x
        p = self.p2(p)

        v = F.sigmoid(self.v1(x)) + x
        v = hard_sigmoid(self.v2(v))

        return p, v


class LapSimTransformerModel(nn.Module):

    def __init__(self, foresight=120, embed_dim=256, heads=64, patch_length=5):
        super().__init__()

        self.patch_embed = nn.Conv1d(3, embed_dim, kernel_size=patch_length, stride=patch_length)
        self.pos_embed = nn.Parameter(torch.randn(((foresight * 2) + 1) // patch_length, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.d1 = nn.Linear(embed_dim + 16, embed_dim)

        self.p1 = nn.Linear(embed_dim, embed_dim)
        self.p2 = nn.Linear(embed_dim, 9)
        self.v1 = nn.Linear(embed_dim, embed_dim)
        self.v2 = nn.Linear(embed_dim, 9)

        self.loss = HuberLoss()
        self.optimiser = AdamW(self.parameters(), weight_decay=0.00)

    def forward(self, windows, vehicles):
        x = self.patch_embed(windows).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.transformer(x).mean(dim=1)  # global average pooling for final output
        x = torch.concatenate((vehicles, x), axis=1)
        x = F.gelu(self.d1(x))

        p = F.relu(self.p1(x)) + x
        p = hard_sigmoid(self.p2(p))

        v = F.relu(self.v1(x)) + x
        v = hard_sigmoid(self.v2(v))

        return p, v
