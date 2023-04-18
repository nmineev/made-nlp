# network.py

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm


class Reorder(nn.Module):
    def forward(self, input):
        return input.permute((0, 2, 1))
        

class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, concat_number_of_features, hid_size=64):
        super(ThreeInputsNet, self).__init__()

        self.title_enc = nn.Sequential(
            nn.Embedding(n_tokens, embedding_dim=hid_size), 
            Reorder(), 
            nn.Conv1d(hid_size, hid_size, kernel_size=2), 
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )
        self.description_enc = nn.Sequential(
            nn.Embedding(n_tokens, embedding_dim=hid_size), 
            Reorder(), 
            nn.Conv1d(hid_size, hid_size * 2, kernel_size=3), 
            nn.ReLU(),
            nn.Conv1d(hid_size * 2, hid_size * 2, kernel_size=3), 
            nn.ReLU(),
            nn.BatchNorm1d(hid_size * 2),
            nn.Conv1d(hid_size * 2, hid_size * 2, kernel_size=3), 
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )
        self.categorical_enc = nn.Sequential(
            nn.Linear(n_cat_features, 128),
            nn.ReLU(),
        )
        self.inter_dense = nn.Linear(in_features=concat_number_of_features, out_features=hid_size*2)
        self.final_dense = nn.Linear(in_features=hid_size*2, out_features=1)

        

    def forward(self, title_description_catfeats):
        title, description, catfeats = title_description_catfeats
        title_emb = self.title_enc(title)
        description_emb = self.description_enc(description)
        catfeats_emb = self.categorical_enc(catfeats)
        emb = torch.cat([title_emb, description_emb, catfeats_emb], dim=1)
        out = self.inter_dense(emb)
        out = self.final_dense(out)
        return out
