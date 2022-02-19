import os
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import BertTokenizerFast

def load_tokenizer(save_path='./bert.tokenizer'):
    if not os.path.exists(save_path):
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        tokenizer.save_pretrained(save_path)
        return tokenizer
    return BertTokenizerFast.from_pretrained(save_path)

class AbsSummaries(Dataset):
    def __init__(self, data_path, xcol, ycol, tokenizer, xmax=512, ymax=50):
        self.df = pd.read_csv(data_path, usecols=[xcol, ycol])
        self.xcol = xcol
        self.ycol = ycol
        self.xmax = xmax
        self.ymax = ymax
        self.tokenizer = tokenizer

    def encode_str(self, s, lim, target=0):
        t = self.tokenizer.encode_plus(s, max_length=lim, truncation=True, padding=True)
        return t['input_ids'], t['attention_mask'] if not target else t['input_ids']

    def __getitem__(self, idx):
        x, mask = self.encode_str(self.df.loc[idx, self.xcol], self.xmax)
        y = torch.tensor(self.encode_str(self.df.loc[idx, self.ycol], self.ymax, target=1))
        return torch.tensor(x), torch.tensor(mask), y


