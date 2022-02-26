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

def light_load_csv(path, cols, chunksize=1000):
    df = pd.read_csv(path, usecols=cols, chunksize=chunksize)
    xdf = pd.DataFrame(columns=cols)
    for chunk in df:
        xdf = pd.concat([xdf, chunk])
    return xdf

class AbsSummary(Dataset):
    def __init__(self, data_path, xcol, ycol, tokenizer, xmax=512, ymax=50):
        self.df = light_load_csv(data_path, [xcol, ycol]) 
        self.xcol = xcol
        self.ycol = ycol
        self.xmax = xmax
        self.ymax = ymax
        self.tokenizer = tokenizer

    def encode_str(self, s, lim):
        t = self.tokenizer.encode_plus(s, max_length=lim, 
                                       truncation=True, padding='max_length')
        return t['input_ids'], t['attention_mask'] 

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        x, xmask = self.encode_str(self.df.loc[idx, self.xcol], self.xmax)
        y, ymask = self.encode_str(self.df.loc[idx, self.ycol], self.ymax)
        x, xmask = torch.tensor(x), torch.tensor(xmask)
        y, ymask = torch.tensor(y), torch.tensor(ymask)
        return x, xmask, y, ymask, y 


