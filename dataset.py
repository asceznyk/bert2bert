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

def light_load_csv(path, cols, nrows=None, chunksize=1000):
    df = pd.read_csv(path, nrows=nrows, usecols=cols, chunksize=chunksize)
    xdf = pd.DataFrame(columns=cols)
    for chunk in df:
        xdf = pd.concat([xdf, chunk])
    return xdf

class AbsSummary(Dataset):
    def __init__(self, data_path, xcol, ycol, tokenizer, xmax=512, ymax=128, nrows=None):
        self.df = light_load_csv(data_path, [xcol, ycol], nrows=nrows) 
        self.xcol = xcol
        self.ycol = ycol
        self.xmax = xmax
        self.ymax = ymax
        self.tokenizer = tokenizer

    def encode_str(self, s, lim):
        return self.tokenizer(s, 
                       max_length=lim,  
                       truncation=True, 
                       padding='max_length',
                       return_tensors='pt').input_ids[0] 

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        x = self.encode_str(self.df.loc[idx, self.xcol], self.xmax)
        y = self.encode_str(self.df.loc[idx, self.ycol], self.ymax)
        y = torch.tensor([torch.tensor(-100) if token == self.tokenizer.pad_token_id else token for token in y])
        return x, y


