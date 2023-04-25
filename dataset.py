from typing import List, Dict, Union, Tuple

import os
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import BertTokenizerFast


def load_tokenizer(save_path:str = './bert.tokenizer') -> BertTokenizerFast:
    if not os.path.exists(save_path):
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        tokenizer.save_pretrained(save_path)
        return tokenizer
    return BertTokenizerFast.from_pretrained(save_path)


def light_load_csv(path:str, cols:List, nrows:Union[None,int]=None, chunksize:int=1000) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows, usecols=cols, chunksize=chunksize)
    xdf = pd.DataFrame(columns=cols)
    for chunk in df:
        xdf = pd.concat([xdf, chunk])
    return xdf


class AbsSummary(Dataset):
    def __init__(
        self, 
        data_path:str, 
        xcol:str, 
        ycol:str, 
        tokenizer:BertTokenizerFast, 
        xmax:int=512, 
        ymax:int=48, 
        nrows:Union[None,int]=None
    ):
        self.df = light_load_csv(data_path, [xcol, ycol], nrows=nrows) 
        self.xcol = xcol
        self.ycol = ycol
        self.xmax = xmax
        self.ymax = ymax
        self.tokenizer = tokenizer

    def encode_str(self, s:str, lim:int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self.tokenizer(s, max_length=lim, truncation=True, 
                           padding='max_length', return_tensors='pt')
        return t.input_ids[0], t.attention_mask[0] 

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, idx:int) -> Dict:
        x, xmask = self.encode_str(self.df.loc[idx, self.xcol], self.xmax)
        y, ymask = self.encode_str(self.df.loc[idx, self.ycol], self.ymax)
        labels = y[1:].clone()
        labels = torch.tensor([
            torch.tensor(-100) if t == self.tokenizer.pad_token_id else t 
            for t in labels
        ])
        return {
            'input_ids':x,
            'attention_mask':xmask,
            'labels': labels,
            'decoder_input_ids':y[:-1],
            'decoder_attention_mask':ymask[:-1]
        }


