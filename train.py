import os
import argparse
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from transformers import EncoderDecoderModel

from config import *
from dataset import *
from fit import *

def main(args):
    train_file, valid_file = args.train_file, args.valid_file
    xcol, ycol = args.xcol, args.ycol
    nrows = args.nrows

    tokenizer = load_tokenizer()
    
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        'bert-base-uncased', 'bert-base-uncased'
    )
    model = warm_start(model, tokenizer)

    ##dataset prep
    train_loader = DataLoader(
        AbsSummary(train_file, xcol, ycol, tokenizer, nrows=nrows), 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    valid_loader = None
    if valid_file:
        valid_loader = DataLoader(
            AbsSummary(valid_file, xcol, ycol, tokenizer, nrows=nrows//2),
            batch_size=BATCH_SIZE,
            num_workers=2,
            pin_memory=True
        ) 

    ##train encoder_decoder model
    fit(model, train_loader, valid_loader, args.ckpt_path)

    del model
    del tokenizer
    del train_loader
    del valid_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, help='path for training csv')
    parser.add_argument('--valid_file', type=str, help='full csv file with labels')
    parser.add_argument('--xcol', type=str, help='inputs')
    parser.add_argument('--ycol', type=str, help='labels')
    parser.add_argument('--nrows', type=int, default=10000, help='no of rows used for training')
    parser.add_argument('--ckpt_path', type=str, default='./encdec.summarizer', help='ckpt_path for saving model weights')

    options = parser.parse_args()
    main(options)

 

