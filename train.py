import os
import argparse
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

from dataset import *
from fit import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_file, valid_file = args.train_file, args.valid_file
    xcol, ycol = args.xcol, args.ycol
    nrows = args.nrows
    batch_size = args.batch_size

    tokenizer = load_tokenizer()

    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        'bert-base-uncased', 'bert-base-uncased'
    )
    model = warm_start(model, tokenizer)
    model = model.to(device)

    train_loader = DataLoader(
        AbsSummary(train_file, xcol, ycol, tokenizer, nrows=nrows),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    valid_loader = None
    if valid_file:
        valid_loader = DataLoader(
            AbsSummary(valid_file, xcol, ycol, tokenizer, nrows=nrows//2),
            batch_size=batch_size,
            num_workers=2,
            pin_memory=False
        )

    fit(
        model,
        train_loader,
        valid_loader,
        epochs=args.epochs,
        lr=args.lr,
        ckpt_path=args.ckpt_path,
        device
    )

    del model
    del tokenizer
    del train_loader
    del valid_loader
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, help='path for training csv')
    parser.add_argument('--valid_file', type=str, default='', help='full csv file with labels')
    parser.add_argument('--xcol', type=str, help='inputs')
    parser.add_argument('--ycol', type=str, help='labels')
    parser.add_argument('--nrows', type=int, default=10000, help='no of rows used for training')
    parser.add_argument('--ckpt_path', type=str, default='./encdec.summarizer', help='ckpt_path for saving model weights')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)

    options = parser.parse_args()
    main(options)



