from typing import Union, List

from tqdm import tqdm

import numpy as np
import pandas as pd

import torch.optim as optim

from torch.utils.data import DataLoader

from transformers import BertTokenizerFast, EncoderDecoderModel


def warm_start(model:EncoderDecoderModel, tokenizer:BertTokenizerFast) -> EncoderDecoderModel:
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.encoder.config.vocab_size

    model.config.max_length = 142
    model.config.min_length = 56
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    return model


def fit(
    model:EncoderDecoderModel,
    train_loader:DataLoader,
    valid_loader:Union[None,DataLoader]=None,
    epochs:int=5,
    lr:float=1e-5,
    ckpt_path:Union[str,None]=None,
    device=torch.device
):
    def run_epoch(split):
        is_train = split == 'train'
        model.train(is_train)
        loader = train_loader if is_train else valid_loader

        avg_loss = 0
        pbar = tqdm(enumerate(loader), total=len(loader))
        for step, batch in pbar:
            batch = (v.to(device) for k, v in batch.items())
            x, xmask, labels, y, ymask = batch

            with torch.set_grad_enabled(is_train):
                outputs = model(input_ids=x, attention_mask=xmask,
                                labels=labels, decoder_attention_mask=ymask,
                                return_dict=True)
                loss = outputs.loss
                avg_loss += loss.item() / len(loader)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #scheduler.step()

            pbar.set_description(f"epoch: {e+1}, loss: {loss.item():.3f}, avg: {avg_loss:.2f}, latest lr: {optimizer.param_groups[0]['lr']}")
        return avg_loss

    best_loss = float('inf')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    for e in range(epochs):
        train_loss = run_epoch('train')
        valid_loss = run_epoch('valid') if valid_loader is not None else train_loss

        if ckpt_path is not None and valid_loss < best_loss:
            print(f'saving model weights to {ckpt_path}...')
            best_loss = valid_loss
            torch.save(model.state_dict(), ckpt_path)

