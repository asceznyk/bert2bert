from tqdm import tqdm

import numpy as np
import pandas as pd

import torch.optim as optim

from config import *

def fit(model, train_loader, valid_loader=None, ckpt_path=None): 
    def run_epoch(split):
        is_train = split == 'train' 
        model.train(is_train)
        loader = train_loader if is_train else valid_loader

        avg_loss = 0
        pbar = tqdm(enumerate(loader), total=len(loader))
        for step, batch in pbar:
            batch = (v.to(device) for k, v in batch.items())
            x, xmask, y, ymask = batch
            
            with torch.set_grad_enabled(is_train):  
                outputs = model(input_ids=x, attention_mask=xmask, 
                                labels=y, decoder_attention_mask=ymask,
                                return_dict=True)
                loss = outputs.loss
                avg_loss += loss.item() / len(loader)

            if is_train:
                model.zero_grad() 
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                optimizer.step()
            else:
                scheduler.step(loss)

            pbar.set_description(f"epoch: {e+1}, loss: {loss.item():.3f}, avg: {avg_loss:.2f}, latest lr: {optimizer.param_groups[0]['lr']}")     
        return avg_loss

    model.to(device)

    best_loss = float('inf') 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',  patience=1)
    for e in range(EPOCHS):
        train_loss = run_epoch('train')
        valid_loss = run_epoch('valid') if valid_loader is not None else train_loss

        if ckpt_path is not None and valid_loss < best_loss:
            print(f'saving model weights to {ckpt_path}...')
            best_loss = valid_loss
            torch.save(model.state_dict(), ckpt_path)

