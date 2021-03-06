from tqdm import tqdm

import numpy as np
import pandas as pd

import torch.optim as optim

from config import *

def fit(model, train_loader, valid_loader=None, epochs=5, lr=1e-5, ckpt_path=None): 
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

    model.to(device)

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

