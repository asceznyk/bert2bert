from tqdm import tqdm

import numpy as np
import pandas as pd

from config import *

def fit(model, train_loader, valid_loader=None, ckpt_path=None): 
    def run_epoch(split):
        is_train = split == 'train' 
        model.train(is_train)
        loader = train_loader if is_train else valid_loader

        avg_loss = 0
        pbar = tqdm(enumerate(loader), total=len(loader))
        for step, batch in pbar: 
            batch = [i.to(device) for i in batch]
            x, xmask, y, ymask = batch
            
            with torch.set_grad_enabled(is_train):  
                outputs = model(input_ids=x, attention_mask=xmask, 
                                labels=y, decoder_attention_mask=ymask,
                                return_dict=True)
                loss, logits = outputs.loss, outputs.logits 
                avg_loss += loss.item() / len(loader)

            if is_train:
                model.zero_grad() 
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                optimizer.step()

            pbar.set_description(f"epoch: {e+1}, loss: {loss.item():.3f}, avg: {avg_loss:.2f}")     
        return avg_loss

    model.to(device)

    best_loss = float('inf') 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    for e in range(EPOCHS):
        train_loss = run_epoch('train')
        valid_loss = run_epoch('valid') if valid_loader is not None else train_loss

        if ckpt_path is not None and valid_loss < best_loss:
            print(f'saving model weights to {ckpt_path}...')
            best_loss = valid_loss
            torch.save(model.state_dict(), ckpt_path)

