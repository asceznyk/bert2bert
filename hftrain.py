import os
import argparse
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from transformers import EncoderDecoderModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from config import *
from dataset import *
from fit import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def main(args):
    train_file, valid_file = args.train_file, args.valid_file
    xcol, ycol = args.xcol, args.ycol
    nrows = args.nrows

    tokenizer = load_tokenizer()
    
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        'bert-base-uncased', 'bert-base-uncased'
    )
    model = warm_start(model, tokenizer)
    model = model.to(device)

    train_data = AbsSummary(train_file, xcol, ycol, tokenizer, nrows=nrows)
    valid_data = AbsSummary(valid_file, xcol, ycol, tokenizer, nrows=nrows//2)

    ##train encoder_decoder model
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        fp16=True, 
        output_dir="./",
        logging_steps=10,
        save_steps=10,
        eval_steps=4,
        # logging_steps=1000,
        # save_steps=500,
        # eval_steps=7500,
        # warmup_steps=2000,
        # save_total_limit=3,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
    )
    trainer.train()

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

 

