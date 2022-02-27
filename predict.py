import os
import argparse
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from transformers import EncoderDecoderModel

from config import *
from dataset import *

def main(args):
    tokenizer = load_tokenizer()
    
    ##model warm_starting
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        'bert-base-uncased', 'bert-base-uncased'
    )

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    model.config.max_length = 142
    model.config.min_length = 56
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    
    if args.ckpt_path: model.load_state_dice(torch.load(args.ckpt_path))

    inputs = tokenizer(args.text, padding="max_length", truncation=True, max_length=SEQ_MAX_LEN, return_tensors="pt")
    input_ids = inputs.input_ids[0].to(device)
    attention_mask = inputs.attention_mask[0].to(device)

    outputs = model.generate(input_ids, attention_mask=attention_mask)
    output_str = tokenizer.decode(outputs, skip_special_tokens=True)

    print(' ')
    print('full text: ' + input_text)
    print('=' * 20)
    print('summary: ' + output_text)
    print(' ')

    del model
    del tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help='path for training csv')
    parser.add_argument('--ckpt_path', type=str, help='full csv file with labels')
    options = parser.parse_args()
    main(options) 

