import os
import argparse
import numpy as np
import pandas as pd

from transformers import EncoderDecoderModel

from config import *
from dataset import *

def main(args):
    tokenizer = load_tokenizer()
    
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        'bert-base-uncased', 'bert-base-uncased'
    )
    model = warm_start(model, tokenizer)
    
    if args.ckpt_path: model.load_state_dict(torch.load(args.ckpt_path))

    inputs = tokenizer(args.text, padding="max_length", truncation=True, max_length=SEQ_MAX_LEN, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

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

