import os
import argparse
import numpy as np
import pandas as pd

from transformers import EncoderDecoderModel

from dataset import *
from fit import *

def main(args):
    text_file = args.text_file

    tf = open(text_file, 'r')
    text = tf.read()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = load_tokenizer()
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        'bert-base-uncased', 'bert-base-uncased'
    )
    model = warm_start(model, tokenizer)
    model.load_state_dict(torch.load(args.ckpt_path))
    model.to(device)

    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(' ')
    print('text: ' + text)
    print('= ' * 20)
    print('summary: ' + summary)
    print(' ')

    del model
    del tokenizer
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file', type=str, help='raw text file')
    parser.add_argument('--ckpt_path', type=str, help='model path after training')
    options = parser.parse_args()
    main(options)

