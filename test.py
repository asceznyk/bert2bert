import os
import numpy as np
import pandas as pd

from transformers import EncoderDecoderModel

from config import *
from dataset import *
from trainer import *

def main():
    tokenizer = load_tokenizer()
    model = EncoderDecoderModel.from_encoder_decoder_model
