import torch

from transformers import BertConfig

CONFIG = BertConfig()
EPOCHS = 5
BATCH_SIZE = 8
HIDDEN_DIM = 512
LEARNING_RATE = 1e-5
VOCAB_SIZE = CONFIG.vocab_size
EMB_DIM = CONFIG.hidden_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



