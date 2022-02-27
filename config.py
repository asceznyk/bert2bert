import torch

from transformers import BertConfig

CONFIG = BertConfig()
EPOCHS = 10
BATCH_SIZE = 32
MAX_LENGTH = 25
HIDDEN_DIM = 512
LEARNING_RATE = 1e-4
VOCAB_SIZE = CONFIG.vocab_size
EMB_DIM = CONFIG.hidden_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



