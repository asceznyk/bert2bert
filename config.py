import torch

EPOCHS = 10
SEQ_MAX_LEN = 512
SUM_MAX_LEN = 128
BATCH_SIZE = 8
LEARNING_RATE = 1e-5  #1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def warm_start(model, tokenizer):
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    model.config.max_length = SUM_MAX_LEN 
    model.config.min_length = 56
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    return model

