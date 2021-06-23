import random
import warnings
warnings.filterwarnings("ignore")
import sys

# local: ../
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import BertTokenizer
from transformers.optimization import AdamW
import os
import json
import logging
import numpy as np
from datetime import datetime
from dataset import DatasetForMLM
from model_config import ModelConfig
from modeling_mlm import MLM
from modeling_encoder import PolyEncoder

from sklearn.metrics import label_ranking_average_precision_score

from apex import amp

class PolyEncoderTrainer(object):
  def __init__(self,
               dataset,
               model,
               tokenizer,
               max_len,
               model_name,
               checkpoint_path,
               device=None,
               batch_size=8,
               log_dir='./logs',
               fp16=True):
    self.dataset = dataset
    self.model = model
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.model_name = model_name
    self.checkpoint_path = checkpoint_path
    self.device = device
    self.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    self.train_batch_size = batch_size
    self.eval_batch_size = batch_size
    self.log_dir = log_dir
    self.fp16 = fp16

    if device is None:
      self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    logging.basicConfig(filename=f'{log_dir}/{self.model_name}-{datetime.now().date()}.log', level=logging.INFO)


  def build_dataloader(self,train_test_split=0.1,train_shuffle=True, eval_shuffle=True):
    pass
  def train(self):
    pass
  def evaluate(self, dataloader):
    self.model.eval()
    eval_loss, eval_hit_times = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    r10 = r2 = r1 = r5 = 0
    mrr = []
    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(self.device) for t in batch)

        context_token_ids_list_batch, context_input_masks_list_batch, \
        response_token_ids_list_batch, response_input_masks_list_batch, labels_batch = batch
        with torch.no_grad():
            logits = self.model(context_token_ids_list_batch, context_input_masks_list_batch,
                                          response_token_ids_list_batch, response_input_masks_list_batch)
            loss = F.cross_entropy(logits, torch.argmax(labels_batch, 1))

        r2_indices = torch.topk(logits, 2)[1] # R 2 @ 100
        r5_indices = torch.topk(logits, 5)[1] # R 5 @ 100
        r10_indices = torch.topk(logits, 10)[1] # R 10 @ 100
        r1 += (logits.argmax(-1) == 0).sum().item()
        r2 += ((r2_indices==0).sum(-1)).sum().item()
        r5 += ((r5_indices==0).sum(-1)).sum().item()
        r10 += ((r10_indices==0).sum(-1)).sum().item()
        # mrr
        logits = logits.data.cpu().numpy()
        for logit in logits:
            y_true = np.zeros(len(logit))
            y_true[0] = 1
            mrr.append(label_ranking_average_precision_score([y_true], [logit]))
        eval_loss += loss.item()
        nb_eval_examples += labels_batch.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = r1 / nb_eval_examples

    result = {
        'eval_loss': eval_loss,
        'eval_accuracy': eval_accuracy,
        'R1': r1 / nb_eval_examples,
        'R2': r2 / nb_eval_examples,
        'R5': r5 / nb_eval_examples,
        'R10': r10 / nb_eval_examples,
        'MRR': np.mean(mrr),
    }

    return result
  def save(self):
    pass

def set_seed(seed_num):
  torch.manual_seed(seed_num)
  np.random.seed(seed_num)
  random.seed(seed_num)

def main():
  seed = 9
  base_path = '.'

  set_seed(seed)
  log_dir = f'{base_path}/logs'
  config_path = f'{base_path}/config-mlm.json'

  # config
  config = ModelConfig(config_path=config_path).get_config()

  # tokenizer
  tokenizer = BertTokenizer(vocab_file=config.vocab_path, do_lower_case=False)


  # dataset
  dataset = DatasetForPolyEncoder(tokenizer, config.max_seq_len, path=config.data_path)

  # model
  mlm_model = MLM(
    vocab_size=tokenizer.vocab_size,
    dim=config.dim,
    depth=config.depth,
    max_seq_len=config.max_seq_len,
    head_num=config.n_head
  )
  poly_encoder = PolyEncoder(
    model= mlm_model,
    poly_m= config.poly_m,
    poly_code_embeddings = config.poly_code_embdeddings
  )


if __name__ == '__main__':
  main()