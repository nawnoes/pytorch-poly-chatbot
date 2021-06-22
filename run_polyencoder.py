import warnings
warnings.filterwarnings("ignore")
import sys

# local: ../
sys.path.append('../')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import BertTokenizer
from transformers.optimization import AdamW
import os
import json
import logging
from datetime import datetime
from dataset import DatasetForMLM
from model_config import ModelConfig
from modeling_mlm import MLM

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
    pass

  def build_dataloader(self,train_test_split=0.1,train_shuffle=True, eval_shuffle=True):
    pass
  def train(self):
    pass
  def evaluate(self):
    pass
  def save(self):
    pass

def main():
  pass

if __name__ == '__main__':
  main()