import os
import torch
import logging
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch.nn.functional as F
from transform import SelectionJoinTransform, SelectionSequentialTransform

# Dataset for prtraining Masked Language Model
class DatasetForMLM(Dataset):
    def __init__(self, tokenizer, max_len, path="../data/namuwiki.txt"):
        logging.info('start wiki data load')
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = []
        
        # 파일 리스트
        file_list = os.listdir(path)
        
        for file_name in file_list:#file_progress_bar:
            path = f'{path}/{file_name}'
            data_file = open(path, 'r', encoding='utf-8')
            for line in data_file:
                line = line[:-1]
                self.docs.append(line)
        logging.info('complete data load')
    
    def mask_tokens(self, inputs: torch.Tensor, mlm_probability=0.15, pad=True):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        labels = inputs.clone()
        # mlm_probability defaults to 0.15 in Bert
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        if pad:
            input_pads = self.max_len - inputs.shape[-1]
            label_pads = self.max_len - labels.shape[-1]
            
            inputs = F.pad(inputs, pad=(0, input_pads), value=self.tokenizer.pad_token_id)
            labels = F.pad(labels, pad=(0, label_pads), value=self.tokenizer.pad_token_id)
        
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def _tokenize_input_ids(self, input_ids: list, pad_to_max_length: bool = True):
        inputs = torch.tensor(self.tokenizer.encode(input_ids, add_special_tokens=False, max_length=self.max_len,
                                                    pad_to_max_length=pad_to_max_length, return_tensors='pt',
                                                    truncation=True))
        return inputs
    
    def __len__(self):
        """ Returns the number of documents. """
        return len(self.docs)
    
    def __getitem__(self, idx):
        inputs = self._tokenize_input_ids(self.docs[idx], pad_to_max_length=True)
        inputs, labels = self.mask_tokens(inputs, pad=True)
        
        inputs = inputs.squeeze()
        inputs_mask = inputs != 0
        labels = labels.squeeze()
        
        return inputs, inputs_mask.unsqueeze(0), labels


class DatasetForPolyEncoder(Dataset):
    """
    Dataset for Poly Encoder
    Dataformat: label    context     response
    retrun: transformed_context, transformed_response, labels 
    """
    def __init__(self, tokenizer, max_len, path):
        self.context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len = max_len)
        self.response_transform = SelectionSequentialTransform(tokenizer=tokenizer, max_len = max_len)
        
        self.data =[]
        negative_response = []
        
        with open(path, encoding='utf-8') as data_file:
            group = {
                'context':None,
                'response': [],
                'labels': []
            }
            
            for line in data_file:
                split_data = line.strip().split('\t')
                label, context, response = int(split_data[0]), split_data[1], split_data[2] # data format (label, context, response). context can have many sentences
                if label == 1 and len(group['response'])>0: # 라벨이 1보다 크고 response가 있는 경우
                    self.data.append(group)
                    group = {'context':None,'response':[], 'labels':[]}
                    
                else: # 아닌경우
                    negative_response.append(response)
                group['response'].append(response)
                group['labels'].append(label)
                group['context'] = context
            if len(group['response']) >0:
                self.data.append(group)
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):
        group = self.data[idx]
        context, response, labels = group['context'], group['response'], group['labels']
        
        transformed_context = self.context_transform(context)
        transformed_response = self.response_transform(response)
        
        return transformed_context, transformed_response, labels

if __name__ == '__main__':
    pass
