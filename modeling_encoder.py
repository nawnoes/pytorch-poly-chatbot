"""
This code referenced from chijames/Poly-Encoder
URL: https://github.com/chijames/Poly-Encoder/blob/master/encoder.py
"""
import torch.nn as nn
import torch
import torch.nn.functional as F


class PolyEncoder(nn.Module):
    def __init__(self,
                 model,
                 poly_m,
                 hidden_size):
        super().__init__()
        self.lm_model  = model
        self.poly_m = poly_m
        self.poly_encoder_embeddings = nn.Embedding(self.poly_m, hidden_size)
        
        nn.init.normal_(self.poly_encoder_embeddings, hidden_size ** -0.5)
    def dot_product_attention(self, q, k, v):
        attention_weights = torch.matmul(q, k.transpose(2,1))
        attention_weights = F.softmax(attention_weights, -1)
        attention_output = torch.matmul(attention_weights, v)
        
        return attention_weights
    def forward(self,
                context_input_ids,
                context_input_masks,
                response_input_ids,
                response_input_masks,
                labels = None):
        
        # during training only select the first response
        # we are using other instances in a batch as negative example
        if labels is None:
            response_input_ids = response_input_ids[:,0,:].unsqueeze(1)
            response_input_masks = response_input_masks[:, 0, :].unsqueeze(1)
        
        batch_size, res_cnt,seq_length = response_input_ids.shape # res_cnt is 1 during training
        