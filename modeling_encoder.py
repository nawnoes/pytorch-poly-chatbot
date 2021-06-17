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
        self.poly_code_embeddings = nn.Embedding(self.poly_m, hidden_size)
        
        nn.init.normal_(self.poly_code_embeddings, hidden_size ** -0.5)
    def dot_product_attention(self, q, k, v):
        attention_weights = torch.matmul(q, k.transpose(2,1))
        attention_weights = F.softmax(attention_weights, -1)
        attention_output = torch.matmul(attention_weights, v)
        
        return attention_output
    
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
        
        batch_size, res_cnt, seq_length = response_input_ids.shape

        # Context Vector
        context_vec = self.lm_model(context_input_ids, context_input_masks)[0][:,0,:] # [batch_size, hidden_size]
        poly_code_ids = torch.arange(self.poly_m, dtype=torch.long).to(context_input_ids.device)
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch_size, self.poly_m)
        poly_codes = self.poly_code_embeddings(poly_code_ids) # [batch_size, poly_m, hidden_size]
        context_embs = self.dot_product_attention(poly_codes, context_vec, context_vec)
        
        # Response Vector
        response_input_ids = response_input_ids.view(-1, seq_length)
        response_input_masks = response_input_masks.view(-1, seq_length)
        candidate_embs = self.lm_model(response_input_ids, response_input_masks)[0][:,0,:]
        candidate_embs = candidate_embs.view(batch_size, res_cnt, -1) # [batch_size, res_cnt, hidden_size]
        
        # Merge
        if labels is not None:
            cand_emb = candidate_embs.permute(1,0,2)
            cand_emb = candidate_embs.expand(batch_size, batch_size, candidate_embs.shape[2]) # [batch_size, batch_size, hidden_size]
            cand_ctxt_attn_output = self.dot_product_attention(candidate_embs, context_embs, context_embs).squeeze()
            dot_product = (cand_ctxt_attn_output * cand_emb).sum(-1)
            mask = torch.eye(batch_size).to(context_input_ids.device)
            loss = F.log_softmax(dot_product, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()
            return loss
        else:
            cand_ctxt_attn_output = self.dot_product_attention(candidate_embs, context_embs, context_embs)
            dot_product = (cand_ctxt_attn_output * candidate_embs).sum(-1)
            return dot_product