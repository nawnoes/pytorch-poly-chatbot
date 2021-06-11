import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def self_attention(query, key, value, mask=None, causal=False):
  key_transpose = torch.transpose(key,-2,-1)                      # (bath, head_num, d_k, token_)
  matmul_result = torch.matmul(query,key_transpose)                # MatMul(Q,K)
  d_k = query.size()[-1]
  attention_score = matmul_result/math.sqrt(d_k)                  # Scale

  if mask is not None:
    attention_score = attention_score.masked_fill(mask == 0, -1e4)

  if causal:
    query_len = query.size()[2]
    i, j = torch.triu_indices(query_len, query_len, 1)
    attention_score[:, :, i, j] = -1e4

  softmax_attention_score = F.softmax(attention_score,dim=-1)  # 어텐션 값
  result = torch.matmul(softmax_attention_score,value)

  return result, softmax_attention_score

class MultiHeadAttention(nn.Module):
  def __init__(self, head_num =8 , d_model = 512,dropout = 0.1, causal=False):
    super(MultiHeadAttention,self).__init__()

    self.head_num = head_num
    self.d_model = d_model
    self.d_k = self.d_v = d_model // head_num
    self.causal = causal

    self.w_q = nn.Linear(d_model,d_model)
    self.w_k = nn.Linear(d_model,d_model)
    self.w_v = nn.Linear(d_model,d_model)
    self.w_o = nn.Linear(d_model,d_model)

    self.self_attention = self_attention
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, query, key, value, mask = None):
    if mask is not None:
      mask = mask.unsqueeze(1)

    batche_num = query.size(0)

    query = self.w_q(query).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)
    key = self.w_k(key).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)
    value = self.w_v(value).view(batche_num, -1, self.head_num, self.d_k).transpose(1, 2)

    attention_result, attention_score = self.self_attention(query, key, value, mask, self.causal)

    attention_result = attention_result.transpose(1,2).contiguous().view(batche_num, -1, self.head_num * self.d_k)


    return self.w_o(attention_result)

class FeedForward(nn.Module):
  def __init__(self,d_model, dropout = 0.1):
    super(FeedForward,self).__init__()
    self.w_1 = nn.Linear(d_model, d_model*4)
    self.w_2 = nn.Linear(d_model*4, d_model)
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, x):
    return self.w_2(self.dropout(F.relu(self.w_1(x))))

class LayerNorm(nn.Module):
  def __init__(self, features, eps=1e-6):
    super(LayerNorm,self).__init__()
    self.a_2 = nn.Parameter(torch.ones(features))
    self.b_2 = nn.Parameter(torch.zeros(features))
    self.eps = eps
  def forward(self, x):
    mean = x.mean(-1, keepdim =True) # 평균
    std = x.std(-1, keepdim=True)    # 표준편차

    return self.a_2 * (x-mean)/ (std + self.eps) + self.b_2

class ResidualConnection(nn.Module):
  def __init__(self, size, dropout):
    super(ResidualConnection,self).__init__()
    self.norm = LayerNorm(size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, sublayer):
    return x + self.dropout((sublayer(self.norm(x))))

class Encoder(nn.Module):
  def __init__(self, d_model, head_num, dropout):
    super(Encoder,self).__init__()
    self.multi_head_attention = MultiHeadAttention(d_model= d_model, head_num= head_num)
    self.residual_1 = ResidualConnection(d_model,dropout=dropout)

    self.feed_forward = FeedForward(d_model)
    self.residual_2 = ResidualConnection(d_model,dropout=dropout)

  def forward(self, input, mask):
    x = self.residual_1(input, lambda x: self.multi_head_attention(x, x, x, mask))
    x = self.residual_2(x, lambda x: self.feed_forward(x))
    return x

class Embeddings(nn.Module):
  def __init__(self, vocab_num, d_model):
    super(Embeddings,self).__init__()
    self.emb = nn.Embedding(vocab_num,d_model)
    self.d_model = d_model
  def forward(self, x):
    return self.emb(x) * math.sqrt(self.d_model)

class PositionalEmbedding(nn.Module):
  def __init__(self, dim, max_seq_len):
    super().__init__()
    self.embedding = nn.Embedding(max_seq_len, dim)

  def forward(self, x):
    t = torch.arange(x.shape[1], device=x.device)
    return self.embedding(t)

class MLM(nn.Module):
  def __init__(self,
               vocab_size,
               dim=2560,
               encoder_depth=1,
               max_seq_len=128,
               head_num=32,
               dropout=0.1):
    super().__init__()
    self.vocab_size = vocab_size

    self.token_emb = nn.Embedding(vocab_size, dim)
    self.position_emb = PositionalEmbedding(dim,max_seq_len)

    self.encoders = nn.ModuleList([Encoder(d_model=dim, head_num=head_num, dropout=dropout) for _ in range(encoder_depth)])

    self.norm = nn.LayerNorm(dim)
    self.lm_head = self.lm_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Linear(dim, vocab_size,bias=False)
            )

  def forward(self, input_ids, input_mask, labels=None):
    inputs_embed = self.token_emb(input_ids)
    position_embed = self.position_emb(input_ids)

    hidden_states = inputs_embed + position_embed

    for encoder in self.encoders:
      hidden_states = encoder(hidden_states, input_mask)

    lm_logits = self.lm_head(self.norm(hidden_states))

    loss = None
    if labels is not None:
      # only calculating loss on masked tokens
      loss_mx = labels != -100
      output = lm_logits[loss_mx].view(-1, self.vocab_size)
      labels = labels[loss_mx].view(-1)
      loss_fn = nn.CrossEntropyLoss()
      loss = loss_fn(output, labels)

    return lm_logits, loss


if __name__=="__main__":
  pass
