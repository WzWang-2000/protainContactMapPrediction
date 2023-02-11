import torch
import torch.nn as nn

class MSAMultiHeadAttention(nn.Module):
    def __init__(self,n_input,n_head,dropout=0.0,bias=False,*args,**kwargs):
        super().__init__()
        self.toK=nn.Linear(n_input,n_input,bias=False)
        self.toQ=nn.Linear(n_input,n_input,bias=False)
        self.toV=nn.Linear(n_input,n_input,bias=False)
        self.Attention=nn.MultiheadAttention(n_input,n_head,dropout=dropout,bias=False)
    def forward(self,X):
        K=self.toK(X)
        Q=self.toQ(X)
        V=self.toV(X)
        Attn=self.Attention(Q,K,V,need_weights=False)
        return Attn


class SeqAttentionBlock(nn.Module):
    def __init__(self,n_input,n_head,ffn_n_hiddens,dropout=0.0,bias=False,*args,**kwargs):
        super().__init__()
        self.MultiHeadAttention=MSAMultiHeadAttention(n_input,n_head)
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(n_input)
        self.ffn=nn.Sequential(
                                nn.Linear(n_input,ffn_n_hiddens),
                                nn.ReLU(),
                                nn.Linear(ffn_n_hiddens,n_input)
                                )
        self.ln2 = nn.LayerNorm(n_input)
    def forward(self,x1d):
        residual=x1d
        x1d=self.MultiHeadAttention(x1d)[0]
        x1d+=residual
        x1d=self.ln1(x1d)
        residual=x1d
        x1d=self.ffn(x1d)
        x1d+=residual
        x1d=self.ln2(x1d)
        return x1d