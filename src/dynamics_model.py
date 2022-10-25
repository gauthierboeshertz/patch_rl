import math 
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1,activation=nn.ReLU):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        
        self.layers = []
        self.layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(activation())
        
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.layers(x)
    
    
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class AttentionDynamicsModel(nn.Module):
    def __init__(self, in_features, out_features, state_dim,action_dim, object_disentanglement=None,num_attention_layers =2, num_emb_layers=2,emb_dim=128, num_heads = 1):
        super(AttentionDynamicsModel, self).__init__()

        self.emb_dim = emb_dim
        self.num_attention_layers = num_attention_layers
        self.object_disentanglement = object_disentanglement
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.attention_layers = nn.ModuleList()
        #self.attention_layers.append(QKV_Attention(in_features, emb_dim, num_heads, num_emb_layers=num_emb_layers))
        #for _ in range(num_attention_layers):
        #    self.attention_layers.append(QKV_Attention(emb_dim, emb_dim, num_heads, num_emb_layers=num_emb_layers))
        self.attention_layers.append(MultiheadAttention(in_features, emb_dim, num_heads ))
        for _ in range(num_attention_layers - 1):
            self.attention_layers.append(MultiheadAttention(emb_dim, emb_dim, num_heads))

        self.out = nn.Linear(emb_dim, out_features)
        
        
    def forward(self, x, return_attention=False):
        
        if self.object_disentanglement:
            obj_x = torch.zeros(x.shape[0],len(self.object_disentanglement),self.action_dim + self.state_dim).to(x.device)
            for obj_idx, obj in enumerate(self.object_disentanglement):
                obj_x[:,obj_idx,obj] = x[:,obj]
            x = obj_x
        else:
            x = torch.diag_embed(x,)
        
        attention_weights = []
        for i in range(self.num_attention_layers):
            x, attention_i = self.attention_layers[i](x, return_attention=True)
            attention_weights.append(attention_i)
            
        x = self.out(x)
        if self.object_disentanglement:
            x = x[:,:len(self.object_disentanglement)//2,]
            entangled_x = torch.zeros(x.shape[0], self.state_dim).to(x.device)
            for obj_idx, obj in enumerate(self.object_disentanglement[:len(self.object_disentanglement)//2]):
                entangled_x[:,obj] = x[:,obj_idx,obj]
            x = entangled_x
        else :
            x = torch.diagonal(x, dim1=-2, dim2=-1)
            x = x[:,:self.state_dim] 
        return x, attention_weights