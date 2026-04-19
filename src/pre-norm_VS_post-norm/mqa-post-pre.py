import copy
import torch
import torch.nn as nn


class GQA(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 kv_head_num: int,
                 q_head_num: int,
                 ):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.kv_head_num = kv_head_num
        self.q_head_num = q_head_num
        
        self.q_head_dim = emb_dim//q_head_num
        
        self.map_qkv = nn.Linear(emb_dim, emb_dim + self.q_head_dim*self.kv_head_num*2)
        self.linear_out = nn.Linear(emb_dim, emb_dim)
                        
    def forward(self, input:torch.Tensor)-> torch.Tensor:
        b, s, _ = input.shape
        
        qkv = self.map_qkv(input)
        q = qkv[..., :self.emb_dim]
        k = qkv[..., self.emb_dim:self.emb_dim+self.q_head_dim*self.kv_head_num]
        v = qkv[..., self.emb_dim+self.q_head_dim*self.kv_head_num:]
        
        q = q.view(b, s, self.q_head_num, self.q_head_dim).transpose(1, 2)
        k = k.view(b, s, self.kv_head_num, self.q_head_dim).transpose(1, 2)
        v = v.view(b, s, self.kv_head_num, self.q_head_dim).transpose(1, 2)
        print(q.shape, k.shape, v.shape)
        
        _qk_ratio = self.q_head_num//self.kv_head_num 
        k = k.repeat_interleave(repeats=_qk_ratio, dim=1)
        v = v.repeat_interleave(repeats=_qk_ratio, dim=1)
        print(q.shape, k.shape, v.shape)
        
        score = torch.matmul(q, k.transpose(-2, -1))/self.q_head_dim**0.5
        casual_mask = torch.tril(torch.ones(s, s, dtype=bool)).unsqueeze(0).unsqueeze(0)
        score = score.masked_fill(~casual_mask, float("-inf"))
        score = torch.softmax(score, dim=-1)
        output = torch.matmul(score, v)
        
        output = output.transpose(1, 2).contiguous().view(b, s, self.emb_dim)
        
        return self.linear_out(output)
    
    
class EncoderBlock(nn.Module):
    def __init__(self, 
                 emb_dim: int,
                 kv_head_num: int,
                 q_head_num: int,
                 max_length: int=1024
                 ):
        super().__init__()
        self.emb_dim = emb_dim
        self.kv_head_num = kv_head_num
        self.q_head_num = q_head_num
        self.max_length = max_length
        
        self.attention = GQA(
                 emb_dim,
                 kv_head_num,
                 q_head_num)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        
        self.fnn = nn.Sequential(
                        nn.Linear(emb_dim, emb_dim),
                        nn.SiLU(),
                        nn.Linear(emb_dim, emb_dim))
        
    def _position_econding_(self,
                            x: torch.Tensor,
                            )->torch.Tensor:
        seq_len = x.shape[1]
        postion = torch.arange(seq_len)
        dim = torch.arange(0, self.emb_dim, 2)
        dim = 1/(10000**(dim/self.emb_dim))
        dim = dim.repeat_interleave(repeats=2, dim=-1)
        pe = torch.outer(postion, dim)
        pe[:, ::2] = torch.sin(pe[:, ::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe
        
    def forward_prenorm(self, 
                x: torch.Tensor,
                if_position: bool=True)->torch.Tensor:
        
        # Use post norm 
        if if_position:
            x =x + self._position_econding_(x)
            
        x = x + self.attention(self.norm1(x))
        x = x + self.fnn(self.norm2(x))
        return x
    
    def forward_postnorm(self, 
                x: torch.Tensor,
                if_position: bool=True)->torch.Tensor:
        
        # Use post norm 
        if if_position:
            x = x + self._position_econding_(x)
            
        x = self.norm1(x+self.attention(x))
        x = self.norm2(x + self.fnn(x))
        return x
        
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    emb_dim = 240
    seq_len = 240
    kv_head_num = 4
    q_head_num = 12
    x = torch.randn(2, seq_len, emb_dim)
    
    block = EncoderBlock(emb_dim, kv_head_num, q_head_num)
    position = block._position_econding_(x)
    print(position.shape)
    
    plt.imshow(position)
    plt.savefig(r"src\pre-norm_VS_post-norm\abs_position.png")
    plt.close()
    
    y = block.forward_postnorm(x)
    print(y.shape)
    
        
        
        
        
        