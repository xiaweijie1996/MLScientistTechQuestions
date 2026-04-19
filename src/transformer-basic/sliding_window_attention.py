import copy
import torch


seq_len = 5
wind_size = 2

casual_mask = torch.tril(torch.ones(seq_len, seq_len))
causal_mask_slide = torch.ones_like(casual_mask)
print("Casual Mask:", casual_mask)
for i in range(casual_mask.shape[0]):
    for j in range(casual_mask.shape[1]):
        if i-j <= wind_size:
            pass
        else:
            causal_mask_slide[i,j]=0
print("Casual Mask:", causal_mask_slide)


i = torch.arange(seq_len).unsqueeze(1)   # rows
j = torch.arange(seq_len).unsqueeze(0)   # cols
causal_mask_slide = ((j <= i) & (i - j < wind_size)).float()

print(causal_mask_slide)

