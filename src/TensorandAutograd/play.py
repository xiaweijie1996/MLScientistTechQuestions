import torch


# torch softmax function is not stable, it can cause overflow when the input is large. So we need to implement a stable softmax function.

softmax = torch.nn.Softmax(dim=-1)   

data = torch.tensor([1.0, 1000.0, 1000.0])
print(softmax(data))    # this will cause overflow and return NaN

def naive_softmax(x):
    exp_x = torch.exp(x)
    sum_exp_x = torch.sum(exp_x)
    return exp_x / sum_exp_x

print(naive_softmax(data))    # this will cause overflow and return NaN


#　if data is batch data, we need to specify the dimension to apply softmax function. For example, if data is of shape (batch_size, num_classes), we need to specify dim=1 to apply softmax function on the num_classes dimension.
data = torch.tensor([[1.0, 1000.0, 1000.0], [1.0, 2.0, 3.0]])
print(softmax(data))    # this will return a valid probability distribution
# sum along the num_classes dimension should be 1
print(torch.sum(softmax(data), dim=-1))   # this will return a tensor

def stable_softmax(x):
    max_x = torch.max(x, dim=-1, keepdim=True).values   # get the max value of each row, keepdim=True to keep the dimension for broadcasting
    exp_x = torch.exp(x - max_x)   # subtract the max value from the input to prevent overflow
    sum_exp_x = torch.sum(exp_x, dim=-1, keepdim=True)   # sum along the num_classes dimension, keepdim=True to keep the dimension for broadcasting
    return exp_x / sum_exp_x
    
    
print(stable_softmax(data))   # this will return a valid probability distribution
# sum along the num_classes dimension should be 1
print(torch.sum(stable_softmax(data), dim=-1))   # this will return a tensor of ones

# Log stoft max 

# pytorch one
log_softmax = torch.nn.LogSoftmax(dim=-1)
print(log_softmax(data))    # this will return a valid log probability distribution

def stable_log_softmax(x):
    max_x = torch.max(x, dim =-1 , keepdim=True).values
    x = x - max_x
    log_sum_exp_x = torch.log(torch.sum(torch.exp(x), dim=-1, keepdim=True))
    return x - log_sum_exp_x
print(stable_log_softmax(data))    # this will return a valid log probability distribution
    