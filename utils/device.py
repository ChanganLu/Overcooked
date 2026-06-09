import torch

use_cuda = torch.cuda.is_available()
device_type = 'cuda' if use_cuda else 'cpu'
autocast_dtype = torch.float16 if use_cuda else torch.bfloat16
grad_init_scale = 2 ** (12 if use_cuda else 16)

device = torch.device('cuda:0' if use_cuda else 'cpu')
autocast = torch.amp.autocast(device_type=device_type, dtype=autocast_dtype)
grad_scaler = torch.amp.GradScaler(device=device_type, init_scale=grad_init_scale)
