import torch

# A = torch.randint(0, 10, (10, 3))
# B = torch.randint(0, 3, (10,))

# C = torch.gather(A, 1, B.unsqueeze(1)).squeeze(1)
# D = A[torch.arange(0, 10), B]

# print(torch.cat([A, B.unsqueeze(1), C.unsqueeze(1), D.unsqueeze(1)], dim=1))

A = torch.randint(0, 10, (64, 6))
B = torch.argmax(A, dim=1)
print(A.shape, B.shape)
