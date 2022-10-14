import torch

def max_loss(pressure, true):
  m = torch.max((true-pressure)**2,1).values
  return torch.sum(m,0)