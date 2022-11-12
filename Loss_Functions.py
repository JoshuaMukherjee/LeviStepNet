import torch

def max_loss(pressure, true):
  if len(true.shape) > 1:
    m = torch.max((true-pressure)**2,1).values
  else:
    m = torch.max((true-pressure)**2,0).values
  return torch.sum(m,0)

def mse_loss(expected, found):
  return torch.nn.MSELoss()(expected,found)