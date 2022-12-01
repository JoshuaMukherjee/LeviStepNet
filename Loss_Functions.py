import torch

def max_loss(pressure, true):
  if len(true.shape) > 1:
    m = torch.max((true-pressure)**2,1).values
  else:
    m = torch.max((true-pressure)**2,0).values
  return torch.sum(m,0)

def mse_loss(expected, found):
  return torch.nn.MSELoss()(expected,found)

def mean_std(output,alpha=-0.01):
  if len(output.shape) > 1:
    dim = 1
  else:
    dim = 0
  m = torch.mean(output,dim) + alpha*torch.std(output,dim) 
  return torch.sum(m,0)
  