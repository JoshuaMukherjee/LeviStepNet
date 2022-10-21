import torch
from torch.nn import Module


class SineActivation(Module):

    def __init__(self):
        super(SineActivation,self).__init__()
    
    def forward(self,x):
        return torch.sin(x)