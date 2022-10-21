from torch.nn import Module
import torch

def identity_func(x):
    return x


class AutoencoderNet(Module):

    def __init__(self,encoder,decoder, in_func = identity_func, out_func = identity_func):
        super(AutoencoderNet,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        self.in_func = in_func
        self.out_func = out_func

    
    def forward(self,x):
        x = self.in_func(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return self.out_func(x)

        