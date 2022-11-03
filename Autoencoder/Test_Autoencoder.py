from enum import auto
from Autoencoder import AutoencoderNet
from Autoencoder_funcs import stack_to_vector

import torch
import os,sys

from torch.utils.data import DataLoader 

p = os.path.abspath('.')
sys.path.insert(1, p)

from Networks import MLP
from Autoencoder import AutoencoderNet
from Train_Autoencoder import train
from Utlilities import device
from Dataset import TimeDataset


if len(sys.argv) > 1:
    path = sys.argv[1]
    autoencoder = torch.load(path,map_location=torch.device(device))
else:
    print("Enter Path")
    exit()

data = TimeDataset(5,4)
points = DataLoader(data,2,shuffle=True)
_,_,act,_ = next(iter(points))
act = act[:,0,:,:][:,:,0]
out = autoencoder(act)

print(torch.view_as_real(act))
print(torch.view_as_real(out))
print(torch.view_as_real(act) - torch.view_as_real(out))
print(torch.nn.MSELoss()(torch.view_as_real(act),torch.view_as_real(out)))
