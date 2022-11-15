import os,sys
import torch
from torch.utils.data import DataLoader 

import matplotlib.pyplot as plt
import matplotlib


p = os.path.abspath('.')
sys.path.insert(1, p)

from Utlilities import *
from Updater import Updater
from Networks import PointNet, MLP
from Dataset import TimeDataset

args = sys.argv
try:
    path = args[1]
    net = torch.load("SavedModels/"+path+"/"+"model_"+path+".pth",map_location=device)
except IndexError:
    print("Invalid Arguments")

dataset = TimeDataset(10,2)
data = iter(DataLoader(dataset,1,shuffle=True))

N = 5

pressures = []

for i in range(N):
    p,c,a,pr = next(data)
    activation_init = a[:,0,:]
    net.init(activation_init)
    change = c[:,1,:,:] #Get batch
    activation_out = net(change)
    pressure_out = torch.abs(propagate(activation_out,p[:,1,:]))
    pressures.append(pressure_out)

pressure = [p.detach().numpy() for p in pressures]
print(pressures)
# plt.boxplot(pressures)