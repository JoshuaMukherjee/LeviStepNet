import os,sys
import torch,pickle
from torch.utils.data import DataLoader 

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

p = os.path.abspath('.')
sys.path.insert(1, p)

from Utlilities import *
from Updater import Updater
from Networks import PointNet, MLP
from Dataset import TimeDataset

args = sys.argv
BOXPLOTS = False
LOSS = False
HELP = False

try:
    path = args[1]
    net = torch.load("SavedModels/"+path+"/"+"model_"+path+".pth",map_location=device)
    BOXPLOTS = "-p" in args
    LOSS = "-l" in args
    HELP = "-h" in args


except IndexError:
    print("Invalid Arguments")

if BOXPLOTS:
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


    pressures = [p.detach().numpy() for p in pressures]

    print(pressures)
    plt.boxplot(pressures)
    plt.title("Pressures at points")
    plt.xlabel("Point")
    plt.ylabel("P")
    plt.show()

if LOSS:
    loss = pickle.load(open("SavedModels/"+path+"/"+"loss_"+path+".pth","rb"))
    train,test = loss
    plt.plot(train,label="train")
    plt.plot(test,label="test")
    plt.yscale("log")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()


if HELP:
    print("-h, help")
    print("-p, boxplots")
    print("-l, losses")