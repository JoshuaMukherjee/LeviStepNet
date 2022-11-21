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
MAX_LOSS = False

try:
    path = args[1]
    net = torch.load("SavedModels/"+path+"/"+"model_"+path+".pth",map_location=device)
    BOXPLOTS = "-p" in args
    LOSS = "-l" in args
    HELP = "-h" in args
    MAX_LOSS = "-m" in args

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

    try:
        max_epoch = net.epoch_saved
        plt.plot(max_epoch,test[max_epoch],"x")
    except AttributeError:
        pass


    plt.show()

if MAX_LOSS:
    from Loss_Functions import max_loss
    dataset = TimeDataset(50,2)
    data = iter(DataLoader(dataset,1,shuffle=True))
    
    N = 50

    losses = []

    for i in range(N):
        p,c,a,pr = next(data)
        activation_init = a[:,0,:]
        net.init(activation_init)
        change = c[:,1,:,:] #Get batch
        activation_out = net(change)
        pressure_out = torch.abs(propagate(activation_out,p[:,1,:]))
        loss = max_loss(pressure_out,torch.abs(pr[:,1]))
        losses.append(loss)
    
    losses = [l.detach().numpy() for l in losses]

    print(losses)
    # plt.bar([i for i in range(len(losses))],losses)
    plt.hist(losses,bins=20)
    plt.title("Max loss Loss of pressure at points")
    plt.xlabel("Max Loss")
    plt.ylabel("Frequency")
    plt.show()

    


if HELP:
    print("-h, help")
    print("-p, pressures")
    print("-l, losses")
    print("-m", "max loss")