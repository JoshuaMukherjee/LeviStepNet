import os,sys
import torch,pickle
from torch.utils.data import DataLoader 
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

p = os.path.abspath('.')
sys.path.insert(1, p)

from Utlilities import *
from Updater import Updater
from Networks import PointNet, MLP
from Dataset import TimeDataset, TimeDatasetAtomic
from Solvers import gspat, wgs

args = sys.argv
BOXPLOTS = False
LOSS = False
HELP = False
MAX_LOSS = False
ACTIVATIONS = False
TIME = False
SIGNED = True
OLD_MOVE = False
PHASES = False
P_32 = False
P_16 = False




try:
    path = args[1]
    if "-latest" not in args:
        net = torch.load("SavedModels/"+path+"/"+"model_"+path+".pth",map_location=device)
    else:
        net = torch.load("SavedModels/"+path+"/"+"model_"+path+"_latest.pth",map_location=device)
    BOXPLOTS = "-p" in args
    LOSS = "-l" in args
    HELP = "-h" in args
    MAX_LOSS = "-m" in args
    ACTIVATIONS = "-a" in args
    TIME = "-t" in args
    SIGNED = not "-unsigned" in args
    DETAILED = "-d" in args
    NORM = "-n" in args
    OLD_MOVE = "-oldmove" in args
    P_32 = "-32" in args
    P_16 = "-16" in args
    P_8 = "-8" in args
    PHASES = "-ph" in args


    if OLD_MOVE:
        movement = 0.005
    else:
        movement = 0.0001
    
    print(movement)
    
    point_count=4
    if P_32:
        point_count = 32
    if P_16:
        point_count = 16
    if P_8:
        point_count = 8
    print(point_count)

except IndexError:
    print("Invalid Arguments")

if BOXPLOTS:
    if DETAILED:
        N = 4  #Number of Plots
    else:
        N = 8
    dataset = TimeDatasetAtomic(N,2,N=point_count,signed=SIGNED,movement=movement)
    data = iter(DataLoader(dataset,1,shuffle=True))


    fig1, f1_axes = plt.subplots(ncols=int(N/2), nrows=2, constrained_layout=True)
    for dim in f1_axes:
        for ax in dim:

            to_plot = {}
            p,c,a,pr = next(data)
            activation_init = a[:,0,:]
            net.init(activation_init)
            change = c[:,1,:,:] #Get batch
            activation_out = net(change) 
            pressure_out = torch.abs(propagate(activation_out,p[:,1,:])).detach().numpy()
            to_plot["Network"] = list(pressure_out)


            points = p[0,1,:]
            A = forward_model(points)
            backward = torch.conj(A).T
            R = A@backward
            _,pres = gspat(R,A,backward,torch.ones(point_count,1).to(device)+0j, 200)
            gs_pat_200 = torch.abs(pres)
            gs_pat_200 = [p.item() for p in gs_pat_200]
            to_plot["GS-PAT-200"] = list(gs_pat_200)
            if DETAILED:
                _,pres = gspat(R,A,backward,torch.ones(point_count,1).to(device)+0j, 20)
                gs_pat_20 = torch.abs(pres)
                gs_pat_20 = [p.item() for p in gs_pat_20]
                to_plot["GS-PAT-20"] = list(gs_pat_20)

                _,pres = gspat(R,A,backward,torch.ones(point_count,1).to(device)+0j, 5)
                gs_pat_5 = torch.abs(pres)
                gs_pat_5 = [p.item() for p in gs_pat_5]
                to_plot["GS-PAT-5"] = list(gs_pat_5)

            _, _, x = wgs(A,torch.ones(point_count,1).to(device)+0j,200)
            wgs_200 = torch.abs(A@x[:,0])
            to_plot["WGS-200"] = list(wgs_200)
            if DETAILED:
                _, _, x = wgs(A,torch.ones(point_count,1).to(device)+0j,20)
                wgs_20 = torch.abs(A@x[:,0])
                to_plot["WGS-20"] = list(wgs_20)

                _, _, x = wgs(A,torch.ones(point_count,1).to(device)+0j,5)
                wgs_5 = torch.abs(A@x[:,0])
                to_plot["WGS-5"] = list(wgs_5)

            print("-"*10)
            for i in to_plot:
                print(i,end=" ")
                for val in to_plot[i]:
                    print(float(val),end=" ")
                print()
                print("mean", np.mean(to_plot[i]),end=" ")
                print("sd", np.std(to_plot[i]))
            ax.boxplot(to_plot.values())
            ax.set_xticklabels(to_plot.keys())
            ax.set_ylim(bottom=0,top=13000)

            

        
    plt.show()




if LOSS:
    loss = pickle.load(open("SavedModels/"+path+"/"+"loss_"+path+".pth","rb"))
    train,test = loss
    if NORM:
        train = train/np.linalg.norm(train)
        test = test/np.linalg.norm(test)
    # train = np.abs(train)
    # test = np.abs(test)
    plt.plot(train,label="train")
    plt.plot(test,label="test")
    plt.yscale("symlog")
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
    N = 50
    dataset = TimeDatasetAtomic(N,2,N=point_count,signed=SIGNED,movement=movement)
    data = iter(DataLoader(dataset,1,shuffle=True))
    

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

    
if ACTIVATIONS:
    N = 5
    dataset = TimeDatasetAtomic(N,2,N=point_count,signed=SIGNED,movement=movement)
    data = iter(DataLoader(dataset,1,shuffle=True))
    



    p,c,a,pr = next(data)
    activation_init = a[:,0,:]
    net.init(activation_init)
    change = c[:,1,:,:] #Get batch
    activation_out = net(change).detach().numpy()
    a = a.detach().numpy()
    print(activation_out.shape)
    print(a[:,1,:].shape)


    x = [i.real for i in activation_out]
    y = [i.imag for i in activation_out]

    x_a = [i.real for i in a[:,1,:]]
    y_a = [i.imag for i in a[:,1,:]]


    plt.plot(x,y,"o",color="blue")
    plt.plot(x_a,y_a,"o",color="red")
    plt.xlabel("Real")
    plt.ylabel("Imag")
    plt.show()

if TIME:
    N = 1
    T = 40
    dataset = TimeDatasetAtomic(N,T,N=point_count,signed=SIGNED,movement=movement)
    data = iter(DataLoader(dataset,1,shuffle=True))
    means = []
    stds = []
    points = []
    for i in range(point_count):
        points.append([])
    for p,c,a,pr in data:
        activation_init = a[:,0,:]
        net.init(activation_init)
        for i in range(1,c.shape[1]):
            change = c[:,i,:,:] #Get batch
            activation_out = net(change)
            pressure_out = torch.abs(propagate(activation_out,p[:,i,:]))
            for i,point in enumerate(pressure_out):
                points[i].append(point.item())       
            means.append(torch.mean(pressure_out))
            stds.append(torch.std(pressure_out))
    means = [m.detach().numpy() for m in means]
    stds =  [s.detach().numpy() for s in stds]
    plt.xlabel("Time")
    # plt.plot(means,label="Mean")
    plt.plot(stds,label="Standard Deviation")
    for i,point in enumerate(points):
        plt.plot(point,label="point "+str(i))
    plt.legend()
    plt.show()

if PHASES:
    N = 2
    T = 5
    dataset = TimeDatasetAtomic(N,T,N=point_count,signed=SIGNED,movement=movement)
    data = iter(DataLoader(dataset,1,shuffle=True))

    to_disp = []
    p,c,a,pr = next(data)
    activation_init = a[:,0,:]
    net.init(activation_init)
    last = activation_init
    
    min_dif = 0
    max_dif = torch.pi
    for i in range(T):
        change = c[:,i,:,:] #Get change batch
        activation_out = net(change)
        diff = torch.angle(activation_out) - torch.angle(last)
        diff = torch.abs(torch.atan2(torch.sin(diff),torch.cos(diff)))
       

        diff = torch.reshape(diff,(2,16,16))
        to_disp.append(diff)
        last = activation_out
    print(min_dif,max_dif)
    gs = matplotlib.gridspec.GridSpec(2,T+1)

    for i,act in enumerate(to_disp):
        ax_up = plt.subplot(gs[0, i])
        ax_down = plt.subplot(gs[1, i])
        mshw = ax_up.matshow(act[0].detach().numpy(),vmin=min_dif, vmax=max_dif)
        ax_down.matshow(act[1].detach().numpy(),vmin=min_dif, vmax=max_dif)
        ax_up.set_axis_off()
        ax_down.set_axis_off()
       
    
    colour_ax = plt.subplot(gs[0:2,T])
    # plt.colorbar(mshw, cax=colour_ax)
    plt.colorbar(mshw, cax=colour_ax,ticks=[0,torch.pi])
    colour_ax.set_yticklabels(["0","π"])


    plt.show()
    

      
    

    

if HELP:
    print("-h, help")
    print("-p, pressures")
    print("-l, losses")
    print("-m, max loss")
    print("-a, activations")