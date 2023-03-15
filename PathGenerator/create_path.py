import os,sys
p = os.path.abspath('.')
sys.path.insert(1, p)
import torch
import json

from Utlilities import *
from Solvers import wgs

import matplotlib.pyplot as plt
import matplotlib
import itertools

matplotlib.use('tkagg')

file = sys.argv[1]
path = sys.argv[2]
shuffle = "-s" in sys.argv
wgs_reset = "-w" in sys.argv
FRAME_THRESHOLD = 10

params = json.load(open("PathGenerator/Paths/"+file+".json","r"))

positions = torch.FloatTensor(params["positions"])
if params["format"] == "cm":
    positions /= 100


def interpolate(start,end, step_size,shuffle=False):
    difference = end - start 
    
    steps = torch.Tensor.int(torch.ceil(difference / step_size)) #May cause off my one - doesnt matter for now
    N = int((torch.sum(torch.abs(steps))).item())
    changes = torch.zeros(N,start.shape[0],start.shape[1])
    M = 0
    for pi,point in enumerate(steps):
        for di,diff in enumerate(point):
            for i in range(int(torch.abs(diff).item())):             
                change = torch.zeros_like(start)
                direction = torch.sign(diff)
                change[pi,di] = step_size*direction
                changes[M,:,:] = change
                M += 1
    
    if shuffle:
        idxs = torch.randperm(changes.shape[0])
        changes = changes[idxs,:,:]

    return changes



net = torch.load("SavedModels/"+path+"/"+"model_"+path+".pth",map_location=device)

N=positions.shape[1] #No. Points
start = positions[0].T
A=forward_model(start, transducers()).to(device)
_, _, act_init = wgs(A,torch.ones(N,1).to(device)+0j,200)

net.init(act_init.T)
point = torch.unsqueeze(start,0)

pressures = [torch.abs(propagate(act_init.T,point)).detach().numpy()]
resets = [[],[]]

f = open("PathGenerator/Experiments/"+file+path+".csv","w")
f.write("XXXXXX\n")
frames = 1
trans_n = "512"


init_phases_ud = convert_pats(torch.angle(act_init).T)

for i,phase in enumerate(init_phases_ud[0,:]):
        f.write(str(phase.item()))
        if i < 511:
            f.write(",")
        else:
            f.write("\n")

a, b = itertools.tee(positions)
next(b, None)
since_reset = 0
for start, end in zip(a, b):
    changes = interpolate(start,end,0.0001,shuffle=shuffle)
    for change in changes:
        change = torch.unsqueeze(change,0)
        change = torch.permute(change,(0,2,1))
        point += change

        activation_out = net(change) 
        phases = torch.angle(activation_out)
        pressures.append(torch.abs(propagate(activation_out,point)).detach().numpy())

        phases_converted = convert_pats(phases)

        for i,phase in enumerate(phases_converted[0]):
            f.write(str(phase.item()))
            if i < 511:
                f.write(",")
            else:
                f.write("\n")
        frames +=1

        if wgs_reset:
            since_reset+=1
            if since_reset == FRAME_THRESHOLD:
                p = torch.squeeze(torch.permute(point,(0,1,2)))
                A=forward_model(p, transducers()).to(device)
                _, _, act_init = wgs(A,torch.ones(N,1).to(device)+0j,200)
                net.init(act_init.T)
                pres =torch.abs(propagate(act_init.T,point)).detach()
                mean_p = torch.mean(pres)
                pres = pres.numpy()
                pressures.append(pres)
                resets[0].append(frames)
                resets[1].append(mean_p)
                since_reset = 0

                phases_converted = convert_pats(torch.angle(act_init).T)

                for i,phase in enumerate(phases_converted[0]):
                    f.write(str(phase.item()))
                    if i < 511:
                        f.write(",")
                    else:
                        f.write("\n")
                frames+=1


        
        
f.seek(0)
f.write(str(frames)+","+trans_n+"\n")        
f.close()





if "-p" in sys.argv:
    pressures = torch.tensor(pressures)
   

    for i in range(pressures.shape[1]):
        plt.plot(pressures[:,i],label="point "+str(i))
    
    plt.plot(resets[0],resets[1],"x",label="WGS re-initialisations")
    
    plt.legend()
    plt.xlabel("Frame")
    plt.ylabel("Pressure (Pa)")
    plt.yticks(torch.linspace(0,10000,11).numpy())
    plt.ylim(bottom=0)
    plt.show()

##Animation? https://matplotlib.org/stable/gallery/animation/random_walk.html