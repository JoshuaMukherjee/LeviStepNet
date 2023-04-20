import os,sys
p = os.path.abspath('.')
sys.path.insert(1, p)
import torch
import json

from Utlilities import *
from Solvers import wgs, temporal_wgs

import matplotlib.pyplot as plt
import matplotlib
import itertools

matplotlib.use('tkagg')

file = sys.argv[1]
path = sys.argv[2]
shuffle = "-s" in sys.argv
wgs_reset = "-w" in sys.argv
temp_reset = "-t" in sys.argv
mean = "-m" in sys.argv
if wgs_reset and temp_reset:
    raise Exception("Can only use one of -w and -t")


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

    # print(changes.shape)
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
f.write("X"*100)
f.write("\n")
frames = 1
trans_n = "512"

FLIP_INDEX = get_convert_indexes()


init_phases = torch.angle(act_init)
init_phases_ud = init_phases[FLIP_INDEX]

for i,phase in enumerate(init_phases_ud):
        f.write(str(phase.item()))
        if i < 511:
            f.write(",")
        else:
            f.write("\n")

a, b = itertools.tee(positions)
next(b, None)
since_reset = 0
phase_frames = [torch.mean(torch.angle(act_init))]
for start, end in zip(a, b):
    print(frames)
    changes = interpolate(start,end,0.0001,shuffle=shuffle)
    for change in changes:
        change = torch.unsqueeze(change,0)
        change = torch.permute(change,(0,2,1))
        point += change

        activation_out = net(change) 
        phases = torch.angle(activation_out)
        phase_frames.append(torch.mean(torch.abs(torch.angle(act_init))))
        
        p_prop = propagate(activation_out,point)
        pressures.append(torch.abs(p_prop).detach().numpy())
       
        p = phases
        phases_converted = phases.T[FLIP_INDEX]

        for i,phase in enumerate(phases_converted):
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
                
                phase_frames.append(torch.mean(torch.abs(torch.angle(act_init))))
                
                pres =torch.abs(propagate(act_init.T,point)).detach()
                mean_p = torch.mean(pres)
                pres = pres.numpy()
                pressures.append(pres)
                resets[0].append(frames)
                resets[1].append(mean_p)
                since_reset = 0

                phases = torch.angle(act_init).T
                phases_converted = phases[FLIP_INDEX]

                for i,phase in enumerate(phases_converted):
                    f.write(str(phase.item()))
                    if i < 511:
                        f.write(",")
                    else:
                        f.write("\n")
                frames+=1
        
        if temp_reset:
            since_reset+=1
            if since_reset == FRAME_THRESHOLD:
                p = torch.squeeze(torch.permute(point,(0,1,2)))
                A=forward_model(p, transducers()).to(device)
                
                T_in=torch.pi/8
                T_out = 0

                act = activation_out.T
                prop = torch.unsqueeze(p_prop,1)

                _, _, act_init = temporal_wgs(A,torch.ones(N,1).to(device)+0j,200, act, prop,T_in,T_out)
                net.init(act_init.T)
                phase_frames.append(torch.mean(torch.abs(torch.angle(act_init))))
                
                pres =torch.abs(propagate(act_init.T,point)).detach()
                mean_p = torch.mean(pres)
                pres = pres.numpy()
                pressures.append(pres)
                resets[0].append(frames)
                resets[1].append(mean_p)
                since_reset = 0

                phases = torch.angle(act_init)
                phases_converted = phases[FLIP_INDEX]


                for i,phase in enumerate(phases_converted):
                    f.write(str(phase.item()))
                    if i < 511:
                        f.write(",")
                    else:
                        f.write("\n")
                frames+=1


        
        
f.seek(0)
f.write(str(frames)+","+trans_n+",")        
f.close()





if "-p" in sys.argv:
    pressures = torch.tensor(pressures)

    # phase_frames = [torch.atan2(torch.sin(i),torch.cos(i)) for i in phase_frames]
    phase_changes = []
    for i in range(1,len(phase_frames)):
        phase_changes.append((phase_frames[i] - phase_frames[i-1]).detach().numpy())
   
    ax = plt.subplot(2,1,1)
    if not mean:
        for i in range(pressures.shape[1]):
            ax.plot(pressures[:,i],label="point "+str(i))
    else:
        ax.plot(torch.mean(pressures,dim=1))
    
    if wgs_reset or temp_reset:
        ax.plot(resets[0],resets[1],"x",label="Re-initialisations")
    
    ax.legend()
    ax.set_xlabel("Frame")
    ax.set_ylabel("Pressure (Pa)")
    # m = round(torch.min(pressures).item(), -3) - 1000
    # ax.set_yticks(torch.linspace(m,10000,11).numpy())

    ax.set_yticks(torch.linspace(0,10000,11).numpy())

    ax = plt.subplot(2,1,2)
    ax.plot(phase_changes[1:])
    ax.set_ylabel("Mean Absolute Phase Change (Rad)")
    ax.set_xlabel("Frame")

    length = len(phase_changes)
    phase_changes_reset_frames = [phase_changes[i] for i in resets[0] if i < length]
    x_resets = [r for r in resets[0] if r < length]

    # ax.plot(x_resets,phase_changes_reset_frames,"x",label="Re-initialisations")
    # if temp_reset:
    #     ax.plot(torch.linspace(0,frames,frames),torch.ones(frames)*T_in,label="T_in limit = pi/" + str(int(torch.pi/T_in)),color="green")
    #     ax.plot(torch.linspace(0,frames,frames),torch.ones(frames)*-1*T_in,color="green")
    # ax.legend()

    # plt.ylim(bottom=0)
    plt.show()

##Animation? https://matplotlib.org/stable/gallery/animation/random_walk.html