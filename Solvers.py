from Utlilities import *
import torch

def wgs(A, b, K):
    #Written by Giorgos Christopoulos 2022
    AT = torch.conj(A).T.to(device)
    b0 = b.to(device)
    x = torch.ones(A.shape[1],1).to(device) + 0j
    for kk in range(K):
        y = torch.matmul(A,x)                                   # forward propagate
        y = y/torch.max(torch.abs(y))                           # normalize forward propagated field (useful for next step's division)
        b = torch.multiply(b0,torch.divide(b,torch.abs(y)))     # update target - current target over normalized field
        b = b/torch.max(torch.abs(b))                           # normalize target
        p = torch.multiply(b,torch.divide(y,torch.abs(y)))      # keep phase, apply target amplitude
        r = torch.matmul(AT,p)                                  # backward propagate
        x = torch.divide(r,torch.abs(r))                        # keep phase for hologram  
                    
    return y, p, x



def gspat(R,forward, backward, target, iterations):
    #Written by Giorgos Christopoulos 2022
    field = target 

    for _ in range(iterations):
        
#     amplitude constraint, keeps phase imposes desired amplitude ratio among points     
        target_field = torch.multiply(target,torch.divide(field,torch.abs(field)))  
#     backward and forward propagation at once
        field = torch.matmul(R,target_field)
#     AFTER THE LOOP
#     impose amplitude constraint and keep phase, after the iterative part this step is different following Dieg
    target_field = torch.multiply(target**2,torch.divide(field,torch.abs(field)**2))
#     back propagate 
    complex_hologram = torch.matmul(backward,target_field)
#     keep phase 
    phase_hologram = torch.divide(complex_hologram,torch.abs(complex_hologram))
    points = torch.matmul(forward,phase_hologram)

    return phase_hologram, points


def naive(points):
    activation = torch.ones(points.shape[1]) +0j
    forward = forward_model(points.T,transducers())
    back = torch.conj(forward).T
    trans = back@activation
    trans_phase=  trans / torch.abs(trans)
    out = forward@trans_phase
    pressure = torch.abs(out)
    return out, pressure