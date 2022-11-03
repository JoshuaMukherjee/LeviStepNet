import torch
from sklearn import StandardScaler


def AE_vac(x):
    x = x.permute(0,2,1).contiguous()
    x = torch.view_as_complex(x)
    return x

def AE_var(x):
    x = torch.view_as_real(x)
    x = x.permute(0,2,1)
    return x

def stack_to_vector(x):
    real = AE_var(x)
    vect = torch.reshape(real,(x.shape[0],-1))
    return vect

def vector_to_mat(x):
    mat = torch.reshape(x,(x.shape[0],2,-1)) #in the wrong order so vac's permute still works
    comp = AE_vac(mat)
    return comp

def scale_stack_to_vector(x):
    real = AE_var(x)
    vect = torch.reshape(real,(x.shape[0],-1))
    scaler = StandardScaler()
    arr_norm = scaler.fit_transform(vect.numpy())
    return arr_norm.from_numpy()




if __name__ == "__main__":
    from torch.utils.data import DataLoader 

    import os,sys
    p = os.path.abspath('.')
    sys.path.insert(1, p)

    from Dataset import *


    data = TimeDataset(5,4)
    points = DataLoader(data,2,shuffle=True)

    a,b, activations ,d = next(iter(points))
    act = activations[:,0,:,:][:,:,0]
    print(act)
    print(act.shape)
    real = stack_to_vector(act)
    print(real.shape)
    comp = vector_to_mat(real)
    print(comp.shape)
    print(torch.all(comp == act))