import torch
from torch.utils.data import Dataset
from Utlilities import *
from Solvers import wgs


class TimeDataset(Dataset):
    def __init__(self,length,timestamps,N=4,threshold = 0.01,seed=None,sort_fun=None):
        self.length = length #Number of point sets
        self.timestamps = timestamps #number of changes in an element
        self.N = N #Number of points per set
        self.seed = seed #custom seed
        self.sort_fun = sort_fun #allows sorting the points within a set
        self.threshold = threshold #The ammount a point can change by +/-
    
        self.points = []
        self.changes = []
        self.activations = []
        self.pressures = []

        for i in range(self.length):
            p = torch.FloatTensor(self.N,3).uniform_(-.06,.06).to(device)
            changes = torch.FloatTensor(self.timestamps,self.N,3).uniform_(-1*self.threshold,self.threshold).to(device)
            changes[0,:,:] = 0
            time_series = torch.zeros_like(changes)
            change = 0
            for i,dxyz in enumerate(changes):
                change += dxyz
                time_series[i] = p + change
            
            self.points.append(time_series)
            self.changes.append(changes)
            

            pressures = torch.zeros((self.timestamps,self.N))
            activations = torch.zeros(self.timestamps,512,1)
            
            for i,points in enumerate(time_series):
                A=forward_model(points, transducers()).to(device)
                _, _, x = wgs(A,torch.ones(self.N,1).to(device)+0j,200)
                pressures[i] = A@x[:,0]
                activations[i] = x
            self.pressures.append(pressures)
            self.activations.append(activations)

            if i % 100 == 0:
                print(".",end="",flush=True)
    
    def __len__(self):
        return self.length

    def __getitem__(self,i):
         pass


if __name__ == "__main__":
    data = TimeDataset(2,3)
