import torch
from torch.utils.data import Dataset
from Utlilities import *
from Solvers import wgs



class TimeDataset(Dataset):
    '''
    outputs points, changes, activations, pressures
    '''
    def __init__(self,length,timestamps,N=4,threshold = 0.01,seed=None,sort_fun=None):
        self.length = length #Number of point sets in the Dataset (length)
        self.timestamps = timestamps #number of changes in an element
        self.N = N #Number of points per set
        self.seed = seed #custom seed
        self.sort_fun = sort_fun #allows sorting the points within a set
        self.threshold = threshold #The ammount a point can change by +/-
    
        self.points = []
        self.changes = []
        self.activations = []
        self.pressures = []
        print(self.length)
        for batch in range(self.length):
            p = torch.FloatTensor(3,self.N).uniform_(-.06,.06).to(device)
            changes = torch.FloatTensor(self.timestamps,3,self.N).uniform_(-1*self.threshold,self.threshold).to(device)
            changes[0,:,:] = 0
            time_series = torch.zeros_like(changes)
            change = 0
            for j,dxyz in enumerate(changes):
                change += dxyz
                time_series[j] = p + change
            
            # self.points.append(time_series.permute(0,2,1)) 
            # self.changes.append(changes.permute(0,2,1))

            self.points.append(time_series) 
            self.changes.append(changes)
            

            pressures = torch.zeros((self.timestamps,self.N)) + 0j
            activations = torch.zeros(self.timestamps,512,1) + 0j
            
            for i,points in enumerate(time_series):
                A=forward_model(points, transducers()).to(device)
                _, _, x = wgs(A,torch.ones(self.N,1).to(device)+0j,200)
                pressures[i] = A@x[:,0]
                activations[i] = x
            self.pressures.append(pressures)
            self.activations.append(activations)

            if batch % 200 == 0:
                print(batch,end=" ",flush=True)
    
    def __len__(self):
        return self.length

    def __getitem__(self,i):
         return self.points[i], self.changes[i],self.activations[i],self.pressures[i]


if __name__ == "__main__":
    timestamps = 2
    length = 120000
    test_length = 0
    N = 4
    train = TimeDataset(length,timestamps,N=N)
    torch.save(train,"Datasets/Train-"+str(timestamps)+"-"+str(length)+"-"+str(N)+".pth")

    if test_length > 0:
        test = TimeDataset(test_length,timestamps,N=N)
        torch.save(test,"Datasets/Test-"+str(timestamps)+"-"+str(test_length)+"-"+str(N)+".pth")
    
   
