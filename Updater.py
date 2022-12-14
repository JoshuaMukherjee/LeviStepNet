from cmath import nan
from torch.nn import Module
import torch
import torch.nn.functional as F

from Utlilities import convert_to_complex, device


class Updater(Module):
    def __init__(self,network,encoder,constrain_amp=False,norm=False):
        super(Updater,self).__init__()

        self.network = network
        self.encoder = encoder

        self.memory = None
        self.points = None

        self.epoch_saved = 0

        self.constrain_amp = constrain_amp #older versions didnt have this - load will not include this
        self.norm = norm
    
    def init(self,start_activations):
        self.memory = start_activations

    def forward(self,changes):
        # self.memory.detach() #needed?
        if self.memory is None:
            raise Exception("memory not initialised")

        z = self.encoder(torch.abs(self.memory))
        N = changes.shape[2] 
        z_expand = torch.Tensor.expand(z.unsqueeze_(2),-1,-1,N)
        out = torch.cat((changes,z_expand),dim=1)
        if "norm" in self.__dict__ and self.norm:
            out = F.normalize(out)
        out = self.network(out) #1024 x N
        out = convert_to_complex(out)
        out = torch.sum(out,dim=2)
        self.memory = self.memory + out
        if "constrain_amp" in self.__dict__ and self.constrain_amp: #Backwards compatability - not very neat
            self.memory = self.memory / torch.abs(self.memory)
        return self.memory


if __name__ == "__main__":
    from Networks import MLP, PointNet
    from torch.utils.data import DataLoader 
    from Dataset import TimeDataset
    from Symmetric_Functions import SymSum
    from Utlilities import propagate
    
    mlp = MLP([10,4])
    m = 1024
    layers = [[64,64],[64,128,1024],[512,256,128,128,m]]
    output_layers = [10,20]
    norm = torch.nn.BatchNorm1d
    network = PointNet(layers,batch_norm=norm, input_size=7)
    updater = Updater(network, mlp)
    
    data = TimeDataset(2,4)
    points = DataLoader(data,2,shuffle=True)
    points,changes,activation,pressures = next(iter(points))
    updater.init(activation[:,0,:])

    for i in range(1,changes.shape[1]): #Want timestampts-1 iterations because first one is zeros
        change = changes[:,i,:,:] #Get batch
        x=updater(change)
        point = points[:,i,:,:]
        # print(torch.abs(pressures))

        print(torch.abs(propagate(x,point)))
        break
  