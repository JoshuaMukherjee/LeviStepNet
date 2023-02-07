import torch
from torch.nn import Module
from Symmetric_Functions import SymMax
from Utlilities import device

import Activations



class PointNet(Module):
    def __init__(self, layer_sets,input_size=3, 
                    activation=torch.nn.ReLU, 
                    kernel=1, kernel_pad="same",padding_mode="zeros",
                    batch_norm=None,batch_args={},
                    sym_function=SymMax, sym_args={},
                    output_funct=None, output_funct_args = {}
                ):
        super(PointNet,self).__init__()
        self.layers = torch.nn.ModuleList()
        self.sym_function = sym_function(**sym_args)
        if output_funct is not None:
            self.output_funct = output_funct(**output_funct_args)
        else:
            self.output_funct = None
        
        if type(batch_norm) == str:
            batch_norm = getattr(torch.nn,batch_norm)
        
        if type(activation) == str:
            activation = getattr(torch.nn,activation) 

        local_features = layer_sets[0][-1]
        for i,layer_set in enumerate(layer_sets):
            self.layers.append(torch.nn.ModuleList())
            for j,layer in enumerate(layer_set):
                if i == 0 and j == 0:
                    in_channels = input_size
                    out_channels = layer
                elif j != 0:
                    in_channels = layer_set[j-1]
                    out_channels = layer
                elif i == 2 and j == 0:
                    in_channels = layer_sets[i-1][-1]+local_features
                    out_channels = layer
                else:
                    in_channels = layer_sets[i-1][-1]
                    out_channels = layer
                
                mod = torch.nn.Conv1d(in_channels,out_channels,kernel_size=kernel,padding=kernel_pad,padding_mode=padding_mode).to(device)
                
                self.layers[i].append(mod)
                
                if type(activation) is not list:
                    self.layers[i].append(activation().to(device))
                else:
                    act = activation[i][j]
                    if act is not None:
                        self.layers[i].append(act().to(device))
                
                if batch_norm is not None:
                    if type(batch_norm) is not list:
                        self.layers[i].append(batch_norm(out_channels,**batch_args).to(device))
                    else:
                        norm = batch_norm[i][j]
                        if act is not None:
                            self.layers[i].append(norm(out_channels,**batch_args[i][j]).to(device))
            
        # print(self.layers)

    def forward(self, x):
        out = x
        for layer in self.layers[0]:
            out = layer(out)
        local_features = out
        for layer in self.layers[1]:
            out = layer(out)
        
        out = self.sym_function(out)
        N = x.shape[2]
        global_features = torch.Tensor.expand(out.unsqueeze_(2),-1,-1,N)
        out = torch.cat((local_features,global_features),dim=1)
        for layer in self.layers[2]:
            out = layer(out)
        
        
        if self.output_funct is not None:
            out = self.output_funct(out)
        

        return out



class MLP(Module):
    def __init__(self, layers, input_size=512,layer_args={},
                    activation=torch.nn.SELU, batch_norm=None, batch_args={},batch_channels=2,batch_old=False):
        super(MLP,self).__init__()
        self.layers = torch.nn.ModuleList()
        
        in_channels= input_size
        out_channels = layers[0]

        if type(batch_norm) == str:
            batch_norm = getattr(torch.nn,batch_norm)
        


        self.layers.append(torch.nn.Linear(in_channels,out_channels,**layer_args).to(device))
        
        if type(activation) is not list and activation is not None:
            if type(activation) == str:
                try:
                    activation = getattr(torch.nn,activation) 
                except AttributeError:
                    activation = getattr(Activations,activation) 
            self.layers.append(activation().to(device))
        elif type(activation) is list :
            if type(activation[0]) == str:
                try:
                    act = getattr(torch.nn,activation[0]) 
                except AttributeError:
                    act = getattr(Activations,activation[0]) 
            self.layers.append(act().to(device))

        if type(batch_norm) is not list and batch_norm is not None:
            if batch_old:
                channel = batch_channels
            else:
                channel = out_channels
            self.layers.append(batch_norm(channel,**batch_args).to(device))
        elif type(batch_norm) is list :
            self.layers.append(batch_norm[0](channel,**batch_args).to(device))
       
        for i,layer in enumerate(layers[1:]):
            in_channels = layers[i] #As starting from [1:] in layers i will be actually one off from position
            out_channels = layer
            self.layers.append(torch.nn.Linear(in_channels,out_channels,**layer_args).to(device))

            if type(activation) is not list and activation is not None:
                if type(activation) == str:
                    try:
                        activation = getattr(torch.nn,activation) 
                    except AttributeError:
                        activation = getattr(Activations,activation) 

                self.layers.append(activation().to(device))
            elif type(activation) is list :
                if type(activation[i+1]) == str:
                    try:
                        activation = getattr(torch.nn,activation[i+1]) 
                    except AttributeError:
                        activation = getattr(Activations,activation[i+1]) 
                
                self.layers.append(activation().to(device))
            
            if type(batch_norm) is not list and batch_norm is not None:
                if batch_old:
                    channel = batch_channels
                else:
                    channel = out_channels
                self.layers.append(batch_norm(channel,**batch_args).to(device))
            elif type(batch_norm) is list :
                self.layers.append(batch_norm[i+1](channel,**batch_args).to(device))
            
    def forward(self,x):
        out = x
        # print()
        for layer in self.layers:
            out = layer(out)
        return out


class ResBlock(Module):
    def __init__(self, D, D1, D2, 
                kernel=1, kernel_pad="same",padding_mode="zeros",
                activation = None, norm = None):
        super(ResBlock, self).__init__()

        if type(norm) == str:
            norm = getattr(torch.nn,norm)
        
        if type(activation) == str:
            activation = getattr(torch.nn,activation) 

        self.block1 = torch.nn.Conv1d(D , D1, kernel_size=kernel,padding=kernel_pad,padding_mode=padding_mode).to(device)
        
        if activation is not None:
            self.act = activation().to(device)
        else:
            self.act = None
        
        self.block1_norm = norm(D1)

        self.block2 = torch.nn.Conv1d(D1, D2, kernel_size=kernel,padding=kernel_pad,padding_mode=padding_mode).to(device)
    

        self.block3 = torch.nn.Conv1d(D , D2, kernel_size=kernel,padding=kernel_pad,padding_mode=padding_mode).to(device)
        self.block3_norm =  norm(D2)
        
    
    def forward(self,x):
        x1 = self.block1(x)
        if self.act is not None:
            x1 = self.act(x1)
        x1 = self.block1_norm(x1)

        x1 = self.block2(x1)
        x = self.block3(x)
        x = self.block3_norm(x)

        x = x+x1
        if self.act is not None:
            x = self.act(x)

        return x

class ResPointNet(Module):
    def __init__(self, layer_sets, input_size=3,
                kernel=1, kernel_pad="same",padding_mode="zeros",
                activation = None, norm = None,
                sym_function=SymMax, sym_args={}):
        super(ResPointNet,self).__init__()

        self.sym_function = sym_function(**sym_args)


        self.blocks = torch.nn.ModuleList()
        D = input_size
        assert(len(layer_sets) == 3)

        local_features = layer_sets[0][-1]

        for layer_i,layer in enumerate(layer_sets):
            self.blocks.append(torch.nn.ModuleList())
            for i in range(0,len(layer),2):
                D1 = layer[i]
                D2 = layer[i+1]
                if layer_i == 2 and i ==0:
                    D += local_features
                block = ResBlock(D,D1,D2,  kernel=kernel, kernel_pad=kernel_pad,padding_mode=padding_mode,  activation = activation, norm = norm)
                self.blocks[layer_i].append(block)
                D = D2
                
    def forward(self,x):
        out = x
        for layer in self.blocks[0]:
            out = layer(out)
        local_features = out
        for layer in self.blocks[1]:
            out = layer(out)
        
        out = self.sym_function(out)
        N = x.shape[2]
        global_features = torch.Tensor.expand(out.unsqueeze_(2),-1,-1,N)
        out = torch.cat((local_features,global_features),dim=1)
        for layer in self.blocks[2]:
            out = layer(out)
        

        return out


if __name__ == "__main__":

    act = "ELU"
    bn = torch.nn.BatchNorm1d
    # net = ResPointNet([[10,20,30,40],[50,60],[70,80]],activation=act,norm=bn)
    m = 512
    layers = [[64,64],[64,128],[512,256,128,m]]
    norm = torch.nn.BatchNorm1d
    net = ResPointNet(layers,norm=norm)
    # print(net)



    from Dataset import *
    from torch.utils.data import DataLoader 
    from Symmetric_Functions import SymSum

    # m = 512
    # layers = [[64,64],[64,128,1024],[512,256,128,128,m]]
    # norm = torch.nn.BatchNorm1d
    # net = PointNet(layers,batch_norm=norm)
    print(net)

    data = TimeDatasetAtomic(5,4)
    points = DataLoader(data,2,shuffle=True)
    points,changes,_,_ = next(iter(points))
    perm = permute_points(changes,[0,2,1,3],axis=3)

    for i in range(1,changes.shape[1]): #Want timestampts-1 iterations because first one is zeros
        #itterate over timestamps
        change = changes[:,i,:,:] #Get batch
        out = net(change)
        p=points[:,i,:,:]
        # print(swap_output_to_activations(out,p))
        print(out)
        print(out.shape)

        permed = perm[:,i,:,:]
        perm_out = torch.sum(net(permed))
        print(torch.all(perm_out == torch.sum(out))) #Should be True
        # print(torch.all(permed==change)) 

        rand = torch.rand_like(change)
        rand_out = net(rand)
        print(torch.all(rand_out == out)) #Should be False
