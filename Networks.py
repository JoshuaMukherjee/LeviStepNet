from re import A
from numpy import zeros_like
import torch
from torch.nn import Module
from SymmetricFunctions import SymMax



class PointNet(Module):
    def __init__(self, layer_sets,input_size=3, 
                    activation=torch.nn.SELU, 
                    kernel=1, kernel_pad="same",padding_mode="zeros",
                    batch_norm=None,batch_args={},
                    sym_function=SymMax, sym_args={},
                    output_funct=None, output_funct_args = {},
                    output_layers = None, out_activation=torch.nn.SELU,output_layers_args={},
                    out_batch_norm=None, out_batch_norm_args={}

                    ):
        super(PointNet,self).__init__()
        self.layers = []
        self.sym_function = sym_function(**sym_args)
        if output_funct is not None:
            self.output_funct = output_funct(**output_funct_args)
        else:
            self.output_funct = None
        self.NN= False

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
                
                mod = torch.nn.Conv1d(in_channels,out_channels,kernel_size=kernel,padding=kernel_pad,padding_mode=padding_mode)
                
                self.layers[i].append(mod)
                
                if type(activation) is not list:
                    self.layers[i].append(activation())
                else:
                    act = activation[i][j]
                    if act is not None:
                        self.layers[i].append(act())
                
                if batch_norm is not None:
                    if type(batch_norm) is not list:
                        self.layers[i].append(batch_norm(out_channels,**batch_args))
                    else:
                        norm = batch_norm[i][j]
                        if act is not None:
                            self.layers[i].append(norm(out_channels,**batch_args[i][j]))
            
        if output_layers is not None: #option to add MLP to output
            self.output_NN = True
            self.layers.append(torch.nn.ModuleList())

            in_channel = layer_sets[-1][-1]
            out_channel = output_layers[0]
            self.layers[-1].append(torch.nn.Linear(in_channel,out_channel))
            if type(out_activation) is not list:
                self.layers[-1].append(out_activation())
            else:
                act = out_activation[0]
                self.layers[-1].append(act())
            
            if out_batch_norm is not None:
                if type(out_batch_norm) is not list:
                    self.layers[-1].append(out_batch_norm(out_channel,**out_batch_norm_args))
                else:
                    norm = out_batch_norm[i+1]
                    self.layers[-1].append(out_batch_norm(out_channel,**out_batch_norm_args))
                    

            for i,layer in enumerate(output_layers[0:-1]):
                in_channel = layer
                out_channel = output_layers[i+1]
                self.layers[-1].append(torch.nn.Linear(in_channel,out_channel,**output_layers_args))

                if type(out_activation) is not list:
                    self.layers[-1].append(out_activation())
                else:
                    act = out_activation[i+1]
                    self.layers[-1].append(act())
                if out_batch_norm is not None:
                    if type(out_batch_norm) is not list:
                        self.layers[-1].append(out_batch_norm(out_channel,**out_batch_norm_args))
                    else:
                        norm = out_batch_norm[i+1]
                        self.layers[-1].append(out_batch_norm(out_channel,**out_batch_norm_args))
   
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
        
        if self.NN is True:
            for layer in self.layers[3]:
                out = layer(out)
        
        if self.output_funct is not None:
           out = self.output_funct(out)
        return out



class MLP(Module):
    def __init__(self, layers, input_size=512, activation=torch.nn.SELU, batch_norm=None,batch_args={},layer_args={}):
        super(MLP,self).__init__()
        self.layers = torch.nn.ModuleList()
        
        in_channels= input_size
        out_channels = layers[0]
        self.layers.append(torch.nn.Linear(in_channels,out_channels,**layer_args))
        if type(activation) is not list and activation is not None:
                self.layers.append(activation())
        elif type(activation) is list :
            self.layers.append(activation[0]())

        if type(batch_norm) is not list and batch_norm is not None:
            self.layers.append(batch_norm(out_channels,**batch_args))
        elif type(batch_norm) is list :
            self.layers.append(batch_norm[0](out_channels,**batch_args))
       
        for i,layer in enumerate(layers[1:]):
            in_channels = layers[i]
            out_channels = layer
            self.layers.append(torch.nn.Linear(in_channels,out_channels,**layer_args))

            if type(activation) is not list and activation is not None:
                self.layers.append(activation())
            elif type(activation) is list :
                self.layers.append(activation[i+1]())
            
            if type(batch_norm) is not list and batch_norm is not None:
                self.layers.append(batch_norm(out_channels,**batch_args))
            elif type(batch_norm) is list :
                self.layers.append(batch_norm[i+1](out_channels,**batch_args))
        
        print(self.layers)
    
    def forward(self,x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

if __name__ == "__main__":

    # layers = [100,200,400,300]
    # act = [torch.nn.ReLU, torch.nn.GELU, torch.nn.SELU,torch.nn.Sigmoid]
    # bn = torch.nn.BatchNorm1d
    # net = MLP(layers,activation=act,batch_norm=bn)

    # point = torch.ones((2,512))
    # print(point)
    # net(point)



    from Dataset import *
    from torch.utils.data import DataLoader 
    from SymmetricFunctions import SymSum

    m = 512
    layers = [[64,64],[64,128,1024],[512,256,128,128,m]]
    output_layers = [10,20]
    norm = torch.nn.BatchNorm1d
    net = PointNet(layers,batch_norm=norm,out_batch_norm=norm)

    data = TimeDataset(5,3)
    points = DataLoader(data,2,shuffle=True)
    changes = next(iter(points))[1]
    perm = permute_points(changes,[0,2,1,3],axis=3)

    for i in range(1,changes.shape[1]):
        #itterate over timestamps
        change = changes[:,i,:,:] #Get batch
        out = net(change)
        print(out)
        print(out.shape)

        permed = perm[:,i,:,:]
        perm_out = net(change)
        print(torch.all(perm_out == out))
        print(torch.all(permed==change))

        rand = torch.rand_like(change)
        rand_out = net(rand)
        print(torch.all(rand_out == out))
