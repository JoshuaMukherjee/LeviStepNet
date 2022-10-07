import torch
from torch.nn import Module



class PointNet(Module):
    def __init__(self, layer_sets,input_size=3, activation=torch.nn.SELU, kernel=1, kernel_pad="same",padding_mode="zeros",batch_norm=None,batch_args={},sym_function=torch.max):
        self.layers = []
        self.sym_function = sym_function

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

        print(self.layers)
        


if __name__ == "__main__":
    m = 512
    layers = [[64,64],[64,128,1024],[512,256,128,128,m]]
    norm = torch.nn.BatchNorm2d
    PointNet(layers,batch_norm=norm,batch_args={"momentum":100})