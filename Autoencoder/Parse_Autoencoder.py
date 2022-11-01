import sys,os
import json
import traceback

import torch
from torch.utils.data import DataLoader


p = os.path.abspath('.')
sys.path.insert(1, p)

from Networks import MLP
from Autoencoder import AutoencoderNet
from Train_Autoencoder import train
from Utlilities import device
from Dataset import TimeDataset

import Autoencoder_funcs
import Loss_Functions


def parse(params,file):
    encoder_args = params["encoder-args"]
    encoder = MLP(**encoder_args).to(device)
    decoder_args = params["decoder-args"]
    decoder = MLP(**decoder_args).to(device)

    in_func = getattr(Autoencoder_funcs, params["in_func"])
    out_func = getattr(Autoencoder_funcs, params["out_func"])

    autoencoder = AutoencoderNet(encoder,decoder,in_func,out_func).to(device)

    start_epochs = params["start-epochs"]
    epochs = params["epochs"]

    train_s = [torch.load("./Datasets/"+pth,map_location=torch.device(device)) for pth in params["train"] ]
    test_s =  [torch.load("./Datasets/"+pth,map_location=torch.device(device)) for pth in params["test"]  ]

    batch = params["batch"]
    train_sets = [DataLoader(d,batch,shuffle=True) for d in train_s]
    test_sets = [DataLoader(d,1,shuffle=True) for d in test_s]

    optimiser = getattr(torch.optim, params["optimiser"])(autoencoder.parameters(),**params["optimiser-args"])
    loss_function = getattr(Loss_Functions, params["loss-function"])
   
    if "scheduler" in params:
        scheduler = getattr(torch.optim.lr_scheduler, params["scheduler"])(optimiser,**params["scheduler-args"])
    else:
        scheduler = None


    train(autoencoder,start_epochs,epochs,train_sets,test_sets,optimiser,loss_function,scheduler,file,batch)


files = [
    "AE11"
]

if __name__ == "__main__":
    for file in files:
        try:
            params = json.load(open("./Autoencoder/Autoencoder_Params/"+file+".json","r"))
            parse(params,file)
        except Exception as e:
            print(traceback.format_exc())

















