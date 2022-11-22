
import json
import traceback
import torch
from torch.utils.data import DataLoader


from Train_Network import train
import Networks, Updater, Loss_Functions
from Utlilities import device
from Dataset import TimeDataset, TimeDatasetAtomic


files = [
    "Updater15"
]


def parse(params,name):

    start_epochs = params["start-epochs"]
    epochs = params["epochs"]

    if start_epochs == 0:
        net = getattr(Networks, params["net"])(**params["net-args"]).to(device)
        encoder = getattr(Networks, params["encoder"])(**params["encoder-args"]).to(device)
        updater = getattr(Updater,params["updater"])(net,encoder).to(device)
    else:
        updater = torch.load(name+".pth",map_location=torch.device(device))


    train_s = [torch.load("./Datasets/"+pth,map_location=torch.device(device)) for pth in params["train"] ]
    test_s =  [torch.load("./Datasets/"+pth,map_location=torch.device(device)) for pth in params["test"]  ]

    batch = params["batch"]
    train_sets = [DataLoader(d,batch,shuffle=True) for d in train_s]
    test_sets = [DataLoader(d,batch,shuffle=True) for d in test_s]

    optimiser = getattr(torch.optim, params["optimiser"])(updater.parameters(),**params["optimiser-args"])
    loss_function = getattr(Loss_Functions, params["loss-function"])
    supervised = params["supervised"]
   
    if "scheduler" in params:
        scheduler = getattr(torch.optim.lr_scheduler, params["scheduler"])(optimiser,**params["scheduler-args"])
    else:
        scheduler = None
    
    batch = params["batch"]


    train(updater,start_epochs,epochs,train_sets,test_sets,optimiser,loss_function, supervised, scheduler, name, batch)


for file in files:
    try:
        print(file, "Parsing....")
        params = json.load(open("Params/"+file+".json","r"))
        parse(params,file)
    except Exception as e:
        print(traceback.format_exc())


