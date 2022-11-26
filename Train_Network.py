import torch
import Dataset
import time
import pickle
from Symmetric_Functions import SymSum
from Utlilities import *


def do_network(net, optimiser,loss_function, datasets,test=False, supervised=True, scheduler = None):
    #TRAINING
    running = 0
    loss = 0
    if not test:
        net.train()
    else:
        net.eval()
    for dataset in datasets:
        for points, changes, activations, pressures in iter(dataset):                    
            activation_init = activations[:,0,:]
            net.init(activation_init)
            
            for i in range(1,changes.shape[1]): #itterate over timestamps - Want timestamps-1 iterations because first one is zeros
                change = changes[:,i,:,:] #Get batch
                if not test:
                    optimiser.zero_grad()
                
                activation_out = net(change)
                pressure_out = torch.abs(propagate(activation_out,points[:,i,:]))
                if supervised:
                    loss = loss_function(pressure_out,torch.abs(pressures[:,i,:]))
                else:
                    loss = loss_function(torch.abs(pressures[:,i,:]))
                
                running += loss.item()
                if not test:
                    loss.backward()
                    optimiser.step()
    if not test:
        if scheduler is not None:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(running)
            else:
                scheduler.step()
    return running



def train(net, start_epochs, epochs, train, test, optimiser, loss_function, supervised, scheduler, name, batch ):
    print(name, "Training....")
    start_time = time.asctime()
    losses = []
    losses_test = []
    best_test = torch.inf


    try:   
        for epoch in range(epochs):
            #Train
            running = do_network(net, optimiser, loss_function, train, scheduler=scheduler, supervised=supervised)
            #Test
            running_test = do_network(net, optimiser, loss_function, test, test=True, supervised=supervised)
            
            losses.append(running) #Store each epoch's losses 
            losses_test.append(running_test)

            print(name,epoch+start_epochs,"Training",running,"Testing",running_test,"Time",time.asctime(),"Start",start_time, end=" ")
            if running_test < best_test: #Only save if the best 
                net.epoch_saved = epoch
                torch.save(net, 'Models/model_' + str(name) + '.pth')
                best_test = running_test
                print("SAVED")
            else:
                print()
            torch.save(net, 'Models/model_' + str(name) + '_latest.pth') #save the newest model too
            loss_to_dump = (losses, losses_test)
            pickle.dump(loss_to_dump, open("Losses/loss_"+ str(name) +'.pth',"wb"))

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    from torch.utils.data import DataLoader 
    from Updater import Updater
    from Networks import MLP, PointNet
    from Symmetric_Functions import SymSum
    dataset = Dataset.TimeDataset(2,4)


    mlp = MLP([10,4])
    m = 512
    layers = [[64,64],[64,128,1024],[512,256,128,128,m]]
    output_layers = [10,20]
    norm = torch.nn.BatchNorm1d
    network = PointNet(layers,batch_norm=norm,out_batch_norm=norm,output_funct=SymSum, input_size=7)
    updater = Updater(network, mlp)
    
    train(updater,1,[DataLoader(dataset,2,shuffle=True)],[DataLoader(dataset,2,shuffle=True)],None,None)
