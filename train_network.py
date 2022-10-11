import torch
import Dataset
import time
import pickle
from Utlilities import *


def train(net, start_epochs, epochs, train, test, optimiser, loss_function, supervised, scheduler, name, batch ):
    start_time = time.asctime()
    losses = []
    losses_test = []


    try:
        for epoch in range(epochs):
            #TRAINING
            running = 0
            loss = 0
            # net.train()
            for training_dataset in train:
                for points, changes, activations, pressures in iter(training_dataset):                    
                    initial_point = points[:,0,:,:]
                    # net.init(initial_point)
                    
                    for i in range(1,changes.shape[1]): #itterate over timestamps
                        change = changes[:,i,:,:] #Get batch
                        optimiser.zero_grad()
                        
                        pressure = net(changes)
                        pressure = 0
                        pressure_out = torch.abs(propagate(pressure,points[:,i,:,:]))
                        
                        if supervised:
                            loss = loss_function(pressure_out,pressures[:,i,:,:])
                        else:
                            loss = loss_function(pressures[:,i,:,:])
                        
                        running += loss.item()

                        loss.backward() #Change for TBPPTT
                        optimiser.step()
                        loss = 0
            
            if scheduler is not None:
                if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                    scheduler.step(running)
                else:
                    scheduler.step()
            
            #TESTING
            net.eval()
            running_test = 0
            for test_dataset in test:
                for points, changes, activations, pressures in iter(test_dataset):
                    initial_point = points[:,0,:,:]
                    # net.init(initial_point)
                    for i in range(1,changes.shape[1]): #itterate over timestamps
                        pressure = net(changes)
                        pressure = 0
                        pressure_out = torch.abs(propagate(pressure,points))
                        
                        if supervised:
                            loss = loss_function(pressure_out,pressures)
                        else:
                            loss = loss_function(activations)
                            
                        running_test += loss.item()
            
            losses.append(running) #Store each epoch's losses 
            losses_test.append(running_test)

            print(epoch+start_epochs,"Training",running, "Testing", running_test, "Time", time.asctime(), "Start", start_time)
            torch.save(net, 'SpamModels/model_' + str(name) + '.pth')
            loss_to_dump = (losses, losses_test)
            pickle.dump(loss_to_dump, open("SpamLoss/loss_"+ str(name) +'.pth',"wb"))

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    from torch.utils.data import DataLoader 
    dataset = Dataset.TimeDataset(2,4)
    train(None,1,[DataLoader(dataset,2,shuffle=True)],None,None,None)