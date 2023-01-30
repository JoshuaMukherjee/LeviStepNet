import torch
import torch.nn.functional as F
import torch.nn as nn
import Dataset
import time
import pickle
import random
from Symmetric_Functions import SymSum
from Utlilities import *



def do_network(net, optimiser,loss_function,loss_params, datasets,test=False, 
                supervised=True, scheduler = None, random_stop=False, clip=False,
                clip_args={}, amp_reg_function = None, amp_reg_lambda = 0):
    #TRAINING
    running = 0
    if not test:
        net.train()
    else:
        net.eval()

    
    for dataset in datasets:
        for points, changes, activations, pressures in iter(dataset):
            if not test:
                optimiser.zero_grad()            
            activation_init = activations[:,0,:]
            net.init(activation_init)
            
            if random_stop:
                rand_len = random.randint(1,changes.shape[1])
            
            optimiser.zero_grad()            
        
            end = 0
            outputs = []
            amp_reg_val = 0
            for i in range(1,changes.shape[1]): #iterate over timestamps - Want timestamps-1 iterations because first one is zeros  
                change = changes[:,i,:,:] #Get batch Bxtx3xN
                activation_out = net(change)

                if random_stop and (i == rand_len):
                    break
                end = i

                field = propagate(activation_out,points[:,i,:])
                pressure_out = torch.abs(field)
                outputs.append(pressure_out)

                if amp_reg_function is not None:
                    phases_out = torch.angle(activation_out)
                    phases_target = torch.angle(activations[:,i,:])
                    amp_reg_val += amp_reg_lambda * amp_reg_function(phases_out,phases_target )

            output = torch.stack(outputs,dim=1) #compare to torch.abs(pressures[:,1:,:])
            target = torch.abs(pressures[:,1:,:])
           
            # if supervised:
            #     loss = loss_function(pressure_out,torch.abs(pressures[:,end,:]),**loss_params)
            # else:
            #     loss = loss_function(pressure_out,**loss_params)

            if supervised:
                loss = loss_function(output,target,**loss_params) + amp_reg_val
            else:
                loss = loss_function(output,**loss_params) + amp_reg_val
            
            
          
                
            running += loss.item()
            grad = None
            if not test:
                loss.backward()
                if clip:
                    grads = [torch.sum(p.grad) for n, p in net.named_parameters()]
                    grad = sum(grads)/len(grads)
                    nn.utils.clip_grad_norm_(net.parameters(), **clip_args)
                    # print({n:p.grad for n, p in net.named_parameters()}["encoder.layers.5.weight"])
                optimiser.step()
    if not test:
        if scheduler is not None:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(running)
            else:
                scheduler.step()
    return running, grad



def train(net, start_epochs, epochs, train, test, optimiser, 
            loss_function, loss_params, supervised, scheduler, name, 
            batch, random_stop, clip=False, clip_args={}, log_grad =False,
            amp_reg_function=None, amp_reg_lambda=None ):
    print(name, "Training....")
    start_time = time.asctime()
    losses = []
    losses_test = []
    best_test = torch.inf

    try:   
        for epoch in range(epochs):
            #Train
            running , grad= do_network(net, optimiser, loss_function, loss_params, train, scheduler=scheduler, supervised=supervised,random_stop=random_stop, clip=clip, clip_args=clip_args, amp_reg_function=amp_reg_function, amp_reg_lambda=amp_reg_lambda )
            #Test
            running_test, _ = do_network(net, optimiser, loss_function, loss_params, test, test=True, supervised=supervised, amp_reg_function=amp_reg_function, amp_reg_lambda=amp_reg_lambda)
            
            losses.append(running) #Store each epoch's losses 
            losses_test.append(running_test)

            print(name,epoch+start_epochs,"Training",running,"Testing",running_test,"Time",time.asctime(),"Start",start_time, end=" ")
            if log_grad:
                print("grad",grad, end = " ")
            
            # if supervised: #what is this for?
            #     running_test = torch.abs(running_test)
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
