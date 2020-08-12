import numpy as np
import os
import sys
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import time
import datetime
import matplotlib.pyplot as plt
from torchvision.models import densenet121,densenet169,densenet201


class MyDataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self,index):
        X=torch.from_numpy(self.X[index]).double()#(in_channel,depth,height,width)
        Y=torch.from_numpy(np.array(self.Y[index])).long()
        return X,Y

def conv3D_output_size(img_size,kernel_size, stride,padding):
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape

def spacing():
    print('\n')
    print('='*70)
    print('='*70) 

class DenseNet():

    def __init__(self,train_data,dev_data):
        
        #train_data
        self.train_data=train_data
        #dev_data
        self.dev_data=dev_data

        #(1 question)
        self.get_num_classes()
        self.get_pretrained_model()
        print(self.net)         # printing model architecture

        self._get_batchsize_input()             # getting a batch size for training and validation

        self._get_loss_function()               # getting a loss function

        self._get_optimizer()               # getting an optimizer input

        self._get_scheduler()               # getting a scheduler input

        self._set_device()              # setting the device to gpu or cpu

        self._get_epoch()           # getting an input for number oftraining epochs


        self.dataloader()

        self.main()             # run function

    def get_num_classes(self):
        print('Question [1/7]: No of classes:')
        print('\n')
        gate = 0
        while gate!= 1:
            self.num_classes = int(input('Please enter the number of classes for classification: '))
            if int(self.num_classes) >1 :
                gate =1
            else:
                print('Please enter a valid input')
        spacing()
    
    def get_pretrained_model(self):
        print('Question [2/7]: Model Selection:')
        print('\n')
        self.pretrained_dict={1:densenet121,2:densenet169,3:densenet201}
        
        gate=0
        while gate!=1:
            pretrained_input=input('Do you want pretrained model? (y/n): ')
            if pretrained_input.lower()=='y':
                self.pretrained=True
                gate=1
            elif pretrained_input.lower()=='n':
                self.pretrained=False
                gate=1
            else:
                print('Please enter valid input')
        
        gate=0
        while gate!=1:
            self.model_select=int(input('Please enter any number between 1 to 3 to select the model:\
                                        \n[1:DenseNet121,2:DenseNet169,3:DenseNet201] \n'))

            if (1<= self.model_select <=4):
                model=self.pretrained_dict[self.model_select](pretrained=True)
                gate=1
            else:
                print('Please enter valid input')
        
        # print(self.train_data[0][0].shape[0])
        model.features[0]=nn.Conv2d(self.train_data[0][0].shape[0], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if self.model_select==1:
            model.classifier=nn.Linear(1024,self.num_classes)
        elif self.model_select==2:
            model.classifier=nn.Linear(1664,self.num_classes)
        else:
            model.classifier=nn.Linear(1920,self.num_classes)

        self.net=model.double()

        spacing()





    def _get_batchsize_input(self):
        print('Question [3/7]: Batchsize:')
        print('\n')
        # Method for getting batch size input
        gate = 0
        while gate!= 1:
            self.batch_size = int(input('Please enter the batch size: '))
            if int(self.batch_size) >0 :
                gate =1
            else:
                print('Please enter a valid input')
        spacing() 

    def _set_device(self):
        if torch.cuda.is_available():
            self.device=torch.device('cuda')
            self.cuda=True
        else:
            self.device=torch.device('cpu')
            self.cuda=False
    
    def _get_loss_function(self):
        print('Question [4/7]: Loss function:')
        print('\n')
        # Method for getting a loss function for training
        self.criterion_list = {1:nn.CrossEntropyLoss(),2:torch.nn.L1Loss(),3:torch.nn.SmoothL1Loss(),4:torch.nn.MSELoss()}
        
        gate = 0
        while gate!= 1:
            self.criterion_input = (input('Please enter the appropriate loss function for the problem: \n Criterion_list - [1: CrossEntropyLoss, 2: L1Loss, 3: SmoothL1Loss, 4: MSELoss]: '))

            if self.criterion_input.isnumeric() and int(self.criterion_input) < 5 and int(self.criterion_input)> 0:
                gate =1
            else:
                print('Please enter a valid input')
        spacing()
            
        self.criterion = self.criterion_list[int(self.criterion_input)]

    def dataloader(self):

        #train and dev loader
        train_dataset = MyDataset(self.train_data[:,0],self.train_data[:,1])

        train_loader_args = dict(shuffle=True, batch_size=self.batch_size)

        self.train_loader = data.DataLoader(train_dataset, **train_loader_args)

        
        dev_dataset = MyDataset(self.dev_data[:,0],self.dev_data[:,1])

        dev_loader_args = dict(shuffle=False, batch_size=self.batch_size)

        self.dev_loader = data.DataLoader(dev_dataset, **dev_loader_args)
    

    def _get_optimizer(self):
        print('Question [5/7]: Optimizer:')
        print('\n')

        # Method for getting a optimizer input

        gate = 0
        while gate!= 1:
            self.optimizer_input = (input('Please enter the optimizer for the problem \n Optimizer_list - [1: Adam, 2: SGD] \n For default optimizer, please directly press enter without any input: '))
            if self.optimizer_input == '':              # handling default case for optimizer
                print('Default optimizer selected')
                self.optimizer_input = '1'

            if self.optimizer_input.isnumeric() and int(self.optimizer_input) >0  and int(self.optimizer_input) < 3:
                gate =1
            else:
                print('Please enter a valid input')
        spacing()
                
        gate = 0
        while gate!= 1:
            self.user_lr = input('Please enter a required postive value for learning rate \n For default learning rate, please directly press enter without any input: ')
            if self.user_lr == '':               # handling default case for learning rate
                print('Default value selected')
                self.user_lr = '0.001'
            if float(self.user_lr) > 0:
                gate = 1
            else:
                print('Please enter a valid input')
        spacing()

        self.lr = float(self.user_lr)
        self.optimizer_list = {1:optim.Adam(self.net.parameters(),lr = self.lr), 2:optim.SGD(self.net.parameters(),lr = self.lr)}
        self.optimizer = self.optimizer_list[int(self.optimizer_input)]


#scheduler
    def _get_scheduler(self):
        print('Question [6/7]: Scheduler:')
        print('\n')

        # Method for getting scheduler

        gate = 0
        while gate!= 1:
            self.scheduler_input = input('Please enter the scheduler for the problem: Scheduler_list - [1: None, 2:StepLR, 3:MultiStepLR] \n For default option of no scheduler, please directly press enter without any input: ')
            if self.scheduler_input == '':
                print('By default no scheduler selected')
                self.scheduler_input = '1'
            if self.scheduler_input.isnumeric() and int(self.scheduler_input) >0  and int(self.scheduler_input) <4:
                gate =1
            else:
                print('Please enter a valid input')
        
        if self.scheduler_input == '1':
            self.scheduler =  None

        elif self.scheduler_input == '2':
            self.step = int(input('Please enter a step value: '))
            print(' ')
            self.gamma = float(input('Please enter a gamma value (Multiplying factor): '))
            self.scheduler =  scheduler.StepLR(self.optimizer, step_size = self.step, gamma = self.gamma)

        elif self.scheduler_input == '3':
            self.milestones_input = (input('Please enter values of milestone epochs: '))
            self.milestones_input = self.milestones_input.split(',')
            self.milestones = [int(x) for x in self.milestones_input if int(x)>0]
            print(' ')
            self.gamma = float(input('Please enter a gamma value (Multiplying factor): '))
            self.scheduler =  scheduler.MultiStepLR(self.optimizer, milestones = self.milestones, gamma = self.gamma)
        
        spacing()
    
    def _get_epoch(self):
        print('Question [7/7]: Number of Epochs:')
        print('\n')

        # Method for getting number of epochs for training the model

        self.numEpochs = int(input('Please enter the number of epochs to train the model: '))
    
    def train_model(self):

        # Method for training the model

        self.net.train()
        self.training_loss = []
        self.training_acc = []
        self.dev_loss = []
        self.dev_accuracy = []
        total_predictions = 0.0
        correct_predictions = 0.0

        for epoch in range(self.numEpochs):
            
            start_time = time.time()
            self.net.train()
            print('Epoch_Number: ',epoch)
            running_loss = 0.0
            

            for batch_idx, (data, target) in enumerate(self.train_loader):   

                self.optimizer.zero_grad()   
                data = data.to(self.device)
                target = target.to(self.device) 

                outputs = self.net(data)

                if self.criterion_input == '1':             # calculating the batch accuracy only if the loss function is Cross entropy

                    loss = self.criterion(outputs, target.long())
                    _, predicted = torch.max(outputs.data, 1)
                    total_predictions += target.size(0)
                    correct_predictions += (predicted == target).sum().item()

                else:

                    loss = self.criterion(outputs, target)
                
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()
        
            
            running_loss /= len(self.train_loader)              
            self.training_loss.append(running_loss)
            print('Training Loss: ', running_loss)

            if self.criterion_input == '1':             # printing the epoch accuracy only if the loss function is Cross entropy
                
                acc = (correct_predictions/total_predictions)*100.0
                self.training_acc.append(acc)
                print('Training Accuracy: ', acc, '%')

            dev_loss,dev_acc = self.validate_model()
    
            if self.scheduler_input != '1':

                self.scheduler.step()
                print('Current scheduler status: ', self.optimizer)
            
            end_time = time.time()
            print( 'Epoch Time: ',end_time - start_time, 's')
            print('#'*50)

            self.dev_loss.append(dev_loss)

            if self.criterion_input == '1':             # saving the epoch validation accuracy only if the loss function is Cross entropy
                
                self.dev_accuracy.append(dev_acc)
    

    def validate_model(self):

        with torch.no_grad():
            self.net.eval()
        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0
        acc = 0

        for batch_idx, (data, target) in enumerate(self.dev_loader): 

            data = data.to(self.device)
            target = target.to(self.device)
            outputs = self.net(data)

            if self.criterion_input == '1':

                loss = self.criterion(outputs, target.long())
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += target.size(0)
                correct_predictions += (predicted == target).sum().item()

            else:
                loss = self.criterion(outputs, target)
            running_loss += loss.item()


        running_loss /= len(self.dev_loader)
        print('Validation Loss: ', running_loss)

        if self.criterion_input == '1':             # calculating and printing the epoch accuracy only if the loss function is Cross entropy
            
            acc = (correct_predictions/total_predictions)*100.0
            print('Validation Accuracy: ', acc, '%')
        
        return running_loss,acc

    def get_loss_graph(self):

        # Method for showing and saving the loss graph in the root directory

        plt.figure(figsize=(8,8))
        plt.plot(self.training_loss,label='Training Loss')
        plt.plot(self.dev_loss,label='Validation Loss')
        plt.legend()
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.savefig('loss.png')
    
    def get_accuracy_graph(self):

        # Method for showing and saving the accuracy graph in the root directory

        plt.figure(figsize=(8,8))
        plt.plot(self.training_acc,label='Training Accuracy')
        plt.plot(self.dev_accuracy,label='Validation Accuracy')
        plt.legend()
        plt.title('Model accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('acc')
        plt.savefig('accuracy.png') 

    def get_prediction(self, x_input):

        # Method to use at the time of inference

        if len(x_input.shape) == 1:             # handling the case of single input

            x_input = (x_input).reshape(1,-1)
            
        x_input = torch.from_numpy(x_input).to(self.device)

        net_output = self.net.predict(x_input)

        if self.criterion_input == '1':             # handling the case of classification problem

             _, net_output = torch.max(net_output.data, 1)

        return net_output
    
    def get_model_summary(self):

        # Method for getting the summary of the model
        print('Model Summary:')
        print(' ')
        print('Criterion: ', self.criterion)
        print('Optimizer: ', self.optimizer)
        print('Scheduler: ', self.scheduler)
        print('Batch size: ', self.batch_size)
        print('Initial learning rate: ', self.lr)
        print('Number of training epochs: ', self.numEpochs)
        print('Device: ', self.device)

        spacing()

    
    def main(self):

        # Method integrating all the functions and training the model

        self.net.to(self.device)

        self.get_model_summary()        # printing summaray of the model

        self.train_model()          # training the model

        self.get_loss_graph()           # saving the loss graph

        if self.criterion_input == '1':

            self.get_accuracy_graph()           # saving the accuracy graph

        self._save_model()              # saving model paramters

    def _save_model(self):

        # Method for saving the model parameters if user wants to

        gate=0
        save_model = input('Do you want to save the model weights? (y/n): ')
        while gate  != 1:
            if save_model.lower() =='y' or save_model.lower() =='yes':
                path = 'model_parameters.pth'
                torch.save(self.net.state_dict(),path)
                gate = 1
            elif save_model.lower() =='n' or save_model.lower() =='no':
                gate = 1
            else:
                print('Please enter a valid input')
