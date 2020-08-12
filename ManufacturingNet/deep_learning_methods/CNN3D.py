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

class MyDataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self,index):
        X=torch.from_numpy(self.X[index]).double().unsqueeze(0)#(in_channel,depth,height,width)
        Y=torch.from_numpy(np.array(self.Y[index])).long()
        return X,Y

def conv3D_output_size(img_size,kernel_size, stride,padding):
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape

def spacing():
    print('\n')
    print('='*75)
    print('='*75) 

class BasicBlock(nn.Module):
    def __init__(self, in_channel,input_size,default =False):
        super(BasicBlock, self).__init__()

        self.input_size=input_size
        self.in_channel=in_channel
        self.get_channel_input()

        if default:
            self.kernel=(3,3,3)
            self.stride=(1,1,1)
            self.padding=(1,1,1)
            self.pooling_input=False
        else:        
            self.get_kernel_size()
            self.get_stride()
            self.get_padding()
            
        self.conv1=nn.Conv3d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.input_size=conv3D_output_size(self.input_size,self.kernel,self.stride,self.padding)
        self.batch=nn.BatchNorm3d(self.out_channel, momentum=0.01)
        
        if default==False:
            self.get_pooling_layer()
            if self.pooling_input:
                self.input_size=conv3D_output_size(self.input_size,(3,3,3),(2,2,2),(0,0,0))
        
        self.get_dropout()

    def get_pooling_layer(self):
        print('Question: Pooling layer for this conv:')
        print('\n')
        gate=0
        self.pooling_qtn= input('Do you want a pooling layer after this convolution layer (y/n): ')
        while gate!=1:
            if (self.pooling_qtn).lower()=='y':
                self.pooling_input=True
                self.pool=nn.MaxPool3d(kernel_size=3,stride=2) #kept constant for now
                gate=1

            elif (self.pooling_qtn).lower()=='n':
                self.pooling_input=False
                gate=1
            else:
                print('Please enter valid input')
        spacing()
    
    def get_dropout(self):
        print('Question: Dropout value for this conv:')
        print('\n')
        # Method for getting dropout input for each hidden layer
        gate = 0
        while gate != 1:
            self.dropout_value = input('Please enter the dropout value between 0 and 1 for this convolution. \n For default option of no dropout, please directly press enter without any input: ')
            if self.dropout_value == '':               
                print('Default value selected')
                self.drop=nn.Dropout3d(p=0)
                gate = 1
            elif 0<= float(self.dropout_value) <= 1:
                self.drop=nn.Dropout3d(p=float(self.dropout_value))
                gate=1
            else:
               print('Please enter valid input')
        spacing()
        
    def get_channel_input(self):
        print('Question: Out-channels for this conv:')
        print('\n')
        gate=0
        while gate!=1:
            channel_input=int(input('Please enter the number of out channels: '))
            if channel_input>0:
                self.out_channel=channel_input
                gate=1
            else:
                print('Please enter a valid input')
        spacing()
    
    def get_kernel_size(self):
        print('Question: Kernel Size for this conv')
        print('\n')
        gate=0
        while gate!=1:
            self.kernel=[]
            kernel_input=list(input('Please enter the kernel size as a list(depth,heigth,width)\nFor example 3,3,3\nFor default size, please directly press enter without any input: '))
            if len(kernel_input)==0:
                print('Default Value selected')
                self.kernel=tuple([3,3,3])
                gate=1
            elif len(kernel_input)==5:
                for i in range(len(kernel_input)):
                    if kernel_input[i]!=',' and (1<= int(kernel_input[i]) <= 20): 
                        self.kernel.append(int(kernel_input[i]))
                self.kernel=tuple(self.kernel)
                gate=1
            else:
                print('Please enter valid input')
        spacing()
                
            
    def get_stride(self):
        print('Question: Stride value for this conv:')
        print('\n')
        gate=0
        while gate!=1:
            self.stride=[]
            stride_input=list(input('Please enter the stride as a list(depth,heigth,width)\nFor example 1,1,1\nFor default size, please directly press enter without any input: '))
            if len(stride_input)==0:
                print('Default Value selected')
                self.stride=tuple([1,1,1])
                gate=1
            elif len(stride_input)==5:
                for i in range(len(stride_input)):
                    if stride_input[i]!=',' and (1<= int(stride_input[i]) <= 20): 
                        self.stride.append(int(stride_input[i]))
                self.stride=tuple(self.stride)
                gate=1
            else:
                print('Please enter valid input')
        spacing()

    def get_padding(self):
        print('Question: Padding value for this conv:')
        print('\n')
        gate=0
        while gate!=1:
            self.padding=[]
            padding_input=list(input('Please enter the value of padding as a list(depth,heigth,width)\nFor example 1,1,1\nFor default size, please directly press enter without any input: '))
            if len(padding_input)==0:
                print('Default Value selected')
                self.padding=tuple([1,1,1])
                gate=1
            elif len(padding_input)==5:
                for i in range(len(padding_input)):
                    if padding_input[i]!=',' and (1<= int(padding_input[i]) <= 20): 
                        self.padding.append(int(padding_input[i]))
                self.padding=tuple(self.padding)
                gate=1
            else:
                print('Please enter valid input')
        spacing()
    
    def forward(self, x):
        out=self.conv1(x)
        out = F.relu(self.batch(out),inplace=True)
        

        if self.pooling_input:
            out=self.pool(out)
        out=self.drop(out)
        return out


class Network(nn.Module):
    def __init__(self,img_size,num_class,in_channel=1):
        super(Network,self).__init__()
        print('NOTE: CNN3D convolution layers requires high memory, recommended not more 3 conv3d layers.')
        spacing()
        
        self.num_class=num_class
        self.img_size=img_size
        self.get_num_conv_layers()
        self.get_default_input()

        self.conv_layers=[]
        
        self.in_channel=in_channel
        for i in range(self.num_conv_layers):
            print('Designing the {} convolution block'.format(i+1))
            conv_layer=BasicBlock(self.in_channel,self.img_size,self.default_input)
            self.conv_layers.append(conv_layer)
            self.img_size=conv_layer.input_size
            print('The image shape after convolution is: ',self.img_size)
            print('\n')
            self.in_channel=conv_layer.out_channel

        
        final_layers = list(filter(None, self.conv_layers)) 
        self.net = nn.Sequential(*final_layers)
        
        self.fc1 = nn.Linear(self.in_channel*self.img_size[0]*self.img_size[1]*self.img_size[2], 1024)   # fully connected layer, output k classes
        self.fc2 = nn.Linear(1024,256)
        self.fc3 = nn.Linear(256, self.num_class)   
        


    def get_default_input(self):
        print('Question: Default value:')
        print('\n')

        self.default_input=input('Do you want default values for convolution layers (y/n): ')
        gate=0

        while gate!=1:
            if (self.default_input).lower()=='y':
                self.default_input=True
                gate=1

            elif (self.default_input).lower()=='n':
                self.default_input=False
                gate=1
            
            else:
                print('Please enter valid input')
        spacing() 

    
    def get_num_conv_layers(self):

        print('Question: Number of Convolution Layers:')
        print('\n')
        gate=0
        while gate!=1:
            conv_input=int(input('Please enter the number of conv_layers: '))
            if conv_input>0:
                self.num_conv_layers=conv_input
                gate=1
            else:
                print('Please enter a valid input')
        spacing()
        
        
    def forward(self,x):
                            
        x=self.net(x)
        x = x.view(x.size(0), -1)
        
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        
        return x


class CNN3D():

    def __init__(self,train_data,dev_data):
        
        #train_data
        self.train_data=train_data
        #dev_data
        self.dev_data=dev_data
        spacing()
        self.get_num_classes()
        
        print('Question [2/7]: Design Architecture: ')
        print('\n')


        self.net = (Network(self.train_data[0][0].shape,3)).double()
        print(self.net)

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

        # Method for setting device type if GPU is available

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

        self.net.to(self.device)         # printing model architecture


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
        
    




        

        