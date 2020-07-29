# Importing all the necessary files and functions

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt


# The following class is used to create necessary inputs for dataset class and dataloader class used during training process
class ModelDataset():

    def __init__(self, X, Y, batchsize, valset_size, shuffle):

        self.x = X                      # Inputs
        self.y = Y                      # Labels
        self.batchsize = batchsize 
        self.valset_size = valset_size
        self.shuffle = shuffle
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x, self.y, test_size = self.valset_size, shuffle = self.shuffle)
    
    def get_trainset(self):

        # Method for getting training set inputs and labels

        return self.x_train, self.y_train

    def get_valset(self):

        # Method for getting validation set inputs and labels

        return self.x_val, self.y_val

    def get_batchsize(self):

        # Method for getting batch size for training and validatioin

        return self.batchsize


# The following class is used for creating a dataset class using torch functionality. Its a standard pytorch class
class Dataset(data.Dataset):

    def __init__(self, X, Y):

        self.X = X
        self.Y = Y

    def __len__(self):

        return len(self.Y)

    def __getitem__(self,index):

        x_item = torch.from_numpy(self.X[index]).double()
        y_item = torch.from_numpy(np.array(self.Y[index]))

        return x_item, y_item


# The following function builds a deep neural network by asking inputs from the user
class DNN(nn.Module):
    
    def __init__(self, negative_slope=0.01):

        super(DNN, self).__init__()
        
        # A list of activation functions used in _get_activation_input()

        self.activation_functions = {0: None, 1: nn.ReLU(), 2: nn.LeakyReLU(negative_slope= negative_slope), 3: nn.GELU(), 4: nn.SELU(), 5: nn.Sigmoid(), 6: nn.Tanh() }

        print(' ')
        print('1/10')
        self._get_neuron_input()

        print(' ')
        print('2/10')
        self._get_activation_input()

        print(' ')
        print('3/10')
        self._get_batchnorm_input()

        print(' ')
        print('4/10')
        self._get_dropout_input()

        self._build_network_architecture()
        

    def _get_neuron_input(self):

        # Method for getting number of neurons for each layer

        gate = 0
        while gate != 1:
            self.list_of_neurons = input('Please enter the number of neurons for each layer including input and output layers sequentially: ')
            self.list_of_neurons = self.list_of_neurons.split(',')
            self.size_list = [int(x) for x in self.list_of_neurons if int(x)>0]
            if (len(self.size_list)) > 2:
                gate = 1
            else:
                print(' ')
                print('Please enter at least 3 values. ' , 'Entered: ', len(self.size_list))

    def _get_activation_input(self):

        # Method for getting activation fucntions for each hidden layer

        gate = 0
        while gate != 1:
            self.activations = input('Please enter the activations for each hidden layer sequentially: \n Activation functions - \n [0: None, 1: ReLU, 2: LeakyReLU, \n 3: GELU(), 4: SELU(), 5: Sigmoid(), 6: Tanh()] \n For default option of ReLU, please directly press enter without any input: ')
            if self.activations == '':              # handling default case for ReLU activation function
                print('Default value selected')
                for i in range((len(self.size_list)-2)):
                    self.activations += '1'
                    self.activations += ','
                self.activations = self.activations[:-1]

            self.activations = self.activations.split(',')
            self.activation_list = [int(x) for x in self.activations if int(x)<7]
            self.activation_list = [int(x) for x in self.activation_list if int(x) >= 0]
            if len(self.activation_list) == (len(self.size_list)-2):
                gate = 1
            else:
                print(' ')
                print('Please enter the activations for the hidden layers. Required: ', (len(self.size_list)-2), ' Entered: ', len(self.activation_list))
    
    def _get_batchnorm_input(self):

        # Method for getting batchnorm input for each hidden layer
        
        gate = 0
        while gate != 1:
            self.batchnorms = input('Please enter 1 if batchnorm is required else enter 0 for each layer. \n For default option of no batchnorm to any layer, please directly press enter without any input: ')
            if self.batchnorms == '':               # handling default case for batchnorm
                print('Default value selected')
                for i in range((len(self.size_list)-2)):
                    self.batchnorms += '0'
                    self.batchnorms += ','
                self.batchnorms = self.batchnorms[:-1]  
            self.batchnorms = self.batchnorms.split(',')
            self.batchnorm_list = [int(x) for x in self.batchnorms if int(x)<2]
            self.batchnorm_list = [int(x) for x in self.batchnorm_list if int(x)>-1]
            if len(self.batchnorm_list) == (len(self.size_list)-2):
                gate = 1
            else:
                print(' ')
                print('Please enter the batchnorm for the hidden layers. Required entries: ', (len(self.size_list)-2), ' Entered: ', len(self.batchnorm_list))

    def _get_dropout_input(self):

        # Method for getting dropout input for each hidden layer

        gate = 0
        while gate != 1:
            self.dropout_values = input('Please enter the dropout values between 0 and 1 for each hidden layer. \n For default option of no dropout in any layer, please directly press enter without any input: ')
            if self.dropout_values == '':               # handling default case for dropout
                print('Default value selected')
                for i in range((len(self.size_list)-2)):
                    self.dropout_values += '0'
                    self.dropout_values += ','
                self.dropout_values = self.dropout_values[:-1] 
            self.dropout_values = self.dropout_values.split(',')
            self.dropout_list = [float(x) for x in self.dropout_values if float(x) < 1]
            self.dropout_list = [float(x) for x in self.dropout_list if float(x) >= 0]
            if len(self.dropout_list) == (len(self.size_list)-2):
                gate = 1
            else:
                print(' ')
                print('Please enter the dropout values for each hidden layers. Required entries: ', (len(self.size_list)-2), ' Entered: ', len(self.dropout_list))

    def _build_network_architecture(self):

        # Method for building a network using all the information provided by a user in above functions

        layers = []

        for i in range(len(self.size_list) - 2):
            layers.append(nn.Linear(self.size_list[i],self.size_list[i+1]))
            if self.batchnorm_list[i] == 1:
                layers.append(nn.BatchNorm1d(self.size_list[i+1]))
            layers.append(self.activation_functions[self.activation_list[i]]) 
            layers.append(nn.Dropout(self.dropout_list[i]))
        layers.append(nn.Linear(self.size_list[-2], self.size_list[-1]))
        final_layers = list(filter(None, layers)) 
        self.net = nn.Sequential(*final_layers)

    def forward(self, x):

        # Standard Pytorch function used during training

        return self.net(x)


    def predict(self, x):

        # Method for getting output during inference time once the model is trained

        return self.net(x)

# The following class will be called by a user. The class calls other necessary classes to build a complete pipeline required for training
class DNNModel():
    """
    Documentation Link:

    """
    def __init__(self,X,Y, shuffle = True):

        # Lists used in the functions below
        self.criterion_list = {1:nn.CrossEntropyLoss(),2:torch.nn.L1Loss(),3:torch.nn.SmoothL1Loss(),4:torch.nn.MSELoss()}
        
        self.x_data = X
        self.y_data = Y
        self.shuffle = shuffle

        self.net = (DNN()).double()             # building a network architecture

        print(' ')
        print('5/10')
        self._get_batchsize_input()             # getting a batch size for training and validation

        print(' ')
        print('6/10')
        self._get_valsize_input()                # getting a train-validation split

        self.model_data = ModelDataset(self.x_data, self.y_data, batchsize = self.batchsize, valset_size = self.valset_size, shuffle = self.shuffle)          # splitting the data into training and validation sets

        print(' ')
        print('7/10')
        self._get_loss_function()               # getting a loss function

        print(' ')
        print('8/10')
        self._get_optimizer()               # getting an optimizer input

        print(' ')
        print('9/10')
        self._get_scheduler()               # getting a scheduler input

        self._set_device()              # setting the device to gpu or cpu

        print(' ')
        print('10/10')
        self._get_epoch()           # getting an input for number oftraining epochs

        self.main()             # run function
    
    def _get_batchsize_input(self):

        # Method for getting batch size input

        gate = 0
        while gate!= 1:
            self.batchsize = int(input('Please enter the batch size: '))
            if int(self.batchsize) >0 :
                gate =1
            else:
                print('Please enter a valid input')
    
    def _get_valsize_input(self):

        # Method for getting validation set size input
        
        gate = 0
        while gate!= 1:
            self.valset_size = (input('Please enter the validation set size (size > 0 and size < 1) \n For default size, please directly press enter without any input: '))
            if self.valset_size == '':              # handling default case for valsize
                print('Default value selected')
                self.valset_size = '0.2'
            if float(self.valset_size) >0 and float(self.valset_size) < 1:
                self.valset_size = float(self.valset_size)
                gate =1
            else:
                print('Please enter a valid input')
        
    def _get_loss_function(self):

        # Method for getting a loss function for training
        
        gate = 0
        while gate!= 1:
            self.criterion_input = (input('Please enter the appropriate loss function for the problem: \n Criterion_list - [1: CrossEntropyLoss, 2: L1Loss, 3: SmoothL1Loss, 4: MSELoss]: '))

            if self.criterion_input.isnumeric() and int(self.criterion_input) < 5 and int(self.criterion_input)> 0:
                gate =1
            else:
                print('Please enter a valid input')
            
        self.criterion = self.criterion_list[int(self.criterion_input)]

    def _get_optimizer(self):

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

        self.lr = float(self.user_lr)
        self.optimizer_list = {1:optim.Adam(self.net.parameters(),lr = self.lr), 2:optim.SGD(self.net.parameters(),lr = self.lr)}
        self.optimizer = self.optimizer_list[int(self.optimizer_input)]


    def _get_scheduler(self):

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
        
    def _set_device(self):

        # Method for setting device type if GPU is available

        if torch.cuda.is_available():
            self.device=torch.device('cuda')
        else:
            self.device=torch.device('cpu')

    def _get_epoch(self):

        # Method for getting number of epochs for training the model

        self.numEpochs = int(input('Please enter the number of epochs to train the model: '))


    def main(self):

        # Method integrating all the functions and training the model

        self.net.to(self.device)
        print('#'*50)

        print(self.net)         # printing model architecture
        print('#'*50)

        self.get_model_summary()        # printing summaray of the model
        print('#'*50)
        
        xt, yt = self.model_data.get_trainset()         # getting inputs and labels for training set

        xv, yv = self.model_data.get_valset()           # getting inputs and labels for validation set

        self.train_dataset = Dataset(xt, yt)            # creating the training dataset

        self.val_dataset = Dataset(xv, yv)              # creating the validation dataset

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size = self.model_data.get_batchsize(), shuffle = True)           # creating the training dataset dataloadet

        self.dev_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size = self.model_data.get_batchsize())           # creating the validation dataset dataloader

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
        
        
    def get_model_summary(self):

        # Method for getting the summary of the model
        print('Model Summary:')
        print(' ')
        print('Criterion: ', self.criterion)
        print('Optimizer: ', self.optimizer)
        print('Scheduler: ', self.scheduler)
        print('Validation set size: ', self.valset_size)
        print('Batch size: ', self.batchsize)
        print('Initial learning rate: ', self.lr)
        print('Number of training epochs: ', self.numEpochs)
        print('Device: ', self.device)
    

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
