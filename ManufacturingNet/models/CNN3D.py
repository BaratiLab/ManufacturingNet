import datetime
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torchvision
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MyDataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        X = torch.from_numpy(self.X[index,:,:,:,:]).double()  # (in_channel,depth,height,width)
        Y = torch.from_numpy(np.array(self.Y[index])).double()
        return X, Y


def conv3D_output_size(img_size, kernel_size, stride, padding):
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] -
                          (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape


def spacing():
    print('\n')
    print('='*75)
    print('='*75)


class BasicBlock(nn.Module):
    def __init__(self, in_channel, input_size, default=False):
        super(BasicBlock, self).__init__()

        self.input_size = input_size
        self.in_channel = in_channel
        self.get_channel_input()

        if default:
            self.kernel = (3, 3, 3)
            self.stride = (1, 1, 1)
            self.padding = (1, 1, 1)
            self.pooling_input = False
        else:
            self.get_kernel_size()
            self.get_stride()
            self.get_padding()

        self.conv1 = nn.Conv3d(in_channels=in_channel, out_channels=self.out_channel,
                               kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        self.input_size = conv3D_output_size(
            self.input_size, self.kernel, self.stride, self.padding)
        self.batch = nn.BatchNorm3d(self.out_channel, momentum=0.01)

        if default == False:
            self.get_pooling_layer()
            if self.pooling_input:
                self.input_size = conv3D_output_size(
                    self.input_size, self.pool_kernel, self.pool_stride, self.pool_padding)

        self.get_dropout()

    def get_pooling_layer(self):
        print('Question: Pooling layer for this conv:')
        print('\n')
        gate = 0

        while gate != 1:
            self.pooling_qtn = input(
                'Do you want a pooling layer after this convolution layer (y/n): ').replace(' ','')
            if (self.pooling_qtn).lower() == 'y':
                self.pooling_input = True

                print('Question: Kernel Size for this pooling layer:')
                print('\n')
                gate = 0
                while gate != 1:
                    self.pool_kernel = []
                    kernel_input = ((input(
                        'Please enter the kernel size (depth,heigth,width)\nFor example 3,3,3\nFor default size, please directly press enter without any input: '))).replace(' ','')
                    if len(kernel_input) == 0:
                        print('Default Value selected')
                        self.pool_kernel = tuple([3, 3, 3])
                        gate = 1
                    k_split = kernel_input.split(",")
                    if len(k_split) == 3:

                        for i in k_split:
                            if i.isnumeric() and int(i) > 0:
                                self.pool_kernel.append(int(i))
                                gate = 1
                            else:
                                gate = 0
                                print('Please enter valid input')
                                break
                        self.pool_kernel = tuple(self.pool_kernel)
                    else:
                        print('Please enter valid input')

                print('Question: Stride value for this pooling layer:')
                print('\n')
                gate = 0
                while gate != 1:
                    self.pool_stride = []
                    stride_input = ((input(
                        'Please enter the stride (depth,heigth,width)\nFor example 1,1,1\nFor default size, please directly press enter without any input: '))).replace(' ','')
                    if len(stride_input) == 0:
                        print('Default Value selected')
                        self.pool_stride = tuple([2, 2, 2])
                        gate = 1
                    s_split = stride_input.split(",")
                    if len(s_split) == 3:

                        for i in s_split:
                            if i.isnumeric() and int(i) > 0:
                                self.pool_stride.append(int(i))
                                gate = 1
                            else:
                                gate = 0
                                print('Please enter valid input')
                                break
                        self.pool_stride = tuple(self.pool_stride)
                    else:
                        print('Please enter valid input')

                print('Question: Padding value for this pooling layer:')
                print('\n')
                gate = 0
                while gate != 1:
                    self.pool_padding = []
                    padding_input = ((input(
                        'Please enter the value of padding (depth,heigth,width)\nFor example 1,1,1\nFor default size, please directly press enter without any input: '))).replace(' ','')
                    if len(padding_input) == 0:
                        print('Default Value selected')
                        self.pool_padding = tuple([0, 0, 0])
                        gate = 1
                    p_split = padding_input.split(",")
                    if len(p_split) == 3:
                        for i in p_split:
                            if i.isnumeric() and int(i) >= 0:
                                self.pool_padding.append(int(i))
                                gate = 1
                            else:
                                gate = 0
                                print('Please enter valid input')
                                break
                        self.pool_padding = tuple(self.pool_padding)
                    else:
                        print('Please enter valid input')
                self.pool = nn.MaxPool3d(kernel_size=self.pool_kernel, stride=self.pool_stride,
                                         padding=self.pool_padding)  # kept constant for now
                gate = 1

            elif (self.pooling_qtn).lower() == 'n':
                self.pooling_input = False
                gate = 1
            else:
                print('Please enter valid input')
        spacing()

    def get_dropout(self):  # Get input for dropout from the user
        gate1 = 0
        value = input("Do you want default values for dropout(press y or n): ").replace(' ','')
        while gate1 != 1:
            if value == "Y" or value == "y" or value == 'n' or value == 'N':
                gate1 = 1
            else:
                print("Please enter valid input it should only be (y or n)")
                value = input(
                    "Do you want default values for dropout(press y or n)").replace(' ','')
                gate1 = 0

        if value == 'N' or value == 'n':
            gate = 0
            drop_out = (input(("Please input the dropout probability: "))).replace(' ','')
            while gate != 1:
                if drop_out.replace('.', '').isdigit():
                    if (float(drop_out) >= 0.0 and float(drop_out) < 1.0):
                        self.drop = nn.Dropout3d(p=float(drop_out))
                        gate = 1
                    else:
                        print(
                            "Please enter the valid numeric values. The value should lie between 0 and 1")
                        drop_out = (
                            input(("Please input the dropout probability"))).replace(' ','')
                        gate = 0
                else:
                    drop_out = (
                        input(("Please input the dropout probability: "))).replace(' ','')
        else:
            self.drop = nn.Dropout3d(p=0)

    def get_channel_input(self):
        print('Question: Out-channels for this conv:')
        print('\n')
        gate = 0
        while gate != 1:
            channel_input = (
                input('Please enter the number of out channels: ')).replace(' ','')
            if channel_input.isnumeric() and int(channel_input) > 0:
                self.out_channel = int(channel_input)
                gate = 1
            else:
                print('Please enter a valid input')
        spacing()

    def get_kernel_size(self):
        print('Question: Kernel Size for this conv')
        print('\n')
        gate = 0
        while gate != 1:
            self.kernel = []
            kernel_input = ((input(
                'Please enter the kernel size (depth,heigth,width)\nFor example 3,3,3\nFor default size, please directly press enter without any input: '))).replace(' ','')
            if len(kernel_input) == 0:
                print('Default Value selected')
                self.kernel = tuple([3, 3, 3])
                gate = 1
            k_split = kernel_input.split(",")
            if len(k_split) == 3:
                for i in k_split:
                    if i.isnumeric() and int(i) > 0:
                        self.kernel.append(int(i))
                        gate = 1
                    else:
                        gate = 0
                        print('Please enter valid input')
                        break
                self.kernel = tuple(self.kernel)
            else:
                print('Please enter valid input')
        spacing()

    def get_stride(self):
        print('Question: Stride value for this conv:')
        print('\n')
        gate = 0
        while gate != 1:
            self.stride = []
            stride_input = ((input(
                'Please enter the stride (depth,heigth,width)\nFor example 1,1,1\nFor default size, please directly press enter without any input: '))).replace(' ','')
            if len(stride_input) == 0:
                print('Default Value selected')
                self.stride = tuple([1, 1, 1])
                gate = 1
            s_split = stride_input.split(",")
            if len(s_split) == 3:

                for i in s_split:
                    if i.isnumeric() and int(i) > 0:
                        self.stride.append(int(i))
                        gate = 1
                    else:
                        gate = 0
                        print('Please enter valid input')
                        break
                self.stride = tuple(self.stride)
            else:
                print('Please enter valid input')
        spacing()

    def get_padding(self):
        print('Question: Padding value for this conv:')
        print('\n')
        gate = 0
        while gate != 1:
            self.padding = []
            padding_input = ((input(
                'Please enter the value of padding (depth,heigth,width)\nFor example 1,1,1\nFor default size, please directly press enter without any input: '))).replace(' ','')
            if len(padding_input) == 0:
                print('Default Value selected')
                self.padding = tuple([0, 0, 0])
                gate = 1
            p_split = padding_input.split(",")
            if len(p_split) == 3:

                for i in p_split:
                    if i.isnumeric() and int(i) >= 0:
                        self.padding.append(int(i))
                        gate = 1
                    else:
                        gate = 0
                        print('Please enter valid input')
                        break
                self.padding = tuple(self.padding)
            else:
                print('Please enter valid input')
        spacing()

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.batch(out), inplace=True)
        if self.pooling_input:
            out = self.pool(out)
        out = self.drop(out)
        return out


class Network(nn.Module):
    def __init__(self, img_size, num_class, in_channel=1):
        super(Network, self).__init__()
        print('NOTE: CNN3D convolution layers requires high memory, recommended not more 3 conv3d layers.')
        spacing()

        self.num_class = num_class
        self.img_size = img_size
        self.get_num_conv_layers()
        self.get_default_input()

        self.conv_layers = []

        self.in_channel = in_channel
        for i in range(self.num_conv_layers):
            print('Designing the {} convolution block'.format(i+1))
            conv_layer = BasicBlock(
                self.in_channel, self.img_size, self.default_input)
            self.conv_layers.append(conv_layer)
            self.img_size = conv_layer.input_size
            self.in_channel = conv_layer.out_channel
            print('The image shape after convolution is: ', (self.in_channel,self.img_size[0],self.img_size[1],self.img_size[2]))
            print('\n')
            

        final_layers = list(filter(None, self.conv_layers))
        self.net = nn.Sequential(*final_layers)


        # fully connected layer, output k classes
        self.fc1 = nn.Linear(
            self.in_channel*self.img_size[0]*self.img_size[1]*self.img_size[2], 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, self.num_class)

    def get_default_input(self):
        print('Question: Default value:')
        print('\n')

        gate = 0

        while gate != 1:
            self.default_input = input(
                'Do you want default values for convolution layers (y/n): ').replace(' ','')
            if (self.default_input).lower() == 'y':
                self.default_input = True
                gate = 1

            elif (self.default_input).lower() == 'n':
                self.default_input = False
                gate = 1

            else:
                print('Please enter valid input')
        spacing()

    def get_num_conv_layers(self):

        print('Question: Number of Convolution Layers:')
        print('\n')
        gate = 0
        while gate != 1:
            conv_input = (input('Please enter the number of conv_layers: ')).replace(' ','')
            if conv_input.isnumeric() and int(conv_input) > 0:
                self.num_conv_layers = int(conv_input)
                gate = 1
            else:
                print('Please enter a valid input')
        spacing()

    def forward(self, x):

        x = self.net(x)
        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class CNN3D():
    """
    Documentation link: https://manufacturingnet.readthedocs.io/en/latest/
    """

    def __init__(self, X, Y, shuffle=True):

        # train_data
        self.X = X
        # dev_data
        self.Y = Y

        self.shuffle = shuffle

        self.get_default_paramters()

        self.get_num_classes()

        print('Question [2/9]: Design Architecture: ')
        print('\n')

        self.net = (Network(self.X[0].shape[1:], self.num_classes,self.X[0].shape[0])).double()
        # getting a batch size for training and validation
        self._get_batchsize_input()
        self._get_valsize_input()
        self._get_loss_function()               # getting a loss function

        self._get_optimizer()               # getting an optimizer input

        self._get_scheduler()               # getting a scheduler input

        self._set_device()              # setting the device to gpu or cpu

        self._get_epoch()           # getting an input for number oftraining epochs

        print(self.net)

        self.dataloader()

        self.main()             # run function

    def get_default_paramters(self):

        # Method for getting a binary input for default paramters

        gate = 0
        while gate != 1:
            self.default = input(
                'Do you want default values for all the training parameters (y/n)? ').replace(' ','')
            if self.default == 'y' or self.default == 'Y' or self.default == 'n' or self.default == 'N':
                if self.default.lower() == 'y':
                    self.default_gate = True
                else:
                    self.default_gate = False
                gate = 1
            else:
                print('Enter a valid input')
                print(' ')

    def get_num_classes(self):
        print('Question [1/9]: No of classes:')
        print('\n')

        gate = 0
        while gate != 1:
            self.num_classes = (input(
                'Please enter the number of classes for classification \n Enter 1 if you are dealing with a regression problem: ')).replace(' ','')
            if (self.num_classes).isnumeric() and int(self.num_classes) >= 1:
                self.num_classes = int(self.num_classes)
                gate = 1
            else:
                print('Please enter a valid input')
        spacing()

    def _get_batchsize_input(self):
        print('Question [3/9]: Batchsize:')
        print('\n')

        # Method for getting batch size input
        gate = 0
        while gate != 1:
            self.batch_size = (input('Please enter the batch size: ')).replace(' ','')
            if (self.batch_size).isnumeric() and int(self.batch_size) > 0:
                self.batch_size = int(self.batch_size)
                gate = 1
            else:
                print('Please enter a valid input')
        spacing()

    def _get_valsize_input(self):
        print('Question [4/9]: Validation_size:')
        # Method for getting validation set size input
        gate = 0
        while gate != 1:
            if self.default_gate == True:
                print('Default value selected : 0.2')
                self.valset_size = '0.2'
            else:
                self.valset_size = (input(
                    'Please enter the train set size float input (size > 0 and size < 1) \n For default size, please directly press enter without any input: ')).replace(' ','')
            if self.valset_size == '':              # handling default case for valsize
                print('Default value selected : 0.2')
                self.valset_size = '0.2'
            if self.valset_size.replace('.', '').isdigit():
                if float(self.valset_size) > 0 and float(self.valset_size) < 1:
                    self.valset_size = float(self.valset_size)
                    gate = 1
            else:
                print('Please enter a valid input')

    def _set_device(self):

        # Method for setting device type if GPU is available

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.cuda = True
        else:
            self.device = torch.device('cpu')
            self.cuda = False

    def _get_loss_function(self):
        print('Question [5/9]: Loss function:')
        print('\n')

        # Method for getting a loss function for training
        self.criterion_list = {1: nn.CrossEntropyLoss(), 2: torch.nn.L1Loss(
        ), 3: torch.nn.SmoothL1Loss(), 4: torch.nn.MSELoss()}

        gate = 0
        while gate != 1:
            self.criterion_input = (input(
                'Please enter the appropriate loss function for the problem: \n Criterion_list - [1: CrossEntropyLoss, 2: L1Loss, 3: SmoothL1Loss, 4: MSELoss]: ')).replace(' ','')

            if self.criterion_input.isnumeric() and int(self.criterion_input) < 5 and int(self.criterion_input) > 0:
                gate = 1
            else:
                print('Please enter a valid input')
        spacing()

        self.criterion = self.criterion_list[int(self.criterion_input)]

    def dataloader(self):

        # train and dev loader

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=self.valset_size)

        train_dataset = MyDataset(self.X_train, self.Y_train)

        train_loader_args = dict(shuffle=True, batch_size=self.batch_size)

        self.train_loader = data.DataLoader(train_dataset, **train_loader_args)

        dev_dataset = MyDataset(self.X_test, self.Y_test)

        dev_loader_args = dict(shuffle=False, batch_size=self.batch_size)

        self.dev_loader = data.DataLoader(dev_dataset, **dev_loader_args)

    def _get_optimizer(self):
        print('Question [6/9]: Optimizer:')
        print('\n')

        # Method for getting a optimizer input

        gate = 0
        while gate != 1:
            if self.default_gate == True:
                print('Default optimizer selected : Adam')
                self.optimizer_input = '1'
            else:
                self.optimizer_input = (input(
                    'Please enter the optimizer index for the problem \n Optimizer_list - [1: Adam, 2: SGD] \n For default optimizer, please directly press enter without any input: ')).replace(' ','')
            if self.optimizer_input == '':              # handling default case for optimizer
                print('Default optimizer selected : Adam')
                self.optimizer_input = '1'

            if self.optimizer_input.isnumeric() and int(self.optimizer_input) > 0 and int(self.optimizer_input) < 3:
                gate = 1
            else:
                print('Please enter a valid input')
                print(' ')
        print(' ')
        gate = 0
        while gate != 1:
            if self.default_gate == True:
                print('Default value for learning rate selected : 0.001')
                self.user_lr = '0.001'
            else:
                self.user_lr = input(
                    'Please enter a required value float input for learning rate (learning rate > 0) \n For default learning rate, please directly press enter without any input: ').replace(' ','')
            if self.user_lr == '':               # handling default case for learning rate
                print('Default value for learning rate selected : 0.001')
                self.user_lr = '0.001'
            if self.user_lr.replace('.', '').isdigit():
                if float(self.user_lr) > 0:
                    self.lr = float(self.user_lr)
                    gate = 1
            else:
                print('Please enter a valid input')
                print(' ')
        spacing()

        self.lr = float(self.user_lr)
        self.optimizer_list = {1: optim.Adam(self.net.parameters(
        ), lr=self.lr), 2: optim.SGD(self.net.parameters(), lr=self.lr)}
        self.optimizer = self.optimizer_list[int(self.optimizer_input)]


# scheduler

    def _get_scheduler(self):
        print('Question [8/9]: Scheduler:')
        print('\n')

        # Method for getting scheduler

        gate = 0
        while gate != 1:
            if self.default_gate == True:
                print('By default no scheduler selected')
                self.scheduler_input = '1'
            else:
                self.scheduler_input = input(
                    'Please enter the scheduler index for the problem: Scheduler_list - [1: None, 2:StepLR, 3:MultiStepLR] \n For default option of no scheduler, please directly press enter without any input: ').replace(' ','')
            if self.scheduler_input == '':
                print('By default no scheduler selected')
                self.scheduler_input = '1'
            if self.scheduler_input.isnumeric() and int(self.scheduler_input) > 0 and int(self.scheduler_input) < 4:
                gate = 1
            else:
                print('Please enter a valid input')
                print(' ')

        if self.scheduler_input == '1':
            print(' ')
            self.scheduler = None

        elif self.scheduler_input == '2':
            print(' ')
            gate = 0
            while gate != 1:
                self.step = (
                    input('Please enter a step value int input (step > 0): ')).replace(' ','')
                if self.step.isnumeric() and int(self.step) > 0:
                    self.step = int(self.step)
                    gate = 1
                else:
                    print('Please enter a valid input')
                    print(' ')
            print(' ')
            gate = 0
            while gate != 1:
                self.gamma = (input(
                    'Please enter a Multiplying factor value float input (Multiplying factor > 0): ')).replace(' ','')
                if self.gamma.replace('.', '').isdigit():
                    if float(self.gamma) > 0:
                        self.gamma = float(self.gamma)
                        gate = 1
                else:
                    print('Please enter a valid input')
                    print(' ')

            self.scheduler = scheduler.StepLR(
                self.optimizer, step_size=self.step, gamma=self.gamma)

        elif self.scheduler_input == '3':
            print(' ')
            gate = 0
            while gate != 1:
                self.milestones_input = (
                    input('Please enter values of milestone epochs int input (Example: 2, 6, 10): ')).replace(' ','')
                self.milestones_input = self.milestones_input.split(',')
                for i in range(len(self.milestones_input)):
                    if self.milestones_input[i].isnumeric() and int(self.milestones_input[i]) > 0:
                        gate = 1
                    else:
                        gate = 0
                        break
                if gate == 0:
                    print('Please enter a valid input')
                    print(' ')

            self.milestones = [int(x)
                               for x in self.milestones_input if int(x) > 0]
            print(' ')

            gate = 0
            while gate != 1:
                self.gamma = (input(
                    'Please enter a Multiplying factor value float input (Multiplying factor > 0): ')).replace(' ','')
                if self.gamma.replace('.', '').isdigit():
                    if float(self.gamma) > 0:
                        self.gamma = float(self.gamma)
                        gate = 1
                else:
                    print('Please enter a valid input')
                    print(' ')
            self.scheduler = scheduler.MultiStepLR(
                self.optimizer, milestones=self.milestones, gamma=self.gamma)
        spacing()

    def _get_epoch(self):
        print('Question [9/9]: Number of Epochs:')
        print('\n')

        # Method for getting number of epochs for training the model

        gate = 0
        while gate != 1:
            self.numEpochs = (input(
                'Please enter the number of epochs int input to train the model (number of epochs > 0): ')).replace(' ','')
            if self.numEpochs.isnumeric() and int(self.numEpochs) > 0:
                self.numEpochs = int(self.numEpochs)
                gate = 1
            else:
                print('Please enter a valid input')
                print(' ')

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
            print('Epoch_Number: ', epoch)
            running_loss = 0.0

            for batch_idx, (data, target) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                data = data.to(self.device)
                target = target.to(self.device)

                outputs = self.net(data)

                # calculating the batch accuracy only if the loss function is Cross entropy
                if self.criterion_input == '1':

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

            # printing the epoch accuracy only if the loss function is Cross entropy
            if self.criterion_input == '1':

                acc = (correct_predictions/total_predictions)*100.0
                self.training_acc.append(acc)
                print('Training Accuracy: ', acc, '%')

            dev_loss, dev_acc = self.validate_model()

            if self.scheduler_input != '1':

                self.scheduler.step()
                print('Current scheduler status: ', self.optimizer)

            end_time = time.time()
            print('Epoch Time: ', end_time - start_time, 's')
            print('#'*50)

            self.dev_loss.append(dev_loss)

            # saving the epoch validation accuracy only if the loss function is Cross entropy
            if self.criterion_input == '1':

                self.dev_accuracy.append(dev_acc)

    def validate_model(self):

        with torch.no_grad():
            self.net.eval()
        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0
        acc = 0
        self.actual = []
        self.predict = []

        for batch_idx, (data, target) in enumerate(self.dev_loader):

            data = data.to(self.device)
            target = target.to(self.device)
            outputs = self.net(data)

            if self.criterion_input == '1':

                loss = self.criterion(outputs, target.long())
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += target.size(0)
                correct_predictions += (predicted == target).sum().item()
                self.predict.append(predicted.detach().cpu().numpy())

            else:
                loss = self.criterion(outputs, target)
                self.predict.append(outputs.detach().cpu().numpy())
            running_loss += loss.item()
            self.actual.append(target.detach().cpu().numpy())

        running_loss /= len(self.dev_loader)
        print('Validation Loss: ', running_loss)

        # calculating and printing the epoch accuracy only if the loss function is Cross entropy
        if self.criterion_input == '1':

            acc = (correct_predictions/total_predictions)*100.0
            print('Validation Accuracy: ', acc, '%')

        return running_loss, acc

    def get_loss_graph(self):

        # Method for showing and saving the loss graph in the root directory

        plt.figure(figsize=(8, 8))
        plt.plot(self.training_loss, label='Training Loss')
        plt.plot(self.dev_loss, label='Validation Loss')
        plt.legend()
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.savefig('loss.png')

    def get_accuracy_graph(self):

        # Method for showing and saving the accuracy graph in the root directory

        plt.figure(figsize=(8, 8))
        plt.plot(self.training_acc, label='Training Accuracy')
        plt.plot(self.dev_accuracy, label='Validation Accuracy')
        plt.legend()
        plt.title('Model accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('acc')
        plt.savefig('accuracy.png')

    def get_prediction(self, x_input):

        # Method to use at the time of inference

        if len(x_input.shape) == 3:             # handling the case of single

            x_input = (x_input).reshape(
                1, x_input.shape[0], x_input.shape[1], x_input.shape[2])

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
            self.get_confusion_matrix()         # printing confusion matrix
        else:

            self.get_r2_score()             # saving r2 score graph

        self._save_model()              # saving model paramters

        print(' Call get_prediction() to make predictions on new data')
        print(' ')
        print('=== End of training ===')

    def _save_model(self):

        # Method for saving the model parameters if user wants to

        gate = 0
        save_model = input('Do you want to save the model weights? (y/n): ').replace(' ','')
        while gate != 1:
            if save_model.lower() == 'y' or save_model.lower() == 'yes':
                path = 'model_parameters.pth'
                torch.save(self.net.state_dict(), path)
                gate = 1
            elif save_model.lower() == 'n' or save_model.lower() == 'no':
                gate = 1
            else:
                print('Please enter a valid input')

    def get_confusion_matrix(self):

        # Method for getting the confusion matrix for classification problem
        print('Confusion Matix: ')

        result = confusion_matrix(np.concatenate(
            np.array(self.predict)), np.concatenate(np.array(self.actual)))
        print(result)

    def get_r2_score(self):

        # Method for getting the r2 score for regression problem
        print('r2 score: ')
        result = r2_score(np.concatenate(np.array(self.predict)),
                          np.concatenate(np.array(self.actual)))
        print(result)

        plt.figure(figsize=(8, 8))
        plt.scatter(np.concatenate(np.array(self.actual)), np.concatenate(
            np.array(self.predict)), label='r2 score', s=1)
        plt.legend()
        plt.title('Model r2 score: ' + str(result))
        plt.xlabel('labels')
        plt.ylabel('predictions')
        plt.savefig('r2_score.png')
