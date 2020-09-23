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
import torch.utils.data as data_utils
import torchvision
from torch.utils import data as data_utils
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (resnet18, resnet34, resnet50, resnet101,
                                resnext50_32x4d)


class MyDataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        # (in_channel,depth,height,width)
        X = torch.from_numpy(self.X[index]).double()
        Y = torch.from_numpy(np.array(self.Y[index])).long()
        return X, Y


def spacing():
    print('\n')
    print('='*70)
    print('='*70)


class ResNet():

    def __init__(self, X, Y, shuffle=True):

        # train_data
        self.X = X
        self.Y = Y
        self.shuffle = shuffle
        spacing()
        # (1 question)
        self.get_num_classes()
        self.get_pretrained_model()
        print(self.net)         # printing model architecture

        # getting a batch size for training and validation
        self._get_batchsize_input()

        self._get_valsize_input()

        self._get_loss_function()               # getting a loss function

        self._get_optimizer()               # getting an optimizer input

        self._get_scheduler()               # getting a scheduler input

        self._set_device()              # setting the device to gpu or cpu

        self._get_epoch()           # getting an input for number oftraining epochs

        self.dataloader()

        self.main()             # run function

    def get_num_classes(self):
        print('Question [1/9]: No of classes:')
        print('\n')
        gate = 0
        while gate != 1:
            self.num_classes_input = (input(
                'Please enter the number of classes \nFor classification (2 or more) \nFor Regression enter 1: ').replace(' ',''))
            if self.num_classes_input.isnumeric() and int(self.num_classes_input) > 0:
                self.num_classes = int(self.num_classes_input)
                gate = 1
            else:
                print('Please enter a valid input')
        spacing()

    def get_pretrained_model(self):
        print('Question [2/9]: Model Selection:')
        print('\n')
        self.pretrained_dict = {1: resnet18, 2: resnet34,
                                3: resnet50, 4: resnet101, 5: resnext50_32x4d}

        gate = 0
        while gate != 1:
            pretrained_input = input('Do you want pretrained model? (y/n): ').replace(' ','')
            if pretrained_input.lower() == 'y':
                self.pretrained = True
                gate = 1
            elif pretrained_input.lower() == 'n':
                self.pretrained = False
                gate = 1
            else:
                print('Please enter valid input')

        gate = 0
        while gate != 1:
            self.model_select = int(input('Please enter any number between 1 to 5 to select the model:\
                                        \n[1:ResNet18,2:ResNet34,3:ResNet50,4:ResNet101,5:ResNext50]').replace(' ',''))
            if (1 <= self.model_select <= 5):
                model = self.pretrained_dict[self.model_select](
                    pretrained=self.pretrained)
                gate = 1
            else:
                print('Please enter valid input')

        print(self.X[0].shape[0])
        model.conv1 = nn.Conv2d(self.X[0].shape[0], 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if self.model_select in [1, 2]:
            model.fc = nn.Linear(512, self.num_classes)
        else:
            model.fc = nn.Linear(2048, self.num_classes)

        self.net = model.double()

        spacing()

    def _get_batchsize_input(self):
        print('Question [3/9]: Batchsize:')
        print('\n')
        # Method for getting batch size input
        gate = 0
        while gate != 1:
            self.batch_size = int(input('Please enter the batch size: ').replace(' ',''))
            if int(self.batch_size) > 0:
                gate = 1
            else:
                print('Please enter a valid input')
        spacing()

    def _get_valsize_input(self):
        print('Question [4/9]: Validation_size:')
        # Method for getting validation set size input
        gate = 0
        while gate != 1:
            self.valset_size = (input(
                'Please enter the validation set size (size > 0 and size < 1) \n For default size, please directly press enter without any input: ').replace(' ',''))
            if self.valset_size == '':              # handling default case for valsize
                print('Default value selected')
                self.valset_size = '0.2'
            if self.valset_size.replace('.', '').isdigit():
                if float(self.valset_size) > 0 and float(self.valset_size) < 1:
                    self.valset_size = float(self.valset_size)
                    gate = 1
            else:
                print('Please enter a valid numeric input')
        spacing()

    def _set_device(self):
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
                'Please enter the appropriate loss function for the problem: \n Criterion_list - [1: CrossEntropyLoss, 2: L1Loss, 3: SmoothL1Loss, 4: MSELoss]: ').replace(' ',''))

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

        train_loader_args = dict(shuffle=self.shuffle,
                                 batch_size=self.batch_size)

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
            self.optimizer_input = (input(
                'Please enter the optimizer for the problem \n Optimizer_list - [1: Adam, 2: SGD] \n For default optimizer, please directly press enter without any input: ').replace(' ',''))
            if self.optimizer_input == '':              # handling default case for optimizer
                print('Default optimizer selected')
                self.optimizer_input = '1'

            if self.optimizer_input.isnumeric() and int(self.optimizer_input) > 0 and int(self.optimizer_input) < 3:
                gate = 1
            else:
                print('Please enter a valid input')
        spacing()

        print('Question [7/9]: Learning_Rate:')
        gate = 0
        while gate != 1:
            self.user_lr = input(
                'Please enter a required postive value for learning rate \n For default learning rate, please directly press enter without any input: ').replace(' ','')
            if self.user_lr == '':               # handling default case for learning rate
                print('Default value selected')
                self.user_lr = '0.001'
            if float(self.user_lr) > 0:
                gate = 1
            else:
                print('Please enter a valid input')
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
            self.scheduler_input = input(
                'Please enter the scheduler for the problem: Scheduler_list - [1: None, 2:StepLR, 3:MultiStepLR] \n For default option of no scheduler, please directly press enter without any input: ').replace(' ','')
            if self.scheduler_input == '':
                print('By default no scheduler selected')
                self.scheduler_input = '1'
            if self.scheduler_input.isnumeric() and int(self.scheduler_input) > 0 and int(self.scheduler_input) < 4:
                gate = 1
            else:
                print('Please enter a valid input')

        if self.scheduler_input == '1':
            self.scheduler = None

        elif self.scheduler_input == '2':
            self.step = int(input('Please enter a step value: ').replace(' ',''))
            print(' ')
            self.gamma = float(
                input('Please enter a gamma value (Multiplying factor): ').replace(' ',''))
            self.scheduler = scheduler.StepLR(
                self.optimizer, step_size=self.step, gamma=self.gamma)

        elif self.scheduler_input == '3':
            self.milestones_input = (
                input('Please enter values of milestone epochs: ').replace(' ',''))
            self.milestones_input = self.milestones_input.split(',')
            self.milestones = [int(x)
                               for x in self.milestones_input if int(x) > 0]
            print(' ')
            self.gamma = float(
                input('Please enter a gamma value (Multiplying factor): ').replace(' ',''))
            self.scheduler = scheduler.MultiStepLR(
                self.optimizer, milestones=self.milestones, gamma=self.gamma)

        spacing()

    def _get_epoch(self):
        print('Question [9/9]: Number of Epochs:')
        print('\n')

        # Method for getting number of epochs for training the model

        gate = 0
        while gate != 1:
            self.numEpochs = (
                input('Please enter the number of epochs to train the model: ').replace(' ',''))
            if self.numEpochs.isnumeric() and int(self.numEpochs) > 0:
                self.numEpochs = int(self.numEpochs)
                gate = 1
            else:
                print(
                    'Please enter a valid numeric input. The number must be integer and greater than zero')

    def main(self):

        # Method integrating all the functions and training the model

        self.net.to(self.device)

        self.get_model_summary()        # printing summaray of the model

        self.train_model()          # training the model

        self.get_loss_graph()           # saving the loss graph

        self.get_loss_graph()           # saving the loss graph

        if self.criterion_input == '1':

            self.get_accuracy_graph()           # saving the accuracy graph
            self.get_confusion_matrix()         # printing confusion matrix
        else:
            self.get_r2_score()             # saving r2 score graph

        self._save_model()               # saving model paramters

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

    def train_model(self):

        # Method for training the model

        self.net.train()
        self.training_loss = []
        self.training_acc = []
        self.dev_loss = []
        self.dev_accuracy = []
        total_predictions = 0.0
        correct_predictions = 0.0

        print('Training the model...')

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

    def get_prediction(self, x_input):
        """
        Pass in an input numpy array for making prediction.
        For passing multiple inputs, make sure to keep number of examples to be the first dimension of the input.
        For example, 10 data points need to be checked and each point has (3, 50, 50) resized or original input size, the shape of the array must be (10, 3, 50, 50).
        For more information, please see documentation.

        """

        # Method to use at the time of inference

        if len(x_input.shape) == 3:             # handling the case of single

            x_input = (x_input).reshape(
                1, x_input.shape[0], x_input.shape[1], x_input.shape[2])

        x_input = torch.from_numpy(x_input).to(self.device)

        net_output = self.net.predict(x_input)

        if self.criterion_input == '1':             # handling the case of classification problem

            _, net_output = torch.max(net_output.data, 1)

        return net_output
