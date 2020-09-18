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
from torchvision.models import alexnet


class MyDataset(data.Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        X = torch.from_numpy(self.X[index]).double()
        Y = torch.from_numpy(np.array(self.Y[index])).long()
        return X, Y


def spacing():
    print("\n")
    print("=" * 70)
    print("=" * 70)


class AlexNet:

    def __init__(self, X, Y, shuffle=True):
        # train_data
        self.X = X
        self.Y = Y
        self.shuffle = shuffle
        spacing()

        # (1 question)
        self.get_num_classes()
        self.get_pretrained_model()
        # Printing model architecture
        print(self.net)

        # getting a batch size for training and validation
        self._get_batchsize_input()

        self._get_valsize_input()

        # Getting a loss function
        self._get_loss_function()

        # Getting an optimizer input
        self._get_optimizer()

        # Getting a scheduler input
        self._get_scheduler()

        # Setting the device to GPU or CPU
        self._set_device()

        # Getting an input for number of training epochs
        self._get_epoch()

        self.dataloader()

        # Run function
        self.main()

    def get_num_classes(self):
        print("Question [1/9]: Number of classes:\n")
        gate = 0

        while gate != 1:
            print("Please enter the number of classes.")
            print("For classification, enter 2 or more.")
            self.num_classes_input = input("For regression, enter 1: ")

            if self.num_classes_input.isnumeric() and int(self.num_classes_input) > 0:
                self.num_classes = int(self.num_classes_input)
                gate = 1
            else:
                print("Please enter a valid input.")

        spacing()

    def get_pretrained_model(self):
        print("Question [2/9]: Model Selection:\n")
        gate = 0

        while gate != 1:
            pretrained_input = input(
                "Do you want the pretrained model (y/n)? ").lower()

            if pretrained_input == "y":
                self.pretrained = True
                gate = 1
            elif pretrained_input == "n":
                self.pretrained = False
                gate = 1
            else:
                print("Please enter valid input")

        model = alexnet(pretrained=self.pretrained)
        model.features[0] = nn.Conv2d(self.X[0].shape[0], 64, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.classifier[-1] = nn.Linear(4096, self.num_classes)

        self.net = model.double()
        spacing()

    def _get_batchsize_input(self):
        print("Question [3/9]: Batchsize:\n")
        gate = 0

        while gate != 1:
            self.batch_size = input("Please enter the batch size: ")
            if self.batch_size.isnumeric() and int(self.batch_size) > 0:
                gate = 1
            else:
                print("Please enter a valid input.")
        spacing()

    def _get_valsize_input(self):
        print("Question [4/9]: Validation_size:\n")
        gate = 0

        while gate != 1:
            print("Please enter the validation set size (0,1).")
            self.valset_size = input(
                "For default size, enter without any input: ")

            if self.valset_size == "":
                print("Default value selected")
                self.valset_size = "0.2"

            if self.valset_size.replace(".", "").isdigit():
                if float(self.valset_size) > 0 and float(self.valset_size) < 1:
                    self.valset_size = float(self.valset_size)
                    gate = 1
            else:
                print("Please enter a valid numeric input.")

        spacing()

    def _set_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.cuda = True
        else:
            self.device = torch.device("cpu")
            self.cuda = False

    def _get_loss_function(self):
        print("Question [5/9]: Loss function:\n")

        self.criterion_list = {1: nn.CrossEntropyLoss(), 2: torch.nn.L1Loss(),
                               3: torch.nn.SmoothL1Loss(), 4: torch.nn.MSELoss()}
        gate = 0

        while gate != 1:
            print("Please enter the appropriate loss function for the problem:")
            self.criterion_input = input(
                "[1: CrossEntropyLoss, 2: L1Loss, 3: SmoothL1Loss, 4: MSELoss]: ")

            if self.criterion_input.isnumeric() and int(self.criterion_input) < 5 and int(self.criterion_input) > 0:
                gate = 1
            else:
                print("Please enter a valid input.")

        self.criterion = self.criterion_list[int(self.criterion_input)]
        spacing()

    def dataloader(self):
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
        print("Question [6/9]: Optimizer:\n")
        gate = 0

        while gate != 1:
            print("Please enter the optimizer for the problem:")
            print("[1: Adam, 2: SGD]")
            self.optimizer_input = input(
                "For default optimizer, press enter without any input: ")

            if self.optimizer_input == "":
                print("Default optimizer selected")
                self.optimizer_input = "1"

            if self.optimizer_input.isnumeric() and int(self.optimizer_input) > 0 and int(self.optimizer_input) < 3:
                gate = 1
            else:
                print("Please enter a valid input.")

        spacing()
        print("Question [7/9]: Learning_Rate:\n")
        gate = 0

        while gate != 1:
            print("Please enter a positive value for the learning rate.")
            self.user_lr = input(
                "For default learning rate, press enter without any input: ")

            if self.user_lr == "":
                print("Default value selected")
                self.user_lr = "0.001"
            if float(self.user_lr) > 0:
                gate = 1
            else:
                print("Please enter a valid input.")

        spacing()

        self.lr = float(self.user_lr)
        self.optimizer_list = {1: optim.Adam(self.net.parameters(
        ), lr=self.lr), 2: optim.SGD(self.net.parameters(), lr=self.lr)}
        self.optimizer = self.optimizer_list[int(self.optimizer_input)]

    # Scheduler

    def _get_scheduler(self):
        print("Question [8/9]: Scheduler:\n")
        gate = 0

        while gate != 1:
            print("Please enter the scheduler for the problem:")
            print("[1: None, 2:StepLR, 3:MultiStepLR]")
            self.scheduler_input = input(
                "For default option of no scheduler, press enter without any input: ")

            if self.scheduler_input == "":
                print("By default no scheduler selected")
                self.scheduler_input = "1"
            if self.scheduler_input.isnumeric() and int(self.scheduler_input) > 0 and int(self.scheduler_input) < 4:
                gate = 1
            else:
                print("Please enter a valid input.")

        if self.scheduler_input == "1":
            self.scheduler = None

        elif self.scheduler_input == "2":
            self.step = int(input("Please enter a step value: "))
            print()
            self.gamma = float(
                input("Please enter a gamma value (multiplying factor): "))
            self.scheduler = scheduler.StepLR(
                self.optimizer, step_size=self.step, gamma=self.gamma)

        elif self.scheduler_input == "3":
            self.milestones_input = (
                input("Please enter values of milestone epochs: "))
            self.milestones_input = self.milestones_input.split(",")
            self.milestones = [int(x)
                               for x in self.milestones_input if int(x) > 0]
            print()
            self.gamma = float(
                input("Please enter a gamma value (Multiplying factor): "))
            self.scheduler = scheduler.MultiStepLR(
                self.optimizer, milestones=self.milestones, gamma=self.gamma)

        spacing()

    def _get_epoch(self):
        print("Question [9/9]: Number of Epochs:\n")
        gate = 0

        while gate != 1:
            self.numEpochs = (
                input("Please enter the number of epochs to train the model: "))
            if self.numEpochs.isnumeric() and int(self.numEpochs) > 0:
                self.numEpochs = int(self.numEpochs)
                gate = 1
            else:
                print("Please enter a valid numeric input.")
                print("The number must be an integer greater than zero.")

    def main(self):
        self.net.to(self.device)

        # Printing summary of the model
        self.get_model_summary()

        # Train the model
        self.train_model()

        # Save the loss graph
        self.get_loss_graph()

        if self.criterion_input == "1":
            # Save the accuracy graph
            self.get_accuracy_graph()
            # Print the confusion matrix
            self.get_confusion_matrix()
        else:
            # Save the r2 score graph
            self.get_r2_score()

        # Save model parameters
        self._save_model()

    def get_model_summary(self):
        print("Model Summary:\n")
        print("Criterion:", self.criterion)
        print("Optimizer:", self.optimizer)
        print("Scheduler:", self.scheduler)
        print("Batch size:", self.batch_size)
        print("Initial learning rate:", self.lr)
        print("Number of training epochs:", self.numEpochs)
        print("Device:", self.device)

        spacing()

    def _save_model(self):
        save_model = input("Do you want to save the model weights (y/n)? ")
        gate = 0
        while gate != 1:
            if save_model.lower() == "y" or save_model.lower() == "yes":
                path = "model_parameters.pth"
                torch.save(self.net.state_dict(), path)
                gate = 1
            elif save_model.lower() == "n" or save_model.lower() == "no":
                gate = 1
            else:
                print("Please enter a valid input.")

    def train_model(self):
        self.net.train()
        self.training_loss = []
        self.training_acc = []
        self.dev_loss = []
        self.dev_accuracy = []
        total_predictions = 0.0
        correct_predictions = 0.0

        print("Training the model...")

        for epoch in range(self.numEpochs):
            start_time = time.time()
            self.net.train()
            print("Epoch_Number: ", epoch)
            running_loss = 0.0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                data = data.to(self.device)
                target = target.to(self.device)

                outputs = self.net(data)

                # Calculating the batch accuracy only if the loss function is cross entropy
                if self.criterion_input == "1":
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
            print("Training Loss: ", running_loss)

            # Printing the epoch accuracy only if the loss function is Cross entropy
            if self.criterion_input == "1":

                acc = (correct_predictions/total_predictions)*100.0
                self.training_acc.append(acc)
                print("Training Accuracy: ", acc, "%")

            dev_loss, dev_acc = self.validate_model()

            if self.scheduler_input != "1":

                self.scheduler.step()
                print("Current scheduler status: ", self.optimizer)

            end_time = time.time()
            print("Epoch Time: ", end_time - start_time, "s")
            print("#"*50)

            self.dev_loss.append(dev_loss)

            # Saving the epoch validation accuracy only if the loss function is Cross entropy
            if self.criterion_input == "1":

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

            if self.criterion_input == "1":
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
        print("Validation Loss:", running_loss)

        # Calculating and printing the epoch accuracy only if the loss function is Cross entropy
        if self.criterion_input == "1":

            acc = (correct_predictions/total_predictions)*100.0
            print("Validation Accuracy:", acc, "%")

        return running_loss, acc

    def get_loss_graph(self):
        plt.figure(figsize=(8, 8))
        plt.plot(self.training_loss, label="Training Loss")
        plt.plot(self.dev_loss, label="Validation Loss")
        plt.legend()
        plt.title("Model Loss")
        plt.xlabel("Epochs")
        plt.ylabel("loss")
        plt.savefig("loss.png")

    def get_accuracy_graph(self):
        plt.figure(figsize=(8, 8))
        plt.plot(self.training_acc, label="Training Accuracy")
        plt.plot(self.dev_accuracy, label="Validation Accuracy")
        plt.legend()
        plt.title("Model accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("acc")
        plt.savefig("accuracy.png")

    def get_confusion_matrix(self):
        print("Confusion Matix:")
        result = confusion_matrix(np.concatenate(
            np.array(self.predict)), np.concatenate(np.array(self.actual)))
        print(result)

    def get_r2_score(self):
        print("r2 score:")
        result = r2_score(np.concatenate(np.array(self.predict)),
                          np.concatenate(np.array(self.actual)))
        print(result)

        plt.figure(figsize=(8, 8))
        plt.scatter(np.concatenate(np.array(self.actual)), np.concatenate(
            np.array(self.predict)), label="r2 score", s=1)
        plt.legend()
        plt.title("Model r2 score:" + str(result))
        plt.xlabel("labels")
        plt.ylabel("predictions")
        plt.savefig("r2_score.png")

    def get_prediction(self, x_input):
        """Pass in an input numpy array for making prediction.
        For passing multiple inputs, make sure to keep number of
        examples to be the first dimension of the input.
        For example, 10 data points need to be checked and each point
        has (3, 50, 50) resized or original input size, the shape of
        the array must be (10, 3, 50, 50).
        For more information, please see documentation.
        """
        # Handling the class of single
        if len(x_input.shape) == 3:
            x_input = (x_input).reshape(
                1, x_input.shape[0], x_input.shape[1], x_input.shape[2])

        x_input = torch.from_numpy(x_input).to(self.device)
        net_output = self.net.predict(x_input)

        # Handling the case of classification
        if self.criterion_input == "1":
            _, net_output = torch.max(net_output.data, 1)

        return net_output
