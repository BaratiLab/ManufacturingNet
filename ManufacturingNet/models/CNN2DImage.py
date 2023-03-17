import datetime
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

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
from tqdm import tqdm


def conv2D_output_size(img_size, kernel_size, stride, padding):
    outshape = (
        np.floor(
            (img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1
        ).astype(int),
        np.floor(
            (img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1
        ).astype(int),
    )
    return outshape


def spacing():

    print("=" * 25)


class BasicBlock(nn.Module):
    def __init__(self, in_channel, input_size, default=False):
        super(BasicBlock, self).__init__()

        self.input_size = input_size
        self.in_channel = in_channel
        self.get_channel_input()

        if default:
            self.kernel = (3, 3)
            self.stride = (1, 1)
            self.padding = (0, 0)
            self.pooling_input = False
        else:
            self.get_kernel_size()
            self.get_stride()
            self.get_padding()

        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=self.out_channel,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
        )
        self.input_size = conv2D_output_size(
            self.input_size, self.kernel, self.stride, self.padding
        )
        self.batch = nn.BatchNorm2d(self.out_channel, momentum=0.01)
        self.get_dropout()

        if default == False:
            self.get_pooling_layer()
            if self.pooling_input:
                self.input_size = conv2D_output_size(
                    self.input_size, (2, 2), (2, 2), (0, 0)
                )

    def get_pooling_layer(self):
        print("Question: Pooling layer: ")
        gate = 0
        self.pooling_qtn = input(
            "Do you want a pooling layer after this convolution layer (y/n): "
        ).replace(" ", "")
        while gate != 1:
            if (self.pooling_qtn).lower() == "y":
                self.pooling_input = True
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                gate = 1

            elif (self.pooling_qtn).lower() == "n":
                self.pooling_input = False
                gate = 1
            else:
                print("Please enter a valid input")
        spacing()

    def get_dropout(self):
        print("Question: Dropout value: ")
        # Method for getting dropout input for each hidden layer
        gate = 0
        while gate != 1:
            self.dropout_value = input(
                "Please enter a dropout value between 0 and 1 for this convolution. \n For default option of no dropout, please directly press enter without any input: "
            ).replace(" ", "")
            if self.dropout_value == "":
                print("Default value selected")
                self.drop = nn.Dropout2d(p=0)
                gate = 1

            if self.dropout_value.replace(".", "").isdigit():
                if float(self.dropout_value) >= 0 and float(self.dropout_value) < 1:
                    self.drop = nn.Dropout2d(p=float(self.dropout_value))
                    gate = 1
                else:
                    print("Please enter a valid input")
        spacing()

    def get_channel_input(self):
        print("Question: Out-channels: ")
        gate = 0
        while gate != 1:
            channel_input = (
                input("Please enter a value for number of out channels: ")
            ).replace(" ", "")
            if channel_input.isnumeric() and int(channel_input) > 0:
                self.out_channel = int(channel_input)
                gate = 1
            else:
                print("Please enter a valid input")
        spacing()

    def get_kernel_size(self):
        print("Question: Kernel Size: ")
        gate = 0
        while gate != 1:
            self.kernel = []
            kernel_input = (
                input(
                    "Please enter kernel size input (heigth,width)\nFor example 3,3\nFor default size, please directly press enter without any input: "
                )
            ).replace(" ", "")
            if kernel_input == "":
                print("Default Value selected")
                kernel_input = "3,3"
            kernel_input = kernel_input.split(",")
            if len(kernel_input) == 2:
                for i in range(len(kernel_input)):
                    if kernel_input[i].isnumeric() and (
                        1 <= int(kernel_input[i]) <= 20
                    ):
                        self.kernel.append(int(kernel_input[i]))
                self.kernel = tuple(self.kernel)
                if len(self.kernel) == 2:
                    gate = 1
                else:
                    print("Please enter a valid input")
            else:
                print("Please enter a valid input")
        spacing()

    def get_stride(self):
        print("Question: Stride value:")
        gate = 0
        while gate != 1:
            self.stride = []
            stride_input = (
                input(
                    "Please enter stride input (heigth,width)\nFor example 1,1\nFor default size, please directly press enter without any input: "
                )
            ).replace(" ", "")
            if stride_input == "":
                print("Default Value selected")
                stride_input = "1,1"
            stride_input = stride_input.split(",")
            if len(stride_input) == 2:
                for i in range(len(stride_input)):
                    if stride_input[i].isnumeric() and (
                        1 <= int(stride_input[i]) <= 20
                    ):
                        self.stride.append(int(stride_input[i]))
                self.stride = tuple(self.stride)
                if len(self.stride) == 2:
                    gate = 1
                else:
                    print("Please enter a valid input")
            else:
                print("Please enter a valid input")
        spacing()

    def get_padding(self):
        print("Question: Padding value:")
        gate = 0
        while gate != 1:
            self.padding = []
            padding_input = (
                input(
                    "Please enter a value for padding (heigth,width)\nFor example 1,1\nFor default size, please directly press enter without any input: "
                )
            ).replace(" ", "")
            if padding_input == "":
                print("Default Value selected")
                padding_input = "0,0"
            padding_input = padding_input.split(",")

            if len(padding_input) == 2:
                for i in range(len(padding_input)):
                    if padding_input[i].isnumeric() and (
                        0 <= int(padding_input[i]) <= 20
                    ):
                        self.padding.append(int(padding_input[i]))
                self.padding = tuple(self.padding)
                if len(self.padding) == 2:
                    gate = 1
                else:
                    print("Please enter a valid input")
            else:
                print("Please enter a valid input")
        spacing()

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.batch(out), inplace=True)

        if self.pooling_input:
            out = self.pool(out)
        out = self.drop(out)
        return out


class Network(nn.Module):
    def __init__(self, img_size, num_class):
        super(Network, self).__init__()

        self.num_class = num_class
        self.img_size = img_size[:-1]
        self.get_num_conv_layers()
        self.get_default_input()
        print("\n")
        self.conv_layers = []

        self.in_channel = img_size[-1]

        for i in range(self.num_conv_layers):
            print("\n")
            print("Designing the {} convolution block".format(i + 1))
            conv_layer = BasicBlock(self.in_channel, self.img_size, self.default_input)
            self.conv_layers.append(conv_layer)
            self.img_size = conv_layer.input_size
            print("The image size after this convolution is: ", self.img_size)
            print("\n")
            self.in_channel = conv_layer.out_channel

        final_layers = list(filter(None, self.conv_layers))
        self.net = nn.Sequential(*final_layers)

        # fully connected layer, output k classes
        self.fc1 = nn.Linear(
            self.in_channel * self.img_size[0] * self.img_size[1], 1024
        )
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, self.num_class)

    def get_default_input(self):
        print("3/8 - Default value:")

        self.default_input = input(
            "Do you want default values for convolution layers (y/n): "
        ).replace(" ", "")
        gate = 0

        while gate != 1:
            if (self.default_input).lower() == "y":
                self.default_input = True
                gate = 1

            elif (self.default_input).lower() == "n":
                self.default_input = False
                gate = 1

            else:
                print("Please enter a valid input")
        spacing()
        print("\n")

    def get_num_conv_layers(self):

        print("2/8 - Number of Convolution Layers:")

        gate = 0
        while gate != 1:
            conv_input = (input("Please enter the number of conv_layers: ")).replace(
                " ", ""
            )
            if conv_input.isnumeric() and int(conv_input) > 0:
                self.num_conv_layers = int(conv_input)
                gate = 1
            else:
                print("Please enter a valid input")
        spacing()

    def forward(self, x):

        x = self.net(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


# The following class will be called by a user. The class calls other necessary classes to build a complete pipeline required for training


class CNN2DImage:
    """
    Documentation Link:https://manufacturingnet.readthedocs.io/en/latest/

    """

    def __init__(self, train_data_address, val_data_address, shuffle=True):

        # Lists used in the functions below
        self.criterion_list = {
            1: nn.CrossEntropyLoss(),
            2: torch.nn.L1Loss(),
            3: torch.nn.SmoothL1Loss(),
            4: torch.nn.MSELoss(),
        }

        self.train_address = train_data_address
        self.val_address = val_data_address
        self.shuffle = shuffle

        self.get_default_paramters()  # getting default parameters argument

        self.num_classes = self.get_num_classes()  # getting the number of classes

        print("1/8 - Image size")
        self.get_image_size()  # getting the image size (resized or original)

        # building a network architecture
        self.net = (Network(self.img_size, self.num_classes)).double()

        print("=" * 25)
        print("4/8 - Batch size input")
        # getting a batch size for training and validation
        self._get_batchsize_input()

        print("=" * 25)
        print("5/8- Loss function")
        self._get_loss_function()  # getting a loss function

        print("=" * 25)
        print("6/8 - Optimizer")
        self._get_optimizer()  # getting an optimizer input

        print("=" * 25)
        print("7/8 - Scheduler")
        self._get_scheduler()  # getting a scheduler input

        self._set_device()  # setting the device to gpu or cpu

        print("=" * 25)
        print("8/8 - Number of epochs")
        self._get_epoch()  # getting an input for number oftraining epochs

        self.main()  # run function

    def get_default_paramters(self):

        # Method for getting a binary input for default paramters

        gate = 0
        while gate != 1:
            self.default = input(
                "Do you want default values for all the training parameters (y/n)? "
            ).replace(" ", "")
            if (
                self.default == "y"
                or self.default == "Y"
                or self.default == "n"
                or self.default == "N"
            ):
                if self.default.lower() == "y":
                    self.default_gate = True
                else:
                    self.default_gate = False
                gate = 1
            else:
                print("Enter a valid input")
                print(" ")
        print(" ")

    def check_address(self, address):

        isfile = os.path.isfile(address)

        return isfile

    def get_num_classes(self):

        train_num_folder = 0
        train_num_files = 0

        for _, dirnames, filenames in os.walk(self.train_address):
            train_num_folder += len(dirnames)
            train_num_files += len(filenames)

        if train_num_files == 0:
            print("Train data: Zero images found.\n System exit initialized")
            sys.exit()

        val_num_folder = 0
        val_num_files = 0

        for _, dirnames, filenames in os.walk(self.val_address):
            val_num_folder += len(dirnames)
            val_num_files += len(filenames)

        if val_num_files == 0:
            print("Validation data: Zero images found.\n System exit initialized")
            sys.exit()

        if train_num_folder != val_num_folder:
            print(
                "Warning: Number of folders in the Validation set and Training set is not the same."
            )

        print("Number of classes: ", train_num_folder)
        print("Total number of training images: ", train_num_files)
        print("Total number of validation images: ", val_num_files)
        spacing()
        return train_num_folder

    def get_image_size(self):

        gate = 0
        while gate != 1:
            self.img_size = []
            print("All the images must have same size.")
            size_input = (
                input(
                    "Please enter the dimensions to which images need to be resized (heigth, width, channels): \nFor example - 228, 228, 1 (For gray scale conversion)\n If all images have same size, enter the actual image size (heigth, width, channels) :\n "
                )
            ).replace(" ", "")

            size_input = size_input.split(",")
            if len(size_input) == 3:
                for i in range(len(size_input)):
                    if size_input[i].isnumeric() and (1 <= int(size_input[i])):
                        self.img_size.append(int(size_input[i]))
                self.img_size = tuple(self.img_size)
                if len(self.img_size) == 3 and (
                    self.img_size[-1] == 1 or self.img_size[-1] == 3
                ):
                    gate = 1
                else:
                    print(
                        "Please enter a valid input.\n Image size must be positive integers and number of channels can be 1 or 3"
                    )
            else:
                print("Please enter a valid input")
        spacing()

    def _get_batchsize_input(self):

        # Method for getting batch size input

        gate = 0
        while gate != 1:
            self.batchsize = (input("Please enter the batch size: ")).replace(" ", "")
            if self.batchsize.isnumeric() and int(self.batchsize) > 0:
                self.batchsize = int(self.batchsize)
                gate = 1
            else:
                print("Please enter a valid input")

    def _get_loss_function(self):

        # Method for getting a loss function for training

        self.criterion_input = "1"
        self.criterion = self.criterion_list[int(self.criterion_input)]
        print("Loss function: CrossEntropy()")

    def _get_optimizer(self):

        # Method for getting a optimizer input

        gate = 0
        while gate != 1:
            if self.default_gate == True:
                print("Default optimizer selected : Adam")
                self.optimizer_input = "1"
            else:
                self.optimizer_input = (
                    input(
                        "Please enter the optimizer index for the problem \n Optimizer_list - [1: Adam, 2: SGD] \n For default optimizer, please directly press enter without any input: "
                    )
                ).replace(" ", "")
            if self.optimizer_input == "":  # handling default case for optimizer
                print("Default optimizer selected : Adam")
                self.optimizer_input = "1"

            if (
                self.optimizer_input.isnumeric()
                and int(self.optimizer_input) > 0
                and int(self.optimizer_input) < 3
            ):
                gate = 1
            else:
                print("Please enter a valid input")
                print(" ")
        print(" ")
        gate = 0
        while gate != 1:
            if self.default_gate == True:
                print("Default value for learning rate selected : 0.001")
                self.user_lr = "0.001"
            else:
                self.user_lr = input(
                    "Please enter a required value float input for learning rate (learning rate > 0) \n For default learning rate, please directly press enter without any input: "
                ).replace(" ", "")
            if self.user_lr == "":  # handling default case for learning rate
                print("Default value for learning rate selected : 0.001")
                self.user_lr = "0.001"
            if self.user_lr.replace(".", "").isdigit():
                if float(self.user_lr) > 0:
                    self.lr = float(self.user_lr)
                    gate = 1
            else:
                print("Please enter a valid input")
                print(" ")

        self.optimizer_list = {
            1: optim.Adam(self.net.parameters(), lr=self.lr),
            2: optim.SGD(self.net.parameters(), lr=self.lr),
        }
        self.optimizer = self.optimizer_list[int(self.optimizer_input)]
        print(" ")

    def _get_scheduler(self):

        # Method for getting scheduler

        gate = 0
        while gate != 1:
            if self.default_gate == True:
                print("By default no scheduler selected")
                self.scheduler_input = "1"
            else:
                self.scheduler_input = input(
                    "Please enter the scheduler index for the problem: Scheduler_list - [1: None, 2:StepLR, 3:MultiStepLR] \n For default option of no scheduler, please directly press enter without any input: "
                ).replace(" ", "")
            if self.scheduler_input == "":
                print("By default no scheduler selected")
                self.scheduler_input = "1"
            if (
                self.scheduler_input.isnumeric()
                and int(self.scheduler_input) > 0
                and int(self.scheduler_input) < 4
            ):
                gate = 1
            else:
                print("Please enter a valid input")
                print(" ")

        if self.scheduler_input == "1":
            print(" ")
            self.scheduler = None

        elif self.scheduler_input == "2":
            print(" ")
            gate = 0
            while gate != 1:
                self.step = (
                    input("Please enter a step value int input (step > 0): ")
                ).replace(" ", "")
                if self.step.isnumeric() and int(self.step) > 0:
                    self.step = int(self.step)
                    gate = 1
                else:
                    print("Please enter a valid input")
                    print(" ")
            print(" ")
            gate = 0
            while gate != 1:
                self.gamma = (
                    input(
                        "Please enter a Multiplying factor value float input (Multiplying factor > 0): "
                    )
                ).replace(" ", "")
                if self.gamma.replace(".", "").isdigit():
                    if float(self.gamma) > 0:
                        self.gamma = float(self.gamma)
                        gate = 1
                else:
                    print("Please enter a valid input")
                    print(" ")

            self.scheduler = scheduler.StepLR(
                self.optimizer, step_size=self.step, gamma=self.gamma
            )

        elif self.scheduler_input == "3":
            print(" ")
            gate = 0
            while gate != 1:
                self.milestones_input = (
                    input(
                        "Please enter values of milestone epochs int input (Example: 2, 6, 10): "
                    )
                ).replace(" ", "")
                self.milestones_input = self.milestones_input.split(",")
                for i in range(len(self.milestones_input)):
                    if (
                        self.milestones_input[i].isnumeric()
                        and int(self.milestones_input[i]) > 0
                    ):
                        gate = 1
                    else:
                        gate = 0
                        break
                if gate == 0:
                    print("Please enter a valid input")
                    print(" ")

            self.milestones = [int(x) for x in self.milestones_input if int(x) > 0]
            print(" ")

            gate = 0
            while gate != 1:
                self.gamma = (
                    input(
                        "Please enter a Multiplying factor value float input (Multiplying factor > 0): "
                    )
                ).replace(" ", "")
                if self.gamma.replace(".", "").isdigit():
                    if float(self.gamma) > 0:
                        self.gamma = float(self.gamma)
                        gate = 1
                else:
                    print("Please enter a valid input")
                    print(" ")
            self.scheduler = scheduler.MultiStepLR(
                self.optimizer, milestones=self.milestones, gamma=self.gamma
            )

    def _set_device(self):

        # Method for setting device type if GPU is available

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def _get_epoch(self):

        # Method for getting number of epochs for training the model

        gate = 0
        while gate != 1:
            self.numEpochs = (
                input("Please enter the number of epochs to train the model: ")
            ).replace(" ", "")
            if self.numEpochs.isnumeric() and int(self.numEpochs) > 0:
                self.numEpochs = int(self.numEpochs)
                gate = 1
            else:
                print("Please enter a valid input")

    def main(self):

        # Method integrating all the functions and training the model

        self.net.to(self.device)
        print("=" * 25)

        print("Neural network architecture: ")
        print(" ")
        print(self.net)  # printing model architecture
        print("=" * 25)

        self.get_model_summary()  # printing summaray of the model
        print(" ")
        print("=" * 25)

        image_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=self.img_size[-1]),
                transforms.Resize((self.img_size[:-1]), interpolation=2),
                transforms.ToTensor(),
            ]
        )

        self.train_dataset = torchvision.datasets.ImageFolder(
            root=self.train_address, transform=image_transform
        )  # creating the training dataset

        self.val_dataset = torchvision.datasets.ImageFolder(
            root=self.val_address, transform=image_transform
        )  # creating the validation dataset

        # creating the training dataset dataloadet
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batchsize, shuffle=True
        )

        # creating the validation dataset dataloader
        self.dev_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batchsize
        )

        self.train_model()  # training the model

        self.get_loss_graph()  # saving the loss graph

        if self.criterion_input == "1":

            self.get_accuracy_graph()  # saving the accuracy graph
            self.get_confusion_matrix()  # printing confusion matrix

        self._save_model()  # saving model paramters

        print(" Call get_prediction() to make predictions on new data")
        print(" ")
        print("=== End of training ===")

    def _save_model(self):

        # Method for saving the model parameters if user wants to

        gate = 0
        while gate != 1:
            save_model = input(
                "Do you want to save the model weights? (y/n): "
            ).replace(" ", "")
            if save_model.lower() == "y" or save_model.lower() == "yes":
                path = "model_parameters.pth"
                torch.save(self.net.state_dict(), path)
                gate = 1
            elif save_model.lower() == "n" or save_model.lower() == "no":
                gate = 1
            else:
                print("Please enter a valid input")
        print("=" * 25)

    def get_model_summary(self):

        # Method for getting the summary of the model
        print("Model Summary:")
        print(" ")
        print("Criterion: ", self.criterion)
        print("Optimizer: ", self.optimizer)
        print("Scheduler: ", self.scheduler)
        print("Batch size: ", self.batchsize)
        print("Initial learning rate: ", self.lr)
        print("Number of training epochs: ", self.numEpochs)
        print("Device: ", self.device)

    def train_model(self):

        # Method for training the model

        self.net.train()
        self.training_loss = []
        self.training_acc = []
        self.dev_loss = []
        self.dev_accuracy = []
        total_predictions = 0.0
        correct_predictions = 0.0

        print("Training the model...")

        for epoch in tqdm(range(self.numEpochs), total=numEpochs):

            start_time = time.time()
            self.net.train()
            print("Epoch_Number: ", epoch)
            running_loss = 0.0

            for batch_idx, (data, target) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                data = data.double().to(self.device)
                target = target.to(self.device)
                outputs = self.net(data)

                # calculating the batch accuracy only if the loss function is Cross entropy
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

            # printing the epoch accuracy only if the loss function is Cross entropy
            if self.criterion_input == "1":

                acc = (correct_predictions / total_predictions) * 100.0
                self.training_acc.append(acc)
                print("Training Accuracy: ", acc, "%")

            dev_loss, dev_acc = self.validate_model()

            if self.scheduler_input != "1":

                self.scheduler.step()
                print("Current scheduler status: ", self.optimizer)

            end_time = time.time()
            print("Epoch Time: ", end_time - start_time, "s")
            print("#" * 50)

            self.dev_loss.append(dev_loss)

            # saving the epoch validation accuracy only if the loss function is Cross entropy
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

            data = data.double().to(self.device)
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
        print("Validation Loss: ", running_loss)

        # calculating and printing the epoch accuracy only if the loss function is Cross entropy
        if self.criterion_input == "1":

            acc = (correct_predictions / total_predictions) * 100.0
            print("Validation Accuracy: ", acc, "%")

        return running_loss, acc

    def get_loss_graph(self):

        # Method for showing and saving the loss graph in the root directory

        plt.figure(figsize=(8, 8))
        plt.plot(self.training_loss, label="Training Loss")
        plt.plot(self.dev_loss, label="Validation Loss")
        plt.legend()
        plt.title("Model Loss")
        plt.xlabel("Epochs")
        plt.ylabel("loss")
        plt.savefig("loss.png")

    def get_accuracy_graph(self):

        # Method for showing and saving the accuracy graph in the root directory

        plt.figure(figsize=(8, 8))
        plt.plot(self.training_acc, label="Training Accuracy")
        plt.plot(self.dev_accuracy, label="Validation Accuracy")
        plt.legend()
        plt.title("Model accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("acc")
        plt.savefig("accuracy.png")

    def get_confusion_matrix(self):

        # Method for getting the confusion matrix for classification problem
        print("Confusion Matix: ")

        result = confusion_matrix(
            np.concatenate(np.array(self.predict)),
            np.concatenate(np.array(self.actual)),
        )
        print(result)

    def get_prediction(self, x_input):
        """

        Pass in an input numpy array for making prediction.
        For passing multiple inputs, make sure to keep number of examples to be the first dimension of the input.
        For example, 10 data points need to be checked and each point has (3, 50, 50) resized or original input size, the shape of the array must be (10, 3, 50, 50).
        For more information, please see documentation.

        """

        # Method to use at the time of inference

        if len(x_input.shape) == 3:  # handling the case of single

            x_input = (x_input).reshape(
                1, x_input.shape[0], x_input.shape[1], x_input.shape[2]
            )

        x_input = torch.from_numpy(x_input).to(self.device)

        net_output = self.net.forward(x_input)

        if self.criterion_input == "1":  # handling the case of classification problem

            _, net_output = torch.max(net_output.data, 1)

        return net_output

    def get_mapping(self):
        mapped_labels = self.train_dataset.class_to_idx

        return mapped_labels
