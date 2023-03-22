import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.utils.data as data
from torch.autograd import Variable
from tqdm import tqdm


# The following class is used to create necessary inputs for dataset class and dataloader class used during training process
class ModelDataset():

    def __init__(self, X, Y, batchsize, valset_size, shuffle):

        self.x = X                      # Inputs
        self.y = Y                      # Labels
        self.batchsize = batchsize
        self.valset_size = valset_size
        self.shuffle = shuffle
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x, self.y, test_size=self.valset_size, shuffle=self.shuffle)

    def get_trainset(self):

        # Method for getting training set inputs and labels

        return self.x_train, self.y_train

    def get_valset(self):

        # Method for getting validation set inputs and labels

        return self.x_val, self.y_val

    def get_batchsize(self):

        # Method for getting batch size for training and validatioin

        return self.batchsize


class Dataset(data.Dataset):

    def __init__(self, X, Y):

        self.X = X
        self.Y = Y

    def __len__(self):

        return len(self.Y)

    def __getitem__(self, index):

        x_item = torch.from_numpy(self.X[index]).double()
        y_item = torch.from_numpy(np.array(self.Y[index]))

        return x_item, y_item

# The function that handles the size for the CNN


def conv2D_output_size(img_size, padding, kernel_size, stride, pool=2):
    # compute output shape of conv2D
    outshape = (np.floor(((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1)).astype(int),
                np.floor(((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1)).astype(int))
    return outshape

# The function that handles size if pooling layer is applied


def conv2D_pool_size(img_size, pool_size, stride):
    outshape = (np.floor(((img_size[0] - (pool_size[0] - 1) - 1) / stride[0] + 1)).astype(int),
                np.floor(((img_size[1] - (pool_size[1] - 1) - 1) / stride[1] + 1)).astype(int))
    return outshape

# This is class that creates a CNN block based on the parameters that the user defines. The block is called repeatedly CNN2D class


class CNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pool_size, pool_stride, batch_norm=True, last=False, pooling=False):
        super(CNNBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.last = last
        self.pooling = pooling
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        if self.batch_norm == True:
            self.bn1 = nn.BatchNorm2d(out_channels)
        if self.pooling == True:
            self.pool = nn.MaxPool2d(pool_size, pool_stride)

    def forward(self, x):

        out = self.conv1(x)
        if self.batch_norm == True:
            out = self.bn1(out)
        out = self.relu(out)
        if self.pooling == True:
            out = self.pool(out)
        if self.last == True:
            out = out.view(out.size(0), -1)
        return out
# 2D CNN train from scratch (no transfer learning)
# This is the main code used to create the CNN network. The object of this class will be passed to the to the another class to develop a complete model


class CNN2D(nn.Module):
    def __init__(self, block):
        super(CNN2D, self).__init__()
        self.block = block

        print('1/15 - Get 2D folded signal Size')
        self.img_x, self.img_y = self.get_image_size()
        print("Image: ", (self.img_x), (self.img_y))
        print("="*25)

        print('2/15 - Number of Convolutions')
        self.n_conv = self.get_number_convolutions()
        print("Convolutions: ", self.n_conv)
        print("="*25)

        print('3/15 - Channels')
        self.channels = self.get_channels(self.n_conv)
        print("Channels: ", self.channels)
        print("="*25)

        print('4/15 - Kernels')
        self.kernel_size = self.get_kernel_size(self.n_conv)
        print("Kernel sizes: ", self.kernel_size)
        print("="*25)

        print('5/15 - Padding and Stride')
        self.padding, self.stride = self.get_stride_padding(self.n_conv)
        print("Padding: ", self.padding)
        print("Stride: ", self.stride)
        print("="*25)

        print("6/15 - Dropout")
        self.drop = self.get_dropout()
        print('Dropout ratio: ', self.drop)
        print("="*25)

        print("7/15 - Max Pooling")
        self.pooling_list, self.pooling_size, self.pooling_stride = self.get_pooling(
            self.n_conv)
        print("Pooling Layers: ", self.pooling_list)
        print("Pooling Size: ", self.pooling_size)
        print("Pooling Stride", self.pool_stride)
        print("="*25)

        print("8/15 - Batch Normalization")
        self.batch_normalization_list = self.get_batch_norm(self.n_conv)
        print("Batch normalization", self.batch_normalization_list)
        print("="*25)

        print("9/15 - Number of classes")
        self.num_classes = self.get_number_classes()
        print("="*25)

        self.make_CNN(self.block, self.n_conv, self.pooling_list,
                      self.drop, self.batch_normalization_list)
        # print(self.net)# not returning from make_CNN purposely

# the gates are added to the code in cases where the user accidentally inputs an unacceptable parameter
    def get_number_classes(self):  # get number of classes
        gate = 0
        while gate != 1:
            self.num_classes = (input(
                "Please enter the number of classes \n Enter 1 if you are dealing with a regression problem: ")).replace(' ','')
            if (self.num_classes.isnumeric() and int(self.num_classes) > 0):
                gate = 1
            else:
                gate = 0
                print(
                    "Please enter valid number of classes.  The value must be an integer and greater than zero")
        return int(self.num_classes)

    def get_image_size(self):  # Get image size as the input from user
        gate = 0
        while gate != 1:
            self.img_x = (input("Please enter the size of the first dimension of the folded 2D signal: ")).replace(' ','')
            self.img_y = (input("Please enter the size of the second dimension of the folded 2D signal: ")).replace(' ','')
            if (self.img_x.isnumeric() and int(self.img_x) > 0) and (self.img_y.isnumeric() and int(self.img_y) > 0):
                gate = 1
            else:
                gate = 0
                print("Please enter valid numeric output")
        return int(self.img_x), int(self.img_y)

    # Get number of convolutions as imput from user
    def get_number_convolutions(self):
        gate = 0
        while gate != 1:
            self.n_conv = (input("Please enter the number of convolutions: ")).replace(' ','')
            if (self.n_conv.isnumeric() and int(self.n_conv) > 0):
                gate = 1
            else:
                gate = 0
                print(
                    "Please enter valid number of convolutions.  The value must be an integer and greater than zero")
        return int(self.n_conv)

    def get_channels(self, n_conv):  # Get the number of convolutions
        gate = 0
        gate1 = 0
        self.channels = []
        while gate1 != 1:
            channel_inp = ((input("enter the number of input channels: "))).replace(' ','')
            if (channel_inp.isnumeric() and int(channel_inp) > 0):
                self.channels.append(int(channel_inp))
                gate1 = 1
            else:
                print("Please enter valid number of channels.  The value must be an integer and greater than zero")
                gate1 = 0

        while gate != 1:
            for i in range(n_conv):
                channel = (
                    (input("enter the number of output channels for convolution {}: ".format(i+1)))).replace(' ','')
                if (channel.isnumeric() and int(channel) > 0):
                    self.channels.append(int(channel))
                    if i == n_conv-1:
                        gate = 1
                else:
                    gate = 0
                    print(
                        "Please enter valid number of channels.  The value must be an integer and greater than zero")
                    self.channels = []
                    break

        return self.channels

    def get_kernel_size(self, n_conv):  # Get the kernel size
        gate1 = 0
        value = input(
            "Do you want default values for kernel size(press y or n): ").replace(' ','')
        while gate1 != 1:
            if value == "Y" or value == "y" or value == 'n' or value == 'N':
                gate1 = 1
            else:
                print("Please enter valid input it should only be (y or n)")
                value = input(
                    "Do you want default values for kernel size(press y or n)")
                gate1 = 0

        gate2 = 0
        self.kernel_list = []
        while gate2 != 1:
            for i in range(n_conv):
                if value == 'N' or value == 'n':
                    k_size = (
                        ((input("Enter the kernel size for convolutional layer {} \n For Example: 3,3: ".format(i+1))))).replace(' ','')
                    k_split = k_size.split(",")
                    # print(len(k_split), "is the length of k_split") # for debugging 
                    if len(k_split) != 2:
                        gate2 = 0
                        print(
                            "Please enter valid kernel size.  The value must be in the form of 3,3")
                        # self.kernel_list = []
                        break
                    if k_split[0].isnumeric() and int(k_split[0]) > 0 and k_split[1].isnumeric() and int(k_split[1]) > 0:
                        self.kernel_list.append(
                            (int(k_split[0]), int(k_split[1])))
                        if i == n_conv-1:
                            gate2 = 1
                    else:
                        gate2 = 0
                        print(
                            "Please enter valid numeric values.  The value must be an integer and greater than zero")
                        self.kernel_list = []
                        break
                else:
                    self.kernel_list.append((3, 3))
                    if i == n_conv-1:
                        gate2 = 1
        return self.kernel_list

    def get_stride_padding(self, n_conv):  # Get the stride and padding
        self.padding = []
        self.stride = []
        gate1 = 0
        value = input(
            "Do you want default values for padding and stride (press y or n): ").replace(' ','')
        while gate1 != 1:
            if (value == "Y" or value == "y" or value == 'n' or value == 'N'):
                gate1 = 1
            else:
                print("Please enter valid input it should only be (y or n)")
                value = input(
                    "Do you want default values for padding and stride(press y or n): ").replace(' ','')
                gate1 = 0

        gate2 = 0
        while gate2 != 1:
            for i in range(n_conv):
                if value == 'N' or value == 'n':
                    pad_size = input(
                        "Enter padding for the image for convolutional layer {}  \n For Example 2,2: ".format(i+1)).replace(' ','')
                    pad_split = pad_size.split(",")
                    if pad_split[0].isnumeric() and int(pad_split[0]) >= 0 and pad_split[1].isnumeric() and int(pad_split[1]) >= 0:
                        self.padding.append(
                            (int(pad_split[0]), int(pad_split[1])))
                        if i == n_conv-1:
                            gate2 = 1
                    else:
                        gate2 = 0
                        print(
                            "Please enter valid numeric values.  The value must be an integer and greater than or equal to zero")
                        self.padding = []
                        break
                else:
                    self.padding.append((0, 0))
                    if i == n_conv-1:
                        gate2 = 1

        gate3 = 0
        while gate3 != 1:
            for i in range(n_conv):
                if value == 'N' or value == 'n':
                    stride_size = input(
                        "Enter stride for the convolutions for convolutional layer {} \n For Example 2,2: ".format(i+1)).replace(' ','')
                    stride_split = stride_size.split(",")
                    if stride_split[0].isnumeric() and int(stride_split[0]) >= 0 and stride_split[1].isnumeric() and int(stride_split[1]) >= 0:
                        self.stride.append(
                            (int(stride_split[0]), int(stride_split[1])))
                        if i == n_conv-1:
                            gate3 = 1
                    else:
                        gate3 = 0
                        print(
                            "Please enter valid numeric values.  The value must be an integer and greater than zero")
                        self.stride = []
                        break
                else:
                    self.stride.append((1, 1))
                    if i == n_conv-1:
                        gate3 = 1
        return self.padding, self.stride

    def get_batch_norm(self, n_conv):  # Get input for batch normalization
        gate1 = 0
        value = input(
            "Do you want default values for batch normalization (press y or n): ").replace(' ','')
        while gate1 != 1:
            if (value == "Y" or value == "y" or value == 'n' or value == 'N'):
                gate1 = 1
            else:
                print("Please enter valid input it should only be (y or n)")
                value = input(
                    "Do you want default values for batch normalization (press y or n): ")
                gate1 = 0

        self.batch_norm = []

        gate1 = 0
        while gate1 != 1:
            for i in range(n_conv):
                if value == "N" or value == 'n':
                    batch_boolean = (input(
                        "Please enter 0(No) or 1(yes) for using batch normalization in convolutional layer {} : ".format(i+1))).replace(' ','')
                    if (batch_boolean.isnumeric() and (int(batch_boolean) == 0 or int(batch_boolean) == 1)):
                        self.batch_norm.append(int(batch_boolean))
                        gate1 = 1
                    else:
                        gate1 = 0
                        print("Please enter valid numeric values")
                        self.batch_norm = []
                        break
                elif (value == "Y" or value == 'y'):
                    self.batch_norm.append(1)
                    if i == n_conv-1:
                        gate1 = 1

        return self.batch_norm

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

        gate = 0
        if value == 'N' or value == 'n':
            drop_out = (input(("Please input the dropout probability: "))).replace(' ','')
            while gate != 1:
                if drop_out.replace('.', '').isdigit():
                    if (float(drop_out) >= 0 and float(drop_out) < 1):
                        self.drop = drop_out
                        gate = 1
                    else:
                        print(
                            "Please enter the valid numeric values. The value should lie between 0 and 1 [0, 1) ")
                        drop_out = (
                            input(("Please input the dropout probability"))).replace(' ','')
                        gate = 0
                else:
                    drop_out = (
                        input(("Please input the dropout probability: "))).replace(' ','')
        else:
            self.drop = 0

        return float(self.drop)

    def get_pooling(self, n_conv):  # get input for pooling from the user
        gate1 = 0
        value = value = input(
            "Do you want default pooling values (press y or n): ").replace(' ','')
        while gate1 != 1:
            if value == "Y" or value == "y" or value == 'n' or value == 'N':
                gate1 = 1
            else:
                print("Please enter valid input it should only be (y or n)")
                value = input(
                    "Do you want default pooling values (press y or n): ").replace(' ','')
                gate1 = 0

        gate2 = 0
        self.pool_bool = []
        self.pool_size = []
        self.pool_stride = []
        while gate2 != 1:
            for i in range(n_conv):
                if value == "N" or value == 'n':
                    pool_boolean = (input(
                        "Please enter 0(No) or 1(yes) for using pooling in convolutional layer {} : ".format(i+1))).replace(' ','')
                    if (pool_boolean.isnumeric() and (int(pool_boolean) == 0 or int(pool_boolean) == 1)):
                        self.pool_bool.append(int(pool_boolean))
                        if i == n_conv-1:
                            gate2 = 1
                    else:
                        gate2 = 0
                        print("Please enter valid numeric values")
                        self.pool_bool = []
                        break
                elif (value == "Y" or value == 'y'):
                    if i <= n_conv - 2:
                        self.pool_bool.append(0)
                    elif i > n_conv-2:
                        self.pool_bool.append(1)
                        if i == n_conv-1:
                            gate2 = 1

        gate3 = 0
        while gate3 != 1:
            for i in range(len(self.pool_bool)):
                if value == 'N' or value == 'n':
                    if self.pool_bool[i] == 0:
                        self.pool_size.append((0, 0))
                        gate3 = 1
                    else:
                        pooling_size = input(
                            "Please enter pool size for convolutional layer {} \n For example 2,2: ".format(i+1)).replace(' ','')
                        pooling_size_split = pooling_size.split(',')
                        if (pooling_size_split[0].isnumeric() and int(pooling_size_split[0]) > 0 and pooling_size_split[1].isnumeric() and int(pooling_size_split[1]) > 0):
                            self.pool_size.append(
                                (int(pooling_size_split[0]), int(pooling_size_split[1])))
                            if i == len(self.pool_bool) - 1:
                                gate3 = 1
                        else:
                            gate3 = 0
                            print(
                                "please enter valid numeric values. The value must be an integer and greater than zero")
                            self.pool_size = []
                            break
                else:
                    self.pool_size.append((2, 2))
                    if i == len(self.pool_bool) - 1:
                        gate3 = 1

        gate4 = 0
        while gate4 != 1:
            for i in range(len(self.pool_bool)):
                if value == 'N' or value == 'n':
                    if self.pool_bool[i] == 0:
                        self.pool_stride.append((0, 0))
                        gate4 = 1
                    else:
                        pooling_stride = input(
                            "Please enter pool stride for convolutional layer {} \n For example 2,2: ".format(i+1)).replace(' ','')
                        pooling_stride_split = pooling_stride.split(',')
                        if (pooling_stride_split[0].isnumeric() and int(pooling_stride_split[0]) > 0 and pooling_stride_split[1].isnumeric() and int(pooling_stride_split[1]) > 0):
                            self.pool_stride.append(
                                (int(pooling_stride_split[0]), int(pooling_stride_split[1])))
                            if i == len(self.pool_bool) - 1:
                                gate4 = 1
                        else:
                            gate4 = 0
                            print(
                                "please enter valid numeric values. The value must be an integer and greater than zero")
                            self.pool_stride = []
                            break
                else:
                    self.pool_stride.append((2, 2))
                    if i == len(self.pool_bool) - 1:
                        gate4 = 1

        return self.pool_bool, self.pool_size, self.pool_stride

    # Makes the CNN with forward pass
    def make_CNN(self, block, n_conv, pool_list, drop, batch_norm_list):
        layers = []
        for i in range(n_conv):
            if pool_list[i] == 0:
                if i < n_conv-1:
                    if batch_norm_list[i] == 0:
                        layers.append(block(in_channels=self.channels[i], out_channels=self.channels[i+1], kernel_size=self.kernel_size[i], stride=self.stride[i],
                                            pool_size=self.pooling_size[i], pool_stride=self.pooling_stride[i], padding=self.padding[i], batch_norm=False, last=False, pooling=False))
                    else:
                        layers.append(block(in_channels=self.channels[i], out_channels=self.channels[i+1], kernel_size=self.kernel_size[i], stride=self.stride[i],
                                            pool_size=self.pooling_size[i], pool_stride=self.pooling_stride[i], padding=self.padding[i], batch_norm=True, last=False, pooling=False))
                else:
                    if batch_norm_list[i] == 0:
                        layers.append(block(in_channels=self.channels[i], out_channels=self.channels[i+1], kernel_size=self.kernel_size[i], stride=self.stride[i],
                                            pool_size=self.pooling_size[i], pool_stride=self.pooling_stride[i], padding=self.padding[i], batch_norm=False, last=True, pooling=False))
                    else:
                        layers.append(block(in_channels=self.channels[i], out_channels=self.channels[i+1], kernel_size=self.kernel_size[i], stride=self.stride[i],
                                            pool_size=self.pooling_size[i], pool_stride=self.pooling_stride[i], padding=self.padding[i], batch_norm=True, last=True, pooling=False))
            else:
                if i < n_conv-1:
                    if batch_norm_list[i] == 0:
                        layers.append(block(in_channels=self.channels[i], out_channels=self.channels[i+1], kernel_size=self.kernel_size[i], stride=self.stride[i],
                                            pool_size=self.pooling_size[i], pool_stride=self.pooling_stride[i], padding=self.padding[i], batch_norm=False, last=False, pooling=True))
                    else:
                        layers.append(block(in_channels=self.channels[i], out_channels=self.channels[i+1], kernel_size=self.kernel_size[i], stride=self.stride[i],
                                            pool_size=self.pooling_size[i], pool_stride=self.pooling_stride[i], padding=self.padding[i], batch_norm=True, last=False, pooling=True))
                else:
                    if batch_norm_list[i] == 0:
                        layers.append(block(in_channels=self.channels[i], out_channels=self.channels[i+1], kernel_size=self.kernel_size[i], stride=self.stride[i],
                                            pool_size=self.pooling_size[i], pool_stride=self.pooling_stride[i], padding=self.padding[i], batch_norm=False, last=True, pooling=True))
                    else:
                        layers.append(block(in_channels=self.channels[i], out_channels=self.channels[i+1], kernel_size=self.kernel_size[i], stride=self.stride[i],
                                            pool_size=self.pooling_size[i], pool_stride=self.pooling_stride[i], padding=self.padding[i], batch_norm=True, last=True, pooling=True))


        # print(pool_list)
        if pool_list[0] == 1:
            conv_shape = conv2D_output_size(
            (self.img_x, self.img_y), self.padding[0], self.kernel_size[0], self.stride[0])
            conv_shape_pool = conv2D_pool_size(
                conv_shape, self.pooling_size[0], self.pooling_stride[0])
            shape = [conv_shape_pool]
        else:
            conv_shape = conv2D_output_size(
            (self.img_x, self.img_y), self.padding[0], self.kernel_size[0], self.stride[0])
            shape = [conv_shape]

        for i in range(1, n_conv):
            if pool_list[i] == 1:
                conv_shape_rep = conv2D_output_size(
                    shape[i-1], self.padding[i], self.kernel_size[i], self.stride[i])
                conv_shape_pool = conv2D_pool_size(
                    conv_shape_rep, self.pooling_size[i], self.pooling_stride[i])
                shape.append(conv_shape_pool)
            else:
                conv_shape_rep = conv2D_output_size(
                    shape[i-1], self.padding[i], self.kernel_size[i], self.stride[i])
                shape.append(conv_shape_rep)
        print("Shapes after the Convolutions", shape)
        print("="*25)

        linear_size = self.channels[-1] * shape[-1][0] * shape[-1][1]
        layers.append(nn.Linear(linear_size, int(linear_size/2)))
        layers.append(nn.Linear(int(linear_size/2), int(linear_size/4)))
        layers.append(nn.Dropout(p=drop))
        layers.append(nn.Linear(int(linear_size/4), self.num_classes))
        self.ConvNet = nn.Sequential(*layers)

    def forward(self, x):
        return self.ConvNet(x)


class CNN2DSignal(object):
    """
    Documentation Link:https://manufacturingnet.readthedocs.io/en/latest/
    """

    def __init__(self, X, Y, shuffle=True):

        # Lists used in the functions below
        self.criterion_list = {1: nn.CrossEntropyLoss(), 
                               2: torch.nn.L1Loss(), 
                               3: torch.nn.SmoothL1Loss(), 
                               4: torch.nn.MSELoss()}

        self.x_data = X
        self.y_data = Y
        self.shuffle = shuffle
        #self.num_classes = num_classes
        # building a network architecture
        self.net = CNN2D(CNNBlock).double()
        # print(self.net.parameters())

        print('10/15 - Batch size input')
        # getting a batch size for training and validation
        self._get_batchsize_input()
        print("Batch Size: ", self.batchsize)
        print('='*25)

        print('11/15 - Validation set size')
        self._get_valsize_input()                # getting a train-validation split
        # splitting the data into training and validation sets
        self.model_data = ModelDataset(
            self.x_data, self.y_data, batchsize=self.batchsize, 
            valset_size=self.valset_size, shuffle=self.shuffle)
        print("Validation set ratio: ", self.valset_size)
        print('='*25)

        print('12/15 - Loss function')
        self._get_loss_function()               # getting a loss function
        print("Loss Function: ", self.criterion)
        print('='*25)

        print('13/15 - Optimizer')
        self._get_optimizer()               # getting an optimizer input
        print("Optimizer: ", self.optimizer)
        print('='*25)

        print('14/15 - Scheduler')
        self._get_scheduler()               # getting a scheduler input
        print("Scheduler: ", self.scheduler)
        print('='*25)

        self._set_device()              # setting the device to gpu or cpu

        print('15/15 - Number of epochs')
        self._get_epoch()           # getting an input for number oftraining epochs
        print("Epochs: ", self.numEpochs)

        self.main()             # run function

    def _get_batchsize_input(self):

        # Method for getting batch size input

        gate = 0
        while gate != 1:
            self.batchsize = (input('Please enter the batch size: ')).replace(' ','')
            if self.batchsize.isnumeric() and int(self.batchsize) > 0:
                self.batchsize = int(self.batchsize)
                gate = 1
            else:
                print(
                    'Please enter a valid numeric input. The value must be an integer and greater than zero')

    def _get_valsize_input(self):

        # Method for getting validation set size input

        gate = 0
        while gate != 1:
            self.valset_size = (input(
                'Please enter the validation set size (size > 0 and size < 1) \n For default size, please directly press enter without any input: ')).replace(' ','')
            if self.valset_size == '':              # handling default case for valsize
                print('Default value selected')
                self.valset_size = '0.2'
            if self.valset_size.replace('.', '').isdigit():
                if float(self.valset_size) > 0 and float(self.valset_size) < 1:
                    self.valset_size = float(self.valset_size)
                    gate = 1
            else:
                print('Please enter a valid numeric input')

    def _get_loss_function(self):

        # Method for getting a loss function for training

        gate = 0
        while gate != 1:
            self.criterion_input = (input(
                'Please enter the appropriate loss function for the problem: \n Criterion_list - [1: CrossEntropyLoss, 2: L1Loss, 3: SmoothL1Loss, 4: MSELoss]: ')).replace(' ','')

            if self.criterion_input.isnumeric() and int(self.criterion_input) < 5 and int(self.criterion_input) > 0:
                gate = 1
            else:
                print('Please enter a valid numeric input')

        self.criterion = self.criterion_list[int(self.criterion_input)]

    def _get_optimizer(self):

        # Method for getting a optimizer input

        gate = 0
        while gate != 1:
            self.optimizer_input = (input(
                'Please enter the optimizer for the problem \n Optimizer_list - [1: Adam, 2: SGD] \n For default optimizer, please directly press enter without any input: ')).replace(' ','')
            if self.optimizer_input == '':              # handling default case for optimizer
                print('Default optimizer selected')
                self.optimizer_input = '1'

            if self.optimizer_input.isnumeric() and int(self.optimizer_input) > 0 and int(self.optimizer_input) < 3:
                gate = 1
            else:
                print('Please enter a valid numeric input')

        gate = 0
        while gate != 1:
            self.user_lr = input(
                'Please enter a required postive value for learning rate \n For default learning rate, please directly press enter without any input: ').replace(' ','')
            if self.user_lr == '':               # handling default case for learning rate
                print('Default value selected')
                self.user_lr = '0.001'
            if self.user_lr.replace('.', '').isdigit():
                if float(self.user_lr) > 0:
                    self.lr = float(self.user_lr)
                    gate = 1
            else:
                print('Please enter a valid input')

        self.optimizer_list = {1: optim.Adam(self.net.parameters(), lr=self.lr), 
                               2: optim.SGD(self.net.parameters(), lr=self.lr)}
        self.optimizer = self.optimizer_list[int(self.optimizer_input)]

    def _get_scheduler(self):

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
                print('Please enter a valid numeric input')

        if self.scheduler_input == '1':
            self.scheduler = None

        elif self.scheduler_input == '2':
            gate = 0
            while gate != 1:
                self.step = (input('Please enter a step value: ')).replace(' ','')
                if self.step.isnumeric() and int(self.step) > 0:
                    self.step = int(self.step)
                    gate = 1
                else:
                    print('Please enter a valid numeric input')
            print(' ')
            gate = 0
            while gate != 1:
                self.gamma = (
                    input('Please enter a gamma value (Multiplying factor): ')).replace(' ','')
                if self.gamma.replace('.', '').isdigit():
                    if float(self.gamma) > 0:
                        self.gamma = float(self.gamma)
                        gate = 1
                else:
                    print('Please enter a valid numeric input')

            self.scheduler = scheduler.StepLR(
                self.optimizer, step_size=self.step, gamma=self.gamma)

        elif self.scheduler_input == '3':
            gate = 0
            while gate != 1:
                self.milestones_input = (
                    input('Please enter values of milestone epochs: ')).replace(' ','')
                self.milestones_input = self.milestones_input.split(',')
                for i in range(len(self.milestones_input)):
                    if self.milestones_input[i].isnumeric() and int(self.milestones_input[i]) > 0:
                        gate = 1
                    else:
                        gate = 0
                        break
                if gate == 0:
                    print('Please enter a valid numeric input')

            self.milestones = [int(x)
                               for x in self.milestones_input if int(x) > 0]
            print(' ')

            gate = 0
            while gate != 1:
                self.gamma = (
                    input('Please enter a gamma value (Multiplying factor): ')).replace(' ','')
                if self.gamma.replace('.', '').isdigit():
                    if float(self.gamma) > 0:
                        self.gamma = float(self.gamma)
                        gate = 1
                else:
                    print('Please enter a valid input')
            self.scheduler = scheduler.MultiStepLR(
                self.optimizer, milestones=self.milestones, gamma=self.gamma)

    def _set_device(self):

        # Method for setting device type if GPU is available

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def _get_epoch(self):

        # Method for getting number of epochs for training the model

        gate = 0
        while gate != 1:
            self.numEpochs = (
                input('Please enter the number of epochs to train the model: ')).replace(' ','')
            if self.numEpochs.isnumeric() and int(self.numEpochs) > 0:
                self.numEpochs = int(self.numEpochs)
                gate = 1
            else:
                print(
                    'Please enter a valid numeric input. The number must be integer and greater than zero')

    def main(self):

        # Method integrating all the functions and training the model
        print(self.net)
        self.net.to(self.device)
        print('='*25)

        print('Neural network architecture: ')
        print(' ')
        print(self.net)         # printing model architecture
        print('='*25)

        self.get_model_summary()        # printing summaray of the model
        print(' ')
        print('='*25)

        # getting inputs and labels for training set
        xt, yt = self.model_data.get_trainset()

        # getting inputs and labels for validation set
        xv, yv = self.model_data.get_valset()

        # creating the training dataset
        self.train_dataset = Dataset(xt, yt)

        # creating the validation dataset
        self.val_dataset = Dataset(xv, yv)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, 
                                                        batch_size=self.model_data.get_batchsize(), 
                                                        shuffle=True)           # creating the training dataset dataloadet

        # creating the validation dataset dataloader
        self.dev_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.model_data.get_batchsize())

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
        while gate != 1:
            save_model = input(
                'Do you want to save the model weights? (y/n): ')
            if save_model.lower() == 'y' or save_model.lower() == 'yes':
                path = 'model_parameters.pth'
                torch.save(self.net.state_dict(), path)
                gate = 1
            elif save_model.lower() == 'n' or save_model.lower() == 'no':
                gate = 1
            else:
                print('Please enter a valid input')
        print('='*25)

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

        print('Training the model...')

        for epoch in tqdm(range(self.numEpochs), total=self.numEpochs, unit='epoch'):

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