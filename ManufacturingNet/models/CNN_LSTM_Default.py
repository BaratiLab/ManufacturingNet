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


class ModelDataset():
    """ModelDataset creates necessary inputs for the Dataset and
    Dataloader classes used during training.
    """

    def __init__(self, X, Y, batchsize, valset_size, shuffle):
        self.x = X
        self.y = Y
        self.batchsize = batchsize
        self.valset_size = valset_size
        self.shuffle = shuffle
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x, self.y, test_size=self.valset_size, shuffle=self.shuffle)

    def get_trainset(self):
        return self.x_train, self.y_train

    def get_valset(self):
        return self.x_val, self.y_val

    def get_batchsize(self):
        return self.batchsize


class Dataset(data.Dataset):
    """Dataset creates a datatset using standard pytorch functionality.
    """

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        x_item = torch.from_numpy(self.X[index]).double()
        y_item = torch.from_numpy(np.array(self.Y[index])).double()
        return x_item, y_item


def conv2D_output_size(img_size, padding, kernel_size, stride, pool=2):
    outshape = (np.floor(((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1)).astype(int),
                np.floor(((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1)).astype(int))
    return outshape


def conv2D_pool_size(img_size, pool_size, stride):
    outshape = (np.floor(((img_size[0] - (pool_size[0] - 1) - 1) / stride[0] + 1)).astype(int),
                np.floor(((img_size[1] - (pool_size[1] - 1) - 1) / stride[1] + 1)).astype(int))
    return outshape


class CNNBlock(nn.Module):
    """CNNBlock creates a CNN block with the user's parameters. This
    block is repeatedly referred to in the CNN2D class.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 pool_size, pool_stride, batch_norm=True, last=False,
                 pooling=False):
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


class CNN2D(nn.Module):
    """CNN2D trains a 2D CNN model from scratch. CNN2D serves as a basis
    for creating CNN networks. CNN2D objects will be passed to other
    classes to develop complete models.
    """

    def __init__(self, block):
        super(CNN2D, self).__init__()
        self.block = block

        print("=" * 25)
        print("")
        print("Convolutional Neural Network")
        print("")

        print("1/18 - Get Image Size")
        self.img_x, self.img_y = self.get_image_size()
        print("Image:", (self.img_x), (self.img_y))
        print("=" * 25)

        print("2/18 - Number of Convolutions")
        self.n_conv = self.get_number_convolutions()
        print("Convolutions:", self.n_conv)
        print("=" * 25)

        print("3/18 - Channels")
        self.channels = self.get_channels(self.n_conv)
        print("Channels:", self.channels)
        print("=" * 25)

        print("4/18 - Kernels")
        self.kernel_size = self.get_kernel_size(self.n_conv)
        print("Kernel sizes:", self.kernel_size)
        print("=" * 25)

        print("5/18 - Padding and Stride")
        self.padding, self.stride = self.get_stride_padding(self.n_conv)
        print("Padding:", self.padding)
        print("Stride:", self.stride)
        print("=" * 25)

        print("6/18 - Dropout")
        self.drop = self.get_dropout()
        print("Dropout ratio:", self.drop)
        print("=" * 25)

        print("7/18 - Max Pooling")
        self.pooling_list, self.pooling_size, self.pooling_stride = \
            self.get_pooling(self.n_conv)
        print("Pooling Layers:", self.pooling_list)
        print("Pooling Size:", self.pooling_size)
        print("Pooling Stride", self.pool_stride)
        print("=" * 25)

        print("8/18 - Batch Normalization")
        self.batch_normalization_list = self.get_batch_norm(self.n_conv)
        print("Batch normalization", self.batch_normalization_list)
        print("=" * 25)

        self.make_CNN(self.block, self.n_conv, self.pooling_list,
                      self.drop, self.batch_normalization_list)

    def get_image_size(self):
        gate = 0

        while gate != 1:
            self.img_x = input("Please enter the image width: ").replace(' ','')
            self.img_y = input("Please enter the image height: ").replace(' ','')
            if (self.img_x.isnumeric() and int(self.img_x) > 0) and (self.img_y.isnumeric() and int(self.img_y) > 0):
                gate = 1
            else:
                print("Please enter valid numeric output.")

        return int(self.img_x), int(self.img_y)

    def get_number_convolutions(self):
        gate = 0

        while gate != 1:
            self.n_conv = input("Please enter the number of convolutions: ").replace(' ','')
            if self.n_conv.isnumeric() and int(self.n_conv) > 0:
                gate = 1
            else:
                print("Please enter valid number of convolutions.")
                print("The value must be an integer greater than zero.")

        return int(self.n_conv)

    def get_channels(self, n_conv):
        gate = 0
        gate1 = 0
        self.channels = []

        while gate != 1:
            channel_inp = input("enter the number of input channels: ").replace(' ','')
            if (channel_inp.isnumeric() and int(channel_inp) > 0):
                self.channels.append(int(channel_inp))
                gate = 1
            else:
                print("Please enter valid number of channels.")
                print("The value must be an integer greater than zero.")
                gate = 0

        while gate1 != 1:
            for i in range(n_conv):
                channel = \
                    input(
                        "enter the number of output channels for convolution {}: ".format(i + 1)).replace(' ','')
                if (channel.isnumeric() and int(channel) > 0):
                    self.channels.append(int(channel))
                    if i == n_conv - 1:
                        gate1 = 1
                else:
                    gate1 = 0
                    print("Please enter valid number of channels.")
                    print("The value must be an integer greater than zero.")
                    self.channels = []
                    break

        return self.channels

    def get_kernel_size(self, n_conv):
        gate = 0

        while gate != 1:
            value = input("Use default values for kernel size (y/n)? ").lower().replace(' ','')

            if value in {"y", "n"}:
                gate = 1
            else:
                print("Please enter valid input.")

        gate = 0
        self.kernel_list = []

        while gate != 1:
            for i in range(n_conv):
                if value == "n":
                    k_size = input(
                        "Enter the kernel size for convolutional layer {}\nFor Example: 3,3: ".format(i + 1)).replace(' ','')
                    k_split = k_size.split(",")

                    if k_split[0].isnumeric() and int(k_split[0]) > 0 and k_split[1].isnumeric() and int(k_split[0]) > 0:
                        self.kernel_list.append(
                            (int(k_split[0]), int(k_split[1])))
                        if i == n_conv - 1:
                            gate = 1
                    else:
                        gate = 0
                        print("Please enter valid numeric values.")
                        print("The value must be an integer greater than zero.")
                        self.kernel_list = []
                        break
                else:
                    self.kernel_list.append((3, 3))
                    if i == n_conv - 1:
                        gate = 1
        return self.kernel_list

    def get_stride_padding(self, n_conv):
        self.padding = []
        self.stride = []
        gate = 0

        while gate != 1:
            value = \
                input("Use default values for padding and stride (y/n)? ").lower().replace(' ','')
            if value in {"y", "n"}:
                gate = 1
            else:
                print("Please enter valid input.")

        gate = 0
        while gate != 1:
            for i in range(n_conv):
                if value == "n":
                    pad_size = input(
                        "Enter padding for the image for convolutional layer {}\nFor Example 2,2: ".format(i + 1)).replace(' ','')
                    pad_split = pad_size.split(",")

                    if pad_split[0].isnumeric() and int(pad_split[0]) >= 0 and pad_split[1].isnumeric() and int(pad_split[0]) >= 0:
                        self.padding.append(
                            (int(pad_split[0]), int(pad_split[1])))
                        if i == n_conv - 1:
                            gate = 1
                    else:
                        gate = 0
                        print("Please enter valid numeric values.")
                        print(
                            "Values must be integers greater than or equal to zero.")
                        self.padding = []
                        break
                else:
                    self.padding.append((0, 0))
                    if i == n_conv - 1:
                        gate = 1

        gate = 0
        while gate != 1:
            for i in range(n_conv):
                if value == "n":
                    stride_size = input(
                        "Enter stride for the convolutions for convolutional layer {}\nFor Example 2,2: ".format(i + 1)).replace(' ','')
                    stride_split = stride_size.split(",")

                    if stride_split[0].isnumeric() and int(stride_split[0]) >= 0 and stride_split[1].isnumeric() and int(stride_split[0]) >= 0:
                        self.stride.append(
                            (int(stride_split[0]), int(stride_split[1])))
                        if i == n_conv - 1:
                            gate = 1
                    else:
                        gate = 0
                        print("Please enter valid numeric values.")
                        print("Values must be integers greater than zero.")
                        self.stride = []
                        break
                else:
                    self.stride.append((1, 1))
                    if i == n_conv - 1:
                        gate = 1

        return self.padding, self.stride

    def get_batch_norm(self, n_conv):
        gate = 0

        while gate != 1:
            value = \
                input("Use default values for batch normalization (y/n)? ").lower().replace(' ','')
            if value in {"y", "n"}:
                gate = 1
            else:
                print("Please enter valid input.")

        self.batch_norm = []

        gate = 0
        while gate != 1:
            for i in range(n_conv):
                if value == "n":
                    batch_boolean = input(
                        "Please enter 0 (no) or 1 (yes) for using batch normalization in convolutional layer {}: ".format(i+1)).replace(' ','')

                    if batch_boolean.isnumeric() and int(batch_boolean) in {0, 1}:
                        self.batch_norm.append(int(batch_boolean))
                        gate = 1
                    else:
                        gate = 0
                        print("Please enter valid numeric values.")
                        self.batch_norm = []
                        break
                else:
                    self.batch_norm.append(1)
                    if i == n_conv - 1:
                        gate = 1

        return self.batch_norm

    def get_dropout(self):
        gate = 0
        while gate != 1:
            value = input("Use default values for dropout (y/n)? ").replace(' ','')
            if value in {"y", "n"}:
                gate = 1
            else:
                print("Please enter valid input.")

        gate = 0
        if value == "n":
            while gate != 1:
                drop_out = input("Please input the dropout probability: ").replace(' ','')
                if drop_out.replace(".", "").isdigit() and float(drop_out) >= 0 and float(drop_out) < 1:
                    self.drop = drop_out
                    gate = 1
                else:
                    print("Please enter the valid numeric values.")
                    print("The value should lie between 0 and 1 [0, 1).")
        else:
            self.drop = 0

        return float(self.drop)

    def get_pooling(self, n_conv):
        gate = 0
        while gate != 1:
            value = value = input("Use default pooling values (y/n)? ").lower().replace(' ','')
            if value in {"y", "n"}:
                gate = 1
            else:
                print("Please enter valid input.")

        gate = 0
        self.pool_bool = []
        self.pool_size = []
        self.pool_stride = []
        while gate != 1:
            for i in range(n_conv):
                if value == "n":
                    pool_boolean = input(
                        "Please enter 0 (no) or 1 (yes) for using pooling in convolutional layer {}: ".format(i + 1)).replace(' ','')
                    if pool_boolean.isnumeric() and int(pool_boolean) in {0, 1}:
                        self.pool_bool.append(int(pool_boolean))
                        if i == n_conv - 1:
                            gate = 1
                    else:
                        gate = 0
                        print("Please enter valid numeric values.")
                        self.pool_bool = []
                        break
                else:
                    if i <= n_conv - 2:
                        self.pool_bool.append(0)
                    else:
                        self.pool_bool.append(1)
                        if i == n_conv - 1:
                            gate = 1

        gate = 0
        while gate != 1:
            for i in range(len(self.pool_bool)):
                if value == "n":
                    if self.pool_bool[i] == 0:
                        self.pool_size.append((0, 0))
                        gate = 1
                    else:
                        pooling_size = input(
                            "Please enter pool size for convolutional layer {}\nFor example 2,2: ".format(i + 1)).replace(' ','')
                        pooling_size_split = pooling_size.split(",")
                        if pooling_size_split[0].isnumeric() and int(pooling_size_split[0]) > 0 and pooling_size_split[1].isnumeric() and int(pooling_size_split[1]) > 0:
                            self.pool_size.append(
                                (int(pooling_size_split[0]), int(pooling_size_split[1])))
                            if i == len(self.pool_bool) - 1:
                                gate = 1
                        else:
                            gate = 0
                            print("please enter valid numeric values.")
                            print("Values must be integers greater than zero.")
                            self.pool_size = []
                            break
                else:
                    self.pool_size.append((2, 2))
                    if i == len(self.pool_bool) - 1:
                        gate = 1

        gate = 0
        while gate != 1:
            for i in range(len(self.pool_bool)):
                if value == "n":
                    if self.pool_bool[i] == 0:
                        self.pool_stride.append((0, 0))
                        gate = 1
                    else:
                        pooling_stride = input(
                            "Please enter pool stride for convolutional layer {}\nFor example 2,2: ".format(i + 1)).replace(' ','')
                        pooling_stride_split = pooling_stride.split(",")
                        if pooling_stride_split[0].isnumeric() and int(pooling_stride_split[0]) > 0 and pooling_stride_split[1].isnumeric() and int(pooling_stride_split[1]) > 0:
                            self.pool_stride.append(
                                (int(pooling_stride_split[0]), int(pooling_stride_split[1])))
                            if i == len(self.pool_bool) - 1:
                                gate = 1
                        else:
                            gate = 0
                            print("Please enter valid numeric values.")
                            print("Values must be integers greater than zero.")
                            self.pool_stride = []
                            break
                else:
                    self.pool_stride.append((2, 2))
                    if i == len(self.pool_bool) - 1:
                        gate = 1

        return self.pool_bool, self.pool_size, self.pool_stride

    def make_CNN(self, block, n_conv, pool_list, drop, batch_norm_list):
        layers = []
        for i in range(n_conv):
            if pool_list[i] == 0:
                if i < n_conv - 1:
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
                if i < n_conv - 1:
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

        print("Shapes after the convolutions:", shape)
        print("=" * 25)

        self.linear_size = self.channels[-1] * \
            shape[-1][0].item() * shape[-1][1].item()
        self.ConvNet = nn.Sequential(*layers)

    def forward(self, x):
        x = self.ConvNet(x)
        x = x.view(x.shape[0], -1)
        return x


class LSTM(nn.Module):
    """LSTM builds a LSTM network."""

    def __init__(self, input_size, if_default):
        super(LSTM, self).__init__()
        self.cnn_flatten = input_size
        self.default_gate = if_default
        self._get_input_size()

        print("\nLSTM Network\n")
        print("=" * 25)
        print("9/18 - LSTM hidden size")
        self._get_hidden_size()

        print("=" * 25)
        print("10/18 - LSTM number of layers")
        self._get_num_layers()

        print("=" * 25)
        print("11/18 - LSTM bidirectional")
        self. _get_bidirectional()

        print("=" * 25)
        print("12/18 - LSTM output size")
        self. _get_output_size()

        self._build_network_architecture()

    def _get_input_size(self):
        self.input_size = self.cnn_flatten

    def _get_hidden_size(self):
        gate = 0
        while gate != 1:
            if self.default_gate == True:
                print("Default value for hidden size selected: 128")
                self.hidden_size = "128"
            else:
                print(
                    "Please enter an integer greater than 0 for the network's hidden size.")
                self.hidden_size = input(
                    "For default size, press enter without any input: ").replace(' ','')

            if self.hidden_size == "":
                print("Default value for hidden size selected: 128")
                self.hidden_size = "128"

            if self.hidden_size.isnumeric() and int(self.hidden_size) > 0:
                self.hidden_size = int(self.hidden_size)
                gate = 1
            else:
                print("Please enter a valid input.\n")

        print()

    def _get_num_layers(self):
        gate = 0
        while gate != 1:
            if self.default_gate == True:
                print("Default value selected for number of layers: 3")
                self.nlayers = "3"
            else:
                print(
                    "Please enter a positive integer for the network's number of layers.")
                self.nlayers = input(
                    "For default option, press enter without any input: ").replace(' ','')

            if self.nlayers == "":
                print("Default value selected for number of layers: 3")
                self.nlayers = "3"

            if self.nlayers.isnumeric() and int(self.nlayers) > 0:
                self.nlayers = int(self.nlayers)
                gate = 1
            else:
                print("Please enter a valid input.\n")

        print()

    def _get_bidirectional(self):
        gate = 0
        while gate != 1:
            if self.default_gate == True:
                print("By default, unidirectional LSTM network selected.")
                self.bidirection = "0"
            else:
                print("Please enter 1 to create a bidirectional LSTM network.")
                print("Else, enter 0.")
                self.bidirection = input(
                    "For default option, press enter without any input: ").replace(' ','')

            if self.bidirection == "":
                print("By default, unidirectional LSTM network selected")
                self.bidirection = "0"
            if self.bidirection.isnumeric() and int(self.bidirection) in {0, 1}:
                if self.bidirection == "1":
                    self.bidirection_input = True
                else:
                    self.bidirection_input = False
                gate = 1
            else:
                print("Please enter a valid input.\n")

        print()

    def _get_output_size(self):
        gate = 0
        while gate != 1:
            print("Please enter the output size for the network.")
            self.output_size = input(
                "Enter 1 for regression or the number of classes for classification: ").replace(' ','')

            if self.output_size.isnumeric() and int(self.output_size) > 0:
                self.output_size = int(self.output_size)
                gate = 1
            else:
                print("Please enter a valid input.\n")

        print()

    def _build_network_architecture(self):
        self.lstm = nn.LSTM(self.input_size, self.hidden_size,
                            bidirectional=self.bidirection_input,
                            num_layers=self.nlayers, batch_first=True)

        if self.bidirection_input:
            self.linear_input = self.hidden_size * 2
        else:
            self.linear_input = self.hidden_size

        self.linear1 = nn.Linear(self.linear_input, int(self.linear_input / 2))
        self.linear2 = nn.Linear(
            int(self.linear_input / 2), int(self.linear_input / 4))
        self.linear3 = nn.Linear(int(self.linear_input / 4), self.output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear1(out[:, -1, :])
        out = self.linear2(out)
        out = self.linear3(out)
        return out


class CNN_LSTM(nn.Module):
    """CNN_LSTM combines the CNN and LSTM networks."""

    def __init__(self, CNNNetwork, LSTMNetwork, device):
        super(CNN_LSTM, self).__init__()
        self.cnn = CNNNetwork
        self.lstm = LSTMNetwork
        self.device = device

    def forward(self, x):
        b, time, num_channels, h, w = x.shape
        lstm_input = torch.zeros(
            (b, time, self.lstm.input_size)).double().to(self.device)

        for i in range(time):
            lstm_input[:, i, :] = self.cnn.forward(x[:, i, :, :, :])

        output = self.lstm(lstm_input)
        return output

    def predict(self, x):
        b, time, num_channels, h, w = x.shape
        lstm_input = torch.zeros(
            (b, time, self.lstm.input_size)).double().to(self.device)

        for i in range(time):
            lstm_input[:, i, :] = self.cnn.forward(x[:, i, :, :, :])

        output = self.lstm(lstm_input)
        return output


class CNNLSTM():
    """CNNLSTM will be called by the user. This class calls all
    other necessary classes to build a complete pipeline for training
    the model.
    View the documentation at https://manufacturingnet.readthedocs.io/
    """

    def __init__(self, X, Y, shuffle=True):
        self.criterion_list = {1: nn.CrossEntropyLoss(), 2: torch.nn.L1Loss(
        ), 3: torch.nn.SmoothL1Loss(), 4: torch.nn.MSELoss()}
        self.x_data = X
        self.y_data = Y
        self.shuffle = shuffle
        self.get_default_parameters()

        self.cnn_network = CNN2D(CNNBlock).double()
        self.lstm_network = LSTM(
            self.cnn_network.linear_size, self.default_gate).double()
        self._set_device()
        self.net = CNN_LSTM(self.cnn_network, self.lstm_network, self.device)

        print("=" * 25)
        print("13/18 - Batch size input")
        self._get_batchsize_input()

        print("=" * 25)
        print("14/18 - Validation set size")
        self._get_valsize_input()

        self.model_data = ModelDataset(
            self.x_data, self.y_data, batchsize=self.batchsize, valset_size=self.valset_size, shuffle=self.shuffle)

        print("=" * 25)
        print("15/18 - Loss function")
        self._get_loss_function()

        print("=" * 25)
        print("16/18 - Optimizer")
        self._get_optimizer()

        print("=" * 25)
        print("17/18 - Scheduler")
        self._get_scheduler()

        print("=" * 25)
        print("18/18 - Number of epochs")
        self._get_epoch()

        self.main()

    def get_default_parameters(self):
        gate = 0
        while gate != 1:
            self.default = \
                input(
                    "Use default values for all the training parameters (y/n)? ").lower().replace(' ','')
            if self.default in {"y", "n"}:
                if self.default == "y":
                    self.default_gate = True
                else:
                    self.default_gate = False
                gate = 1
            else:
                print("Please enter a valid input.\n")

        print()

    def _get_batchsize_input(self):
        gate = 0
        while gate != 1:
            self.batchsize = input(
                "Please enter a batch size greater than 0: ").replace(' ','')
            if self.batchsize.isnumeric() and int(self.batchsize) > 0:
                self.batchsize = int(self.batchsize)
                gate = 1
            else:
                print("Please enter a valid input.\n")

        print()

    def _get_valsize_input(self):
        gate = 0
        while gate != 1:
            if self.default_gate == True:
                print("Default value selected : 0.2")
                self.valset_size = "0.2"
            else:
                print("Please enter the training set size as a float (0,1).")
                self.valset_size = input(
                    "For default size, press enter without any input: ").replace(' ','')

            if self.valset_size == "":
                print("Default value selected : 0.2")
                self.valset_size = "0.2"

            if self.valset_size.replace(".", "").isdigit():
                if float(self.valset_size) > 0 and float(self.valset_size) < 1:
                    self.valset_size = float(self.valset_size)
                    gate = 1
            else:
                print("Please enter a valid input.\n")

        print()

    def _get_loss_function(self):
        gate = 0
        while gate != 1:
            print("Please choose a loss function.")
            self.criterion_input = input(
                "[1: CrossEntropyLoss, 2: L1Loss, 3: SmoothL1Loss, 4: MSELoss]: ").replace(' ','')

            if self.criterion_input.isnumeric() and int(self.criterion_input) < 5 and int(self.criterion_input) > 0:
                gate = 1
            else:
                print("Please enter a valid input.\n")

        self.criterion = self.criterion_list[int(self.criterion_input)]
        print()

    def _get_optimizer(self):
        gate = 0
        while gate != 1:
            if self.default_gate == True:
                print("Default optimizer selected: Adam")
                self.optimizer_input = "1"
            else:
                print(
                    "Please choose an optimizer from the list: [1: Adam, 2: SGD]")
                self.optimizer_input = input(
                    "For default optimizer, press enter without any input: ").replace(' ','')

            if self.optimizer_input == "":
                print("Default optimizer selected: Adam")
                self.optimizer_input = "1"

            if self.optimizer_input.isnumeric() and int(self.optimizer_input) > 0 and int(self.optimizer_input) < 3:
                gate = 1
            else:
                print("Please enter a valid input.\n")

        print()

        gate = 0
        while gate != 1:
            if self.default_gate == True:
                print("Default learning rate selected: 0.001")
                self.user_lr = "0.001"
            else:
                print("Please enter a positive learning rate.")
                self.user_lr = input(
                    "For default learning rate, press enter without any input: ").replace(' ','')

            if self.user_lr == "":
                print("Default value for learning rate selected : 0.001")
                self.user_lr = "0.001"

            if self.user_lr.replace(".", "").isdigit():
                if float(self.user_lr) > 0:
                    self.lr = float(self.user_lr)
                    gate = 1
            else:
                print("Please enter a valid input.\n")

        self.optimizer_list = {1: optim.Adam(self.net.parameters(
        ), lr=self.lr), 2: optim.SGD(self.net.parameters(), lr=self.lr)}
        self.optimizer = self.optimizer_list[int(self.optimizer_input)]
        print()

    def _get_scheduler(self):
        gate = 0
        while gate != 1:
            if self.default_gate == True:
                print("By default, no scheduler selected.")
                self.scheduler_input = "1"
            else:
                print(
                    "Please choose a scheduler from the list: [1: None, 2:StepLR, 3:MultiStepLR]")
                self.scheduler_input = input(
                    "For default option of no scheduler, press enter without any input: ").replace(' ','')

            if self.scheduler_input == "":
                print("By default no scheduler selected")
                self.scheduler_input = "1"

            if self.scheduler_input.isnumeric() and int(self.scheduler_input) > 0 and int(self.scheduler_input) < 4:
                gate = 1
            else:
                print("Please enter a valid input.\n")

        print()

        if self.scheduler_input == "1":
            self.scheduler = None
        elif self.scheduler_input == "2":
            gate = 0
            while gate != 1:
                self.step = (
                    input("Please enter a step value int input (step > 0): ")).replace(' ','')
                if self.step.isnumeric() and int(self.step) > 0:
                    self.step = int(self.step)
                    gate = 1
                else:
                    print("Please enter a valid input.\n")

            print()

            gate = 0
            while gate != 1:
                self.gamma = input(
                    "Please enter a positive float for the multiplying factor: ").replace(' ','')
                if self.gamma.replace(".", "").isdigit():
                    if float(self.gamma) > 0:
                        self.gamma = float(self.gamma)
                        gate = 1
                else:
                    print("Please enter a valid input.\n")

            self.scheduler = scheduler.StepLR(
                self.optimizer, step_size=self.step, gamma=self.gamma)
        elif self.scheduler_input == "3":
            gate = 0
            while gate != 1:
                self.milestones_input = input("Please enter integers for milestone epochs (Example: 2, 6, 10): ").replace(' ','')

                self.milestones_input = self.milestones_input.split(",")

                for i in range(len(self.milestones_input)):
                    if self.milestones_input[i].isnumeric() and int(self.milestones_input[i]) > 0:
                        gate = 1
                    else:
                        gate = 0
                        break
                if gate == 0:
                    print("Please enter a valid input.\n")

            self.milestones = [int(x)
                               for x in self.milestones_input if int(x) > 0]
            print()

            gate = 0
            while gate != 1:
                self.gamma = input(
                    "Please enter a positive float for the multiplying factor: ").replace(' ','')
                if self.gamma.replace(".", "").isdigit():
                    if float(self.gamma) > 0:
                        self.gamma = float(self.gamma)
                        gate = 1
                else:
                    print("Please enter a valid input.\n")

            self.scheduler = scheduler.MultiStepLR(
                self.optimizer, milestones=self.milestones, gamma=self.gamma)

    def _set_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def _get_epoch(self):
        gate = 0
        while gate != 1:
            self.numEpochs = input(
                "Please enter a positive integer for the number of epochs: ").replace(' ','')
            if self.numEpochs.isnumeric() and int(self.numEpochs) > 0:
                self.numEpochs = int(self.numEpochs)
                gate = 1
            else:
                print("Please enter a valid input.\n")

        print()

    def main(self):
        self.net.to(self.device)
        print("=" * 25)

        print("Network architecture:\n")
        print(self.net)
        print("=" * 25)

        self.get_model_summary()
        print()
        print("=" * 25)

        # getting inputs and labels for training set
        xt, yt = self.model_data.get_trainset()

        # getting inputs and labels for validation set
        xv, yv = self.model_data.get_valset()

        # creating the training dataset
        self.train_dataset = Dataset(xt, yt)

        # creating the validation dataset
        self.val_dataset = Dataset(xv, yv)

        # creating the training dataset dataloader
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.model_data.get_batchsize(
        ), shuffle=True)

        # creating the validation dataset dataloader
        self.dev_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.model_data.get_batchsize())

        self.train_model()
        self.get_loss_graph()

        if self.criterion_input == "1":
            self.get_accuracy_graph()
            self.get_confusion_matrix()
        else:
            self.get_r2_score()

        self._save_model()

        print("Call get_prediction() to make predictions on new data.\n")
        print("=== End of training ===")

    def _save_model(self):
        gate = 0
        while gate != 1:
            save_model = input("Save the model weights (y/n)? ").lower().replace(' ','')
            if save_model.lower() == "y":
                path = "model_parameters.pth"
                torch.save(self.net.state_dict(), path)
                gate = 1
            elif save_model.lower() == "n":
                gate = 1
            else:
                print("Please enter a valid input.")

        print("=" * 25)

    def get_model_summary(self):
        print("Model Summary:\n")
        print("Bidirectional:", self.lstm_network.bidirection_input)
        print("Number of layer:", self.lstm_network.nlayers)
        print("Criterion:", self.criterion)
        print("Optimizer:", self.optimizer)
        print("Scheduler:", self.scheduler)
        print("Validation set size:", self.valset_size)
        print("Batch size:", self.batchsize)
        print("Initial learning rate:", self.lr)
        print("Number of training epochs:", self.numEpochs)
        print("Device:", self.device)

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
            print("Epoch_Number:", epoch)
            running_loss = 0.0

            for batch_idx, (data, target) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                data = data.to(self.device)
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
            print("Training Loss:", running_loss)

            # printing the epoch accuracy only if the loss function is Cross entropy
            if self.criterion_input == "1":
                acc = (correct_predictions/total_predictions)*100.0
                self.training_acc.append(acc)
                print("Training Accuracy:", acc, "%")

            dev_loss, dev_acc = self.validate_model()

            if self.scheduler_input != "1":
                self.scheduler.step()
                print("Current scheduler status:", self.optimizer)

            end_time = time.time()
            print("Epoch Time:", end_time - start_time, "s")
            print("#"*50)

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

        # calculating and printing the epoch accuracy only if the loss function is Cross entropy
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
        result = confusion_matrix(np.concatenate(
            np.array(self.predict)), np.concatenate(np.array(self.actual)))
        print("Confusion Matrix:")
        print(result)

    def get_r2_score(self):
        result = r2_score(np.concatenate(np.array(self.predict)),
                          np.concatenate(np.array(self.actual)))
        print("r2 score: ")
        print(result)

        plt.figure(figsize=(8, 8))
        plt.scatter(np.concatenate(np.array(self.actual)), np.concatenate(
            np.array(self.predict)), label="r2 score", s=1)
        plt.legend()
        plt.title("Model r2 score: " + str(result))
        plt.xlabel("labels")
        plt.ylabel("predictions")
        plt.savefig("r2_score.png")

    def get_prediction(self, x_input):
        """Pass in an input numpy array for making prediction.
        For passing multiple inputs, make sure to keep number of
        examples to be the first dimension of the input.
        For example, 10 data points need to be checked and each point
        has (3, 50, 50) resized or original input size, the shape of the
        array must be (10, 3, 50, 50).
        View the documentation at https://manufacturingnet.readthedocs.io/
        """
        if len(x_input.shape) == 3:
            x_input = (x_input).reshape(
                1, x_input.shape[0], x_input.shape[1], x_input.shape[2])

        x_input = torch.from_numpy(x_input).to(self.device)
        net_output = self.net.predict(x_input)

        if self.criterion_input == "1":
            _, net_output = torch.max(net_output.data, 1)

        return net_output
