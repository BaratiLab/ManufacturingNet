import time
from tqdm import tqdm
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
import math


class TransformerBase(nn.Module):
    def __init__(self, 
                 is_default,
                 seq_len):
        super(TransformerBase, self).__init__()
        self.default_gate = is_default
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(' ')
        print('1/11 - Transformer input size')
        self._get_input_size()
        self._get_num_heads()
        # self.input_size *= 2
        self.pos_embedding = self.position_encoding(seq_len=seq_len)
        print('='*25)
        print('2/11 - Transformer hidden size')
        self._get_hidden_size()

        print('='*25)
        print('3/11 - Transformer number of layers')
        self._get_num_layers()

        print('='*25)
        print('4/11 - Transformer output size')
        self. _get_output_size()

        self._build_network_architecture()
        # self.is_attention = is_attention
    

    def _get_input_size(self):

        # Method for getting an input size parameter for the Transformer network

        gate = 0
        while gate != 1:
            self.input_size = (input(
                'Please enter the input size in int for the Transformer network (input size > 0): ')).replace(' ','')
            if self.input_size.isnumeric() and int(self.input_size) > 0:
                self.input_size = int(self.input_size)
                gate = 1
            else:
                print('Please enter a valid input')
                print(' ')
        print(' ')


    def _get_num_heads(self):

        # Method for getting an input size parameter for the Transformer network

        gate = 0
        while gate != 1:
            self.num_heads = (input(
                'Please enter the number of heads in int for the Transformer network (input size > 0): ')).replace(' ','')
            if self.num_heads.isnumeric() and int(self.num_heads) > 0:
                self.num_heads = int(self.num_heads)
                gate = 1
            else:
                print('Please enter a valid input')
                print(' ')
        print(' ')

    def _get_hidden_size(self):

        # Method for getting an hidden size parameter for the Transformer network

        gate = 0
        while gate != 1:
            if self.default_gate == True:
                print('Default value for hidden size selected: 128')
                self.hidden_size = '128'
            else:
                self.hidden_size = (input(
                    'Please enter the hidden size in int for the Transformer network (hidden size > 0)\n For default size, please directly press enter without any input: ')).replace(' ','')
            if self.hidden_size == '':              # handling default case for hidden size
                print('Default value for hidden size selected: 128')
                self.hidden_size = '128'
            if self.hidden_size.isnumeric() and int(self.hidden_size) > 0:
                self.hidden_size = int(self.hidden_size)
                gate = 1
            else:
                print('Please enter a valid input')
                print(' ')
        print(' ')

    def _get_num_layers(self):

        # Method for getting a number of Transformer layer parameter for the Transformer network

        gate = 0
        while gate != 1:
            if self.default_gate == True:
                print('Default value selected for number of layers: 3')
                self.nlayers = '3'
            else:
                self.nlayers = (input(
                    'Please enter the number of layer for the Transformer network in int (number of layer > 0)\n For default option, please directly press enter without any input: ')).replace(' ','')
            if self.nlayers == '':              # handling default case for number of Transformer layer
                print('Default value selected for number of layers: 3')
                self.nlayers = '3'
            if self.nlayers.isnumeric() and int(self.nlayers) > 0:
                self.nlayers = int(self.nlayers)
                gate = 1
            else:
                print('Please enter a valid input')
                print(' ')
        print(' ')

    def _get_output_size(self):

        # Method for getting output size of the network
        gate = 0
        while gate != 1:
            self.output_size = (input(
                'Please enter the output size for the Transformer network. \n For regression please enter 1 else enter the number of classes for classification problem: ')).replace(' ','')
            if self.output_size.isnumeric() and int(self.output_size) > 0:
                self.output_size = int(self.output_size)
                gate = 1
            else:
                print('Please enter a valid input')
                print(' ')
        print(' ')

    def _build_network_architecture(self):
        # Method for building a network using all the information provided by a user in above functions

        self.layer_instance = nn.TransformerEncoderLayer(self.input_size*self.num_heads,
                                                         nhead=self.num_heads,
                                                         dim_feedforward=512)
        self.feature_extractor = nn.TransformerEncoder(self.layer_instance,
                                                       num_layers=self.nlayers)
        self.multihead_linear = nn.Linear(self.input_size*self.num_heads, self.input_size)
        self.ffn = nn.Linear(self.input_size, self.hidden_size)

        self.linear_input = self.hidden_size
        self.linear1 = nn.Linear(self.linear_input, 
                                 int(self.linear_input/2))
        self.linear2 = nn.Linear(int(self.linear_input/2), 
                                 int(self.linear_input/4))
        self.linear3 = nn.Linear(int(self.linear_input/4), 
                                 self.output_size)

    def forward(self, x):
        # print(x.shape, self.pos_embedding.shape)
        x = x + self.pos_embedding
        x = x.repeat(1, 1, self.num_heads)
        out = self.feature_extractor(x)
        out = self.multihead_linear(out)
        # out = out.reshape(out.shape[0], out.shape[1], -1, )
        out = self.ffn(out)
        # out = out[:, -1, :]
        out = self.linear1(out[:, -1, :])
        out = self.linear2(out)
        out = self.linear3(out)
        return out
    
    def position_encoding(self, seq_len):
        # Method for adding positional encoding to the input data
        # x: input data
        # return: positional encoding added to the input data
        d = self.input_size
        p = seq_len
        embedding = torch.zeros(p, 0)
        mid_point = math.ceil(d / 2)
        for dim in range(mid_point):
            pos = torch.arange(p)
            cos_component = torch.cos(pos.reshape(-1, 1) / math.pow(1000, 2*dim))
            sin_component = torch.sin(pos.reshape(-1, 1) / math.pow(1000, 2*dim))
            embedding = torch.cat((embedding,
                                   cos_component.reshape(-1, 1),
                                   sin_component.reshape(-1, 1)), dim=1)
        if embedding.shape[1] != d:
            embedding = embedding[:, :-1]
        embedding = torch.unsqueeze(embedding, dim=0)
        embedding = embedding.to(self.device)
        return embedding

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


# The following class is used for creating a dataset class using torch functionality. Its a standard pytorch class
class Dataset(data.Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        x_item = torch.from_numpy(self.X[index]).float()
        y_item = torch.from_numpy(np.array(self.Y[index])).float()
        return x_item, y_item


class Transformer():
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
        self.seq_len = X.shape[1]
        self.shuffle = shuffle

        self.get_default_paramters()            # getting default parameters argument

        # building a network architecture
        self.net = (TransformerBase(is_default=self.default_gate, seq_len=self.seq_len)).float()

        print('='*25)
        print('6/11 - Batch size input')
        # getting a batch size for training and validation
        self._get_batchsize_input()

        print('='*25)
        print('7/11 - Validation set size')
        self._get_valsize_input()                # getting a train-validation split

        # splitting the data into training and validation sets
        self.model_data = ModelDataset(
            self.x_data, self.y_data, batchsize=self.batchsize, valset_size=self.valset_size, shuffle=self.shuffle)

        print('='*25)
        print('8/11 - Loss function')
        self._get_loss_function()               # getting a loss function

        print('='*25)
        print('9/11 - Optimizer')
        self._get_optimizer()               # getting an optimizer input

        print('='*25)
        print('10/11 - Scheduler')
        self._get_scheduler()               # getting a scheduler input

        self._set_device()              # setting the device to gpu or cpu

        print('='*25)
        print('11/11 - Number of epochs')
        self._get_epoch()           # getting an input for number oftraining epochs

        self.main()             # run function

    def get_default_paramters(self):

        # Method for getting a binary input for default paramters

        gate = 0
        while gate != 1:
            self.default = input(
                'Do you want default values for all the parameters (y/n)? ').replace(' ','')
            if self.default == 'y' or self.default == 'Y' or self.default == 'n' or self.default == 'N':
                if self.default.lower() == 'y':
                    self.default_gate = True
                else:
                    self.default_gate = False
                gate = 1
            else:
                print('Enter a valid input')
                print(' ')
        print(' ')

    def _get_batchsize_input(self):

        # Method for getting batch size input

        gate = 0
        while gate != 1:
            self.batchsize = (
                input('Please enter the batch size int input (greater than 0): ')).replace(' ','')
            if self.batchsize.isnumeric() and int(self.batchsize) > 0:
                self.batchsize = int(self.batchsize)
                gate = 1
            else:
                print('Please enter a valid input')
        print(' ')

    def _get_valsize_input(self):

        # Method for getting validation set size input

        gate = 0
        while gate != 1:
            if self.default_gate == True:
                print('Default value selected : 0.2')
                self.valset_size = '0.2'
            else:
                self.valset_size = (input(
                    'Please enter the Val set size float input (size > 0 and size < 1) \n For default size, please directly press enter without any input: ')).replace(' ','')
            if self.valset_size == '':              # handling default case for valsize
                print('Default value selected : 0.2')
                self.valset_size = '0.2'
            if self.valset_size.replace('.', '').isdigit():
                if float(self.valset_size) > 0 and float(self.valset_size) < 1:
                    self.valset_size = float(self.valset_size)
                    gate = 1
            else:
                print('Please enter a valid input')
                print(' ')
        print(' ')

    def _get_loss_function(self):

        # Method for getting a loss function for training

        gate = 0
        while gate != 1:
            self.criterion_input = (input(
                'Please enter the appropriate loss function index for the problem: \n Criterion_list - [1: CrossEntropyLoss, 2: L1Loss, 3: SmoothL1Loss, 4: MSELoss]: ')).replace(' ','')

            if self.criterion_input.isnumeric() and int(self.criterion_input) < 5 and int(self.criterion_input) > 0:
                gate = 1
            else:
                print('Please enter a valid input')
                print(' ')

        self.criterion = self.criterion_list[int(self.criterion_input)]
        print(' ')

    def _get_optimizer(self):

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

        self.optimizer_list = {1: optim.Adam(self.net.parameters(
        ), lr=self.lr), 2: optim.SGD(self.net.parameters(), lr=self.lr)}
        self.optimizer = self.optimizer_list[int(self.optimizer_input)]
        print(' ')

    def _get_scheduler(self):

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
            self.numEpochs = (input(
                'Please enter the number of epochs int input to train the model (number of epochs > 0): ')).replace(' ','')
            if self.numEpochs.isnumeric() and int(self.numEpochs) > 0:
                self.numEpochs = int(self.numEpochs)
                gate = 1
            else:
                print('Please enter a valid input')
                print(' ')
        print(' ')

    def main(self):

        # Method integrating all the functions and training the model

        self.net.to(self.device)
        print('='*25)

        print('Network architecture: ')
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
                                                        shuffle=True)           # creating the training dataset dataloader

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
                'Do you want to save the model weights? (y/n): ').replace(' ','')
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
        # print('Bidirectional: ', self.net.bidirection_input)
        print('Number of layer: ', self.net.nlayers)
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

        for epoch in range(self.numEpochs):

            start_time = time.time()
            self.net.train()
            print('Epoch_Number: ', epoch)
            running_loss = 0.0
            loop = tqdm(self.train_loader, total=len(self.train_loader))
            for batch_idx, (data, target) in enumerate(loop):

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
                    # print(outputs.shape, target.shape)
                    # if len(outputs.shape) != 1:
                    outputs = outputs.reshape(-1)
                    loss = self.criterion(outputs, target)
                running_loss += loss.item()
                loop.set_postfix({"loss": running_loss / (batch_idx + 1)})
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
        loop = tqdm(self.dev_loader, total=len(self.dev_loader))
        for batch_idx, (data, target) in enumerate(loop):

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
                outputs = outputs.reshape(-1)
                loss = self.criterion(outputs, target)
                # if len(outputs.shape) != 1:
                self.predict.append(outputs.detach().cpu().numpy())
            running_loss += loss.item()
            loop.set_postfix({"loss": running_loss / (batch_idx + 1)})
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
        np_predict = np.zeros((0,))
        np_actual = np.zeros((0,))
        for i in range(len(self.predict)):
            np_predict = np.concatenate((np_predict, 
                                         np.asarray(self.predict[i]).reshape(-1)),
                                         axis=0)
            np_actual = np.concatenate((np_actual,
                                        self.actual[i].reshape(-1)),
                                        axis=0)
        result = r2_score(np_predict, np_actual)
        print(result)

        plt.figure(figsize=(8, 8))
        plt.scatter(np_actual, np_predict, label='r2 score', s=1)
        plt.legend()
        plt.title('Model r2 score: ' + str(result))
        plt.xlabel('labels')
        plt.ylabel('predictions')
        plt.savefig('r2_score.png')

    def get_prediction(self, x_input):
        """

        Pass in an input numpy array for making prediction.
        For passing multiple inputs, make sure to keep number of examples to be the first dimension of the input.
        For example, 5 data points need to be checked and each point has 14 input size with time sequence dimension of 4, the shape of the array must be (5,4,14).
        For more information, please see documentation.

        """

        # Method to use at the time of inference

        if len(x_input.shape) == 2:             # handling the case of single

            x_input = (x_input).reshape(1, x_input.shape[0], x_input.shape[1])

        x_input = torch.from_numpy(x_input).to(self.device)

        net_output = self.net.predict(x_input)

        if self.criterion_input == '1':             # handling the case of classification problem

            _, net_output = torch.max(net_output.data, 1)

        return net_output

    
if __name__ == '__main__':
    x = np.load('../../tutorials/Mercedes_files/merc_features.npy', allow_pickle = True)
    y = np.load('../../tutorials/Mercedes_files/merc_labels.npy', allow_pickle = True)
    print(x.shape, y.shape)
    # x = x[:50000]
    # y = y[:50000]
    # Transformer(x, y)
