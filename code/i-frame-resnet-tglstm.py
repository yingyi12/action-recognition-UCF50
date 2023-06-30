from time import time
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2 as cv
import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchvision.models as models
from torch.autograd import Variable 
from sklearn.metrics import classification_report, confusion_matrix
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
from torchmetrics import MetricCollection, Accuracy, F1Score
from torchmetrics.classification import MulticlassAccuracy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image
import os
from plotcm import plot_confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from torch.nn import Parameter
import torch.jit as jit
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
import math
from module import TGLSTM

def downsample(sequences, n):
    batch_size, seq_length, c, h, w = sequences.shape
    new_seq_length = seq_length // n
    downsampled = torch.zeros((batch_size, new_seq_length, c, h, w))
    for i in range(new_seq_length):
        downsampled[:, i] = sequences[:, i*n]
    return downsampled

def prepare_dataset(df):
    # Encode the labels
    encoder = LabelEncoder()
    df['Label'] = encoder.fit_transform([s.split("_")[1] for s in df['Label']])  
    
    X = []
    y = []
    for i in range(len(df)):
        sequence = df.iloc[i]['Tensor']
        label = df.iloc[i]['Label']
        X.append(sequence) 
        y.append(label)
    # y = np.array(y)
    # label_reshaped = y.reshape(-1, 1)
    # onehot_encoder = OneHotEncoder(sparse=False)
    # y = onehot_encoder.fit_transform(label_reshaped)
    
    return X, y


class Dataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __getitem__(self, index):
        sequence = self.sequences[index]
        label = self.labels[index]
        return sequence, label
    
    def __len__(self):
        return len(self.sequences)


def collate_fn(batch):
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    sequences =[torch.tensor(seq) for seq in sequences]
    labels = [torch.tensor(label) for label in labels]
    sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    num_sample = sequences.size(0)
    seq_length = sequences.size(1) // 150528
    final_shape = (num_sample, seq_length,  3, 224, 224)
    sequences =sequences.view(final_shape)
    sequences = downsample(sequences, 1) # downsample sequences
    new_seq_length = sequences.size(1)
    padded_time = torch.ones((num_sample,new_seq_length))
    # labels = torch.tensor(labels)
    labels = torch.stack(labels, dim=0)
    return sequences,  labels, padded_time
    
'''def prepare_dataset(df):
    # Encode the labels
    encoder = LabelEncoder()
    df['Label'] = encoder.fit_transform([s.split("_")[1] for s in df['Label']])  
    
    X = []
    y = []
    time = []
    for i in range(len(df)):
        sequence = df.iloc[i]['Tensor']
        label = df.iloc[i]['Label']
        Time = df.iloc[i]['Time']
        X.append(sequence) 
        y.append(label)
        time.append(Time)
    return X, y,time

class Dataset(Dataset):
    def __init__(self, sequences, labels,time):
        self.sequences = sequences
        self.labels = labels
        self.time = time
        
    def __getitem__(self, index):
        sequence = self.sequences[index]
        label = self.labels[index]
        t = self.time[index]
        return sequence, label,t

    def __len__(self):
        return len(self.sequences)
    
    
def collate_fn(batch):
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    time = [item[2] for item in batch]
    sequences =[torch.tensor(seq) for seq in sequences]
    labels = [torch.tensor(label) for label in labels]
    time = [torch.tensor(t) for t in time]
    # sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    seq_len_s = [len(seq) for seq in sequences]
    max_len = max(seq_len_s)
    padded_sequences = []
    for seq in sequences:
        seq_len = len(seq)
        padding_len = max_len - seq_len
        if padding_len > 0:
            subset = seq[-150528:]
            padding = subset.repeat(padding_len // 150528)
            padded_seq = torch.cat((seq, padding), dim=0)
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
        # print("Sequence length before padding:", seq_len//150528)
        # print((padding_len // 150528))
        # print("Sequence length after padding:", len(padded_seq)//150528)
    sequences = torch.nn.utils.rnn.pad_sequence(padded_sequences, batch_first=True)
    num_sample = sequences.size(0)
    seq_length = sequences.size(1) // 150528
    final_shape = (num_sample, seq_length,  3, 224, 224)
    sequences =sequences.view(final_shape)
    padded_time = torch.nn.utils.rnn.pad_sequence(time,batch_first=True,padding_value=0)
    # labels = torch.tensor(labels)
    labels = torch.stack(labels, dim=0)
    return sequences,  labels, padded_time'''

class DataModule(pl.LightningDataModule):

    
    def __init__(self,batch_size,num_workers=0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.X_test = None

    def prepare_data(self):
        pass

    def setup(self, stage):

        if stage == 'fit' and self.X_train is not None and self.X_val is not None:
            return 
        if stage == 'test' and self.X_test is not None:
            return
        if stage is None and self.X_train is not None and self.X_test is not None and self.X_val is not None:  
            return
        
        df_train = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/i-frame/train10-i-frame.pkl')
        df_val = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/i-frame/val10-i-frame.pkl')
        df_test = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/i-frame/test10-i-frame.pkl')
        
        # df_train = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/random/train10-random.pkl')
        # df_val = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/random/val10-random.pkl')
        # df_test = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video//random/test10-random.pkl')
        
        X_train, y_train = prepare_dataset(df_train)
        X_val, y_val = prepare_dataset(df_val)
        X_test, y_test = prepare_dataset(df_test)


        if stage == 'fit' or stage is None:
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val

        if stage == 'test' or stage is None:
            self.X_test = X_test
            self.y_test = y_test
        

    def train_dataloader(self):
        train_dataset = Dataset(self.X_train,self.y_train)
        train_loader = DataLoader(train_dataset, batch_size = self.batch_size, 
                              shuffle = True, drop_last = True,
                              num_workers = self.num_workers,collate_fn=collate_fn)
        return train_loader
    
    def val_dataloader(self):
        val_dataset = Dataset(self.X_val,self.y_val)
        val_loader = DataLoader(val_dataset, batch_size = self.batch_size, 
                            shuffle = False, drop_last = True,
                            num_workers = self.num_workers,collate_fn=collate_fn)

        return val_loader

    def test_dataloader(self):
        test_dataset = Dataset(self.X_test,self.y_test)
        test_loader = DataLoader(test_dataset,  batch_size = self.batch_size, 
                              shuffle = False, drop_last = True,
                              num_workers = self.num_workers,collate_fn=collate_fn)

        return test_loader
    
   
class TimeGatedLSTM(pl.LightningModule):

    def __init__(self, 
                 n_features, 
                 hidden_size, 
                 #seq_len, 
                 batch_size,
                 num_layers, 
                 dropout, 
                 learning_rate):
        super(TimeGatedLSTM, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        #self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, n_features))
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 10)
        self.tglstm = TGLSTM(input_size=n_features, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            bias=True, 
                            batch_first=False, 
                            dropout=False, 
                            bidirectional=False)
        for weight in self.tglstm.parameters():
             if len(weight.shape) >= 2:
                nn.init.kaiming_normal_(weight)
        self.linear = nn.Linear(hidden_size, 10)
        self.mean = None
        self.std = None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.loss = nn.CrossEntropyLoss()
        self.targets = torch.Tensor([])
        self.preds = torch.Tensor([])
        
    def forward(self, x, time):
        # hidden = None
        # for t in range(x.size(1)):
        #     with torch.no_grad():
        #         x = x[:, t, :, :, :]
        #         x = self.resnet(x)  
        #     out, hidden = self.lstm(x.unsqueeze(0), hidden)         
        
        batch_size, seq_len, _, _, _ = x.shape
        x = x.reshape(batch_size * seq_len, 3, 224, 224)

        # Pass input tensor through ResNet18 to obtain feature maps
        with torch.no_grad():
            x = self.resnet(x)
        # Reshape feature maps
        x = x.reshape(batch_size, seq_len, -1)
        # X.shape: [batch_size, seq_len, features]
	    # time.shape: [batch_size, seq_len, features]
	    # swap axis to get batch_first=False
        # time = time.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        
        x = x.permute(1, 0, 2)
        time = time.permute(1, 0)
        
        out, _ = self.tglstm(x, time)
        x = self.fc1(out[-1,:,  :])#
        x = F.relu(x)
        x = self.fc2(x)
        y = F.softmax(x, dim=1)     
        return y
        
    def scale_data(self, x):
        # Scale the input data using the mean and std values of the training set
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        else:
            self.mean = torch.mean(x, dim=(0, 1))
            self.std = torch.std(x, dim=(0, 1))
            x = (x - self.mean) / self.std
        return x
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)

    def training_step(self, batch, batch_idx):
        x, y, train_time = batch
        # x = self.scale_data(x)
        y_hat = self.forward(x, train_time)  #([128, 50])
        print(y)
        print(y_hat)
        # print(torch.argmax(y_hat,dim=1))
        #y = torch.argmax(y, dim=1)  #([128])
        # y = torch.tensor(y, dtype=torch.int64)
        loss = F.cross_entropy(y_hat,y) #Its first argument, input, must be the output logit of your model, of shape (N, C), where C is the number of classes and N the batch size (in general)
                                         #The second argument, target, must be of shape (N), and its elements must be integers (in the mathematical sense) in the range [0, C-1].
        y_pred = torch.argmax(y_hat, dim=1)
        # y_pred = torch.squeeze(y_pred)
        # y_pred = y_pred.float()
        # print(y_pred)
        # #y_pred = y_pred.type(torch.int32)  #([128])

        # y_true = torch.argmax(y, dim=1)
        # print(y_true)
        train_acc = self.accuracy(y_pred, y)
        
        metrics = {"train_acc": train_acc, "train_loss": loss}
        self.log_dict(metrics,on_epoch=True)
        return {'loss': loss, 'train_acc': train_acc, 'preds': y_hat, 'target': y}
    


    def validation_step(self, batch, batch_idx):
        x, y, val_time = batch
        # x = self.scale_data(x)
        y_hat = self.forward(x, val_time)
        #y = torch.argmax(y, dim=1)
        # y = torch.tensor(y, dtype=torch.int64)
        loss = F.cross_entropy(y_hat, y)
        y_pred = torch.argmax(y_hat, dim=1)
        #y_pred = y_pred.type(torch.int32)
        # y_pred = torch.squeeze(y_pred)
        # y_true = torch.argmax(y, dim=1)
        # y_pred = y_pred.float()
        if self.current_epoch == self.trainer.max_epochs - 1:
            self.targets = self.targets.to(y.device)
            self.preds = self.preds.to(y_pred.device)
            self.targets = torch.cat([self.targets, y], dim=0)
            self.preds = torch.cat([self.preds, y_pred], dim=0)
        val_acc = self.accuracy(y_pred, y)
        metrics = {"val_acc": val_acc, "val_loss": loss}
        self.log_dict(metrics,on_epoch=True)
        return {'loss': loss, 'val_acc': val_acc, 'preds': y_hat, 'target': y}
    
    def on_validation_epoch_end(self):
        if self.current_epoch == self.trainer.max_epochs - 1:
            preds_cpu = self.preds.cpu()
            targets_cpu = self.targets.cpu()
            # Calculate the confusion matrix
            cm = confusion_matrix(targets_cpu,preds_cpu)
            names = ('YoYo', 'WalkingWithDog', 'VolleyballSpiking', 'TrampolineJumping', 'ThrowDiscus', 'TennisSwing', 'TaiChi', 'Swing', 'SoccerJuggling','Skijet')
            plt.figure(figsize=(20,10))
            plot_confusion_matrix(cm, names)
            plt.savefig('/nas/lrz/tuei/ldv/studierende/data/video/i-frame-tglstm-8/1/myfigure.png')
    
    def test_step(self, batch, batch_idx):
        x, y, test_time = batch
        # x = self.scale_data(x)
        y_hat = self.forward(x, test_time)
        #y = torch.argmax(y, dim=1)
        # y = torch.tensor(y, dtype=torch.int64)
        loss = F.cross_entropy(y_hat, y)
        y_pred = torch.argmax(y_hat, dim=1)
        # y_pred = torch.squeeze(y_pred)
        # #y_pred = y_pred.type(torch.int32)
        # y_pred = y_pred.float()
        # y_true = torch.argmax(y, dim=1)
        test_acc = self.accuracy(y_pred, y)
        metrics = {"test_acc": test_acc, "test_loss": loss}
        self.log_dict(metrics,on_epoch=True)
        return {'loss': loss, 'test_acc': test_acc, 'preds': y_hat, 'target': y}
    
p = dict(
    #seq_len = 84,
    batch_size = 8, 
    max_epochs = 100,
    n_features = 512,
    hidden_size = 512,
    num_layers = 2,
    dropout = 0.2,
    learning_rate = 0.001,
)




seed_everything(1)
# seeds = [42, 123, 987, 888, 666, 111, 999, 555, 777, 1]
csv_logger = CSVLogger('/nas/lrz/tuei/ldv/studierende/data/video/', name='i-frame-tglstm-8', version='1'),


# Init ModelCheckpoint callback, monitoring 'val_loss'
checkpoint_callback = ModelCheckpoint(monitor="val_loss") 

trainer = Trainer(
        max_epochs=p['max_epochs'],
        logger=csv_logger,log_every_n_steps=20,
        callbacks=[checkpoint_callback])

model =  TimeGatedLSTM(
        n_features = p['n_features'],
        hidden_size = p['hidden_size'],
        #seq_len = p['seq_len'],
        batch_size = p['batch_size'],
        num_layers = p['num_layers'],
        dropout = p['dropout'],
        learning_rate = p['learning_rate']
    )

dm = DataModule(batch_size = p['batch_size'])
trainer.fit(model, dm)
trainer.test(model, dm)
