from time import time
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
import torchmetrics
from torchmetrics import MetricCollection, Accuracy, F1Score
from torchmetrics.classification import MulticlassAccuracy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import os


'''df_train = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/train-i-frame.pkl')
df_val = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/val-i-frame.pkl')
df_test = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/test-i-frame.pkl')
df = pd.concat([df_train, df_val, df_test], ignore_index=True)
max_len = max(len(x) for x in df['Tensor'])
padded_a = torch.nn.utils.rnn.pad_sequence(df['Tensor'].apply(lambda x: torch.tensor(x)), batch_first=True, padding_value=0)
batch_size = padded_a.size(0)
seq_length = padded_a.size(1) // 512
final_shape = ( batch_size, seq_length, 512)
padded_a = padded_a.view(final_shape)
X_train, X_val, X_test = torch.split(padded_a, [3980, 1346, 1355], dim=0)
label_row = df['Video']
encoder = LabelEncoder()
label = encoder.fit_transform([s.split("_")[1] for s in label_row])
label_reshaped = label.reshape(-1, 1)

label_tensor = torch.tensor(label_reshaped)
y_train, y_val, y_test = torch.split(label_tensor, [3980, 1346, 1355], dim=0)
print(np.shape(y_train))'''

class Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
class DataModule(pl.LightningDataModule):

    
    def __init__(self,batch_size,num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.X_test = None
        self.y_hat=None

    def prepare_data(self):
        pass

    def setup(self, stage):

        if stage == 'fit' and self.X_train is not None and self.X_val is not None:
            return 
        if stage == 'test' and self.X_test is not None:
            return
        if stage is None and self.X_train is not None and self.X_test is not None and self.X_val is not None:  
            return
        
        # Load the data
        df_train = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/train-i-frame.pkl')
        df_val = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/val-i-frame.pkl')
        df_test = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/test-i-frame.pkl')
        df = pd.concat([df_train, df_val, df_test], ignore_index=True)

        # Pad the sequences
        max_len = max(len(x) for x in df['Tensor'])
        last_value = df['Tensor'].apply(lambda x: torch.tensor(x)).iloc[-1][-1]
        padded = torch.nn.utils.rnn.pad_sequence(df['Tensor'].apply(lambda x: torch.tensor(x)), batch_first=True, padding_value=last_value)

        # Split the sequences
        num_sample = padded.size(0)
        seq_length = padded.size(1) // 512
        final_shape = (num_sample, seq_length,   512)
        padded = padded.view(final_shape)
        mean = torch.mean(padded, dim=(0, 1), keepdim=True)
        std = torch.std(padded, dim=(0, 1), keepdim=True)
        x_normalized = (padded - mean) / std
        X_train, X_val, X_test = torch.split(x_normalized, [3980, 1346, 1355], dim=0)  #(batch, seq,  feature)([3980, 75, 512])
    

        # Encode the labels
        label_row = df['Video']
        encoder = LabelEncoder()
        label = encoder.fit_transform([s.split("_")[1] for s in label_row])
        #label_reshaped = label.reshape(-1, 1)
        label_tensor = torch.tensor(label)

  
        # Split the labels
        y_train, y_val, y_test = torch.split(label_tensor, [3980, 1346, 1355], dim=0)

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
                              shuffle = True, 
                              num_workers = self.num_workers)
        
        return train_loader

    def val_dataloader(self):
        val_dataset = Dataset(self.X_val,self.y_val)
        val_loader = DataLoader(val_dataset, batch_size = self.batch_size, 
                            shuffle = False, 
                            num_workers = self.num_workers)

        return val_loader

    def test_dataloader(self):
        test_dataset = Dataset(self.X_test,self.y_test)
        test_loader = DataLoader(test_dataset,  batch_size = self.batch_size, 
                              shuffle = False, 
                              num_workers = self.num_workers)

        return test_loader
    
class LSTMRegressor(pl.LightningModule):

    def __init__(self, 
                 n_features, 
                 hidden_size, 
                 seq_len, 
                 batch_size,
                 num_layers, 
                 dropout, 
                 learning_rate):
        super(LSTMRegressor, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=50)

        self.lstm = nn.LSTM(input_size=n_features, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 50)
        self.loss = nn.CrossEntropyLoss()
        
        
    def forward(
        self, x):
        x, _ = self.lstm(x) # x=(batch_size, seq_len, hidden_size)
        x = self.linear(x[:,-1])
        x = F.softmax(x, dim=1)
        return x
        

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)

    def training_step(self, batch, batch_idx):
        x, y = batch
        #print(x.shape)
        y_hat = self.forward(x)  #([128, 50])
        print(y)
        # print(torch.argmax(y_hat,dim=1))
        #y = torch.argmax(y, dim=1)  #([128])
        
        loss = F.cross_entropy(y_hat, y) #Its first argument, input, must be the output logit of your model, of shape (N, C), where C is the number of classes and N the batch size (in general)
                                         #The second argument, target, must be of shape (N), and its elements must be integers (in the mathematical sense) in the range [0, C-1].
        y_pred = torch.argmax(y_hat, dim=1)
        print(y_pred)
        #y_pred = y_pred.type(torch.int32)  #([128])
        train_acc = self.accuracy(y_pred, y)
        
        metrics = {"train_acc": train_acc, "train_loss": loss}
        self.log_dict(metrics,on_epoch=True)
        return {'loss': loss, 'train_acc': train_acc, 'preds': y_hat, 'target': y}
    


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        #y = torch.argmax(y, dim=1)
        loss = F.cross_entropy(y_hat, y)
        y_pred = torch.argmax(y_hat, dim=1)
        #y_pred = y_pred.type(torch.int32)
        
        val_acc = self.accuracy(y_pred, y)
        metrics = {"val_acc": val_acc, "val_loss": loss}
        self.log_dict(metrics,on_epoch=True)
        return {'loss': loss, 'val_acc': val_acc, 'preds': y_hat, 'target': y}
    


    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        #y = torch.argmax(y, dim=1)
        loss = F.cross_entropy(y_hat, y)
        y_pred = torch.argmax(y_hat, dim=1)
        #y_pred = y_pred.type(torch.int32)
        
        test_acc = self.accuracy(y_pred, y)
        metrics = {"test_acc": test_acc, "test_loss": loss}
        self.log_dict(metrics,on_epoch=True)
        return {'loss': loss, 'test_acc': test_acc, 'preds': y_hat, 'target': y}
    
p = dict(
    seq_len = 75,
    batch_size = 64, 
    max_epochs = 5,
    n_features = 512,
    hidden_size = 512,
    num_layers = 2,
    dropout = 0.2,
    learning_rate = 0.01,
)

seed_everything(1)

csv_logger = CSVLogger('/nas/lrz/tuei/ldv/studierende/data/video/', name='lstm', version='0'),
from pytorch_lightning.callbacks import ModelCheckpoint

# Init ModelCheckpoint callback, monitoring 'val_loss'
checkpoint_callback = ModelCheckpoint(monitor="val_loss")


trainer = Trainer(
    max_epochs=p['max_epochs'],
    logger=csv_logger,log_every_n_steps=50,
    callbacks=[checkpoint_callback])

model = LSTMRegressor(
    n_features = p['n_features'],
    hidden_size = p['hidden_size'],
    seq_len = p['seq_len'],
    batch_size = p['batch_size'],
    num_layers = p['num_layers'],
    dropout = p['dropout'],
    learning_rate = p['learning_rate']
)

dm = DataModule(batch_size = p['batch_size'])
trainer.fit(model, dm)
trainer.test(model, dm)

