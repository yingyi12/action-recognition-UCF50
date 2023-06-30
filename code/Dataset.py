from time import time
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
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
import copy

# df_train = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/train-i-frame.pkl')
# df_val = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/val-i-frame.pkl')
# df_test = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/test-i-frame.pkl')
# df = pd.concat([df_train, df_val, df_test], ignore_index=True)


# max_len = max(len(x) for x in df['Tensor'])
# padded = torch.nn.utils.rnn.pad_sequence(df['Tensor'].apply(lambda x: torch.tensor(x)), batch_first=True, padding_value=0)
# num_sample = padded.size(0)
# seq_length = padded.size(1) // 512
# final_shape = (num_sample, seq_length,   512)
# padded = padded.view(final_shape)
# mean = torch.mean(padded, dim=(0, 1), keepdim=True)
# std = torch.std(padded, dim=(0, 1), keepdim=True)
# x_normalized = (padded - mean) / std
# x_normalized = df['Tensor']

# X_train, X_val, X_test = torch.split(x_normalized, [3980, 1346, 1355], dim=0)  #(batch, seq,  feature)([3980, 75, 512])

# # Encode the labels
# label_row = df['Video']
# encoder = LabelEncoder()
# label = encoder.fit_transform([s.split("_")[1] for s in label_row])
# label_tensor = torch.tensor(label)
# #label_reshaped = label.reshape(-1, 1)
  
# # Split the labels
# y_train, y_val, y_test = torch.split(label_tensor, [3980, 1346, 1355], dim=0) 

# def prepare_dataset(df):
#     # Encode the labels
#     encoder = LabelEncoder()
#     df['Video'] = encoder.fit_transform([s.split("_")[1] for s in df['Video']])  
    
#     X = []
#     y = []
#     for i in range(len(df)):
#         sequence = df.iloc[i]['Tensor']
#         label = df.iloc[i]['Video']
#         X.append(sequence) 
#         y.append(label)
    
#     return X, y

# X_train, y_train = prepare_dataset(df_train) 

# class Dataset(Dataset):
#     def __init__(self, data):
#         self.data = data

#     def __getitem__(self, index):
#         sequences = [d[0] for d in data]
#         labels = [d[1] for d in data]
#         x = [torch.tensor(seq) for seq in sequences]
#         y = torch.tensor(labels)

#         return x, len(x), y

#     def __len__(self):
#         return len(self.data)


# def collate_fn(batch):
#     batch = sorted(batch, key=lambda x: x[1], reverse=True)
#     data_length = [len(data) for data in batch]
#     train_data = pad_sequence(batch, batch_first=True, padding_value=0)
#     return train_data.unsqueeze(-1), data_length  

# class Dataset(Dataset):
#     def __init__(self, sequences, labels):
#         self.sequences =[torch.tensor(seq) for seq in sequences]
#         self.labels = [torch.tensor(label) for label in labels]

#     def __getitem__(self, index):
#         sequence = self.sequences[index]
#         label = self.labels[index]
#         return sequence, label

#     def __len__(self):
#         return len(self.sequences)


# def collate_fn(batch):
#     sequences = [item[0] for item in batch]
#     labels = [item[1] for item in batch]
#     lengths = [len(seq)// 512 for seq in sequences]
#     sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
#     labels = torch.tensor(labels)

#     return sequences,  labels, lengths

# df_train = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/i-frame/train3-i-frame.pkl')
# df_val = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/i-frame/val3-i-frame.pkl')
# df_test = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/i-frame/test3-i-frame.pkl')
# df = pd.concat([df_train, df_val, df_test], ignore_index=True)

#         # Pad the sequences
# max_len = max(len(x) for x in df['Tensor'])
# padded = torch.nn.utils.rnn.pad_sequence(df['Tensor'].apply(lambda x: torch.tensor(x)),
#                     batch_first=True,
#                     padding_value=0)
#         # Split the sequences
# num_sample = padded.size(0)
# seq_length = padded.size(1) // 150528
# final_shape = (num_sample, seq_length,  3, 224, 224)
# padded = padded.view(final_shape)
# X_train, X_val, X_test = torch.split(padded, [219, 74, 75], dim=0)  #(batch, seq,  feature)([3980, 75, 512])
    

#         # Encode the labels
# label_row = df['Label']
# encoder = LabelEncoder()
# label = encoder.fit_transform([s.split("_")[1] for s in label_row])
#         #label_reshaped = label.reshape(-1, 1)
# label_tensor = torch.tensor(label)

  
#         # Split the labels
# y_train, y_val, y_test = torch.split(label_tensor, [219, 74, 75], dim=0)
# # train_dataset = Dataset(X_train, y_train)

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
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
    return X, y


class Dataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences =[torch.tensor(seq) for seq in sequences]
        self.labels = [torch.tensor(label) for label in labels]
    
    def __getitem__(self, index):

        data = copy.deepcopy(self.sequences[index]) 
        data2 = copy.deepcopy(self.labels[index]) 
        return data, data2




    def __len__(self):
        return len(self.sequences)

def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    sequences  = torch.tensor(sequences ,dtype=torch.long)
    num_sample, seq_length= sequences.shape
    sequences = sequences.view(num_sample, seq_length // 150528, 3, 224, 224)
    labels = torch.stack(labels, dim=0)
    return sequences, labels




df_train = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/i-frame/train10-i-frame.pkl')
df_val = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/i-frame/val10-i-frame.pkl')
df_test = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/i-frame/test10-i-frame.pkl')

        
X_train, y_train= prepare_dataset(df_train)
X_val, y_val= prepare_dataset(df_val)
X_test, y_test = prepare_dataset(df_test)




train_dataset = Dataset(X_train,y_train)
train_loader = DataLoader(train_dataset, batch_size = 32, 
                              shuffle = True, pin_memory=True,
                              num_workers = 0)   

for batch in train_loader:
    x , y= batch
    print(x, np.shape(x))
    print(y, np.shape(y))
    # print(f"X: {batch[0][1][:512]}")
    # print(f"y: {batch[1][1:2]}")
    break


# print(np.shape(X_train))
# print(np.shape(y_train))