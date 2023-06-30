import os.path  
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable 
import pandas as pd    
import numpy as np
from PIL import Image 
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
import torchmetrics
from torchmetrics import MetricCollection, Accuracy, F1Score
from torchmetrics.classification import MulticlassAccuracy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image
import os

from torch.nn import Parameter
import torch.jit as jit
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
# import math

# df = pd.DataFrame(columns=['Label','Tensor'])
# # ucf_class = [cla for cla in os.listdir('/nas/lrz/tuei/ldv/studierende/data/video/test-keyframe')]
# ucf_class = ['YoYo', 'WalkingWithDog', 'VolleyballSpiking', 'TrampolineJumping', 'ThrowDiscus', 'TennisSwing', 'TaiChi', 'Swing', 'SoccerJuggling','Skijet']
# for cla in ucf_class:
#   dir_path = os.path.join('/nas/lrz/tuei/ldv/studierende/data/video/random/train11-random', cla)   # "/content/drive/My Drive/train-keyframe/Basketball"
#   file_list = [f for f in os.listdir(dir_path)]
  
#   for file in file_list:
#     file_path = os.path.join(dir_path, file)   # "/content/drive/My Drive/train-keyframe/Basketball/v_Basketball_g01_c02/"
#     inputs = []
#     for img in os.listdir(file_path):
      
#       if img.endswith('.jpg'): 
#         img_path = os.path.join(file_path, img)  
        
#         transform1 = transforms.Compose([
#           transforms.Resize(256),
#           transforms.CenterCrop(224),
#           transforms.ToTensor() , 
#           transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])

  
#         img = Image.open(img_path)
#         img1 = transform1(img)
#         x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)
#         y = x.data.numpy()
#         inputs.append(y)
#     inputs = np.array(inputs).reshape(np.size(inputs))
#     print(inputs.shape)
#     df.loc[len(df)] = [file, inputs]
    
# print(df)
# df.to_pickle('/nas/lrz/tuei/ldv/studierende/data/video/random/train11-random.pkl')
    
# print(ucf_class)
# ['YoYo', 'WalkingWithDog', 'VolleyballSpiking', 'TrampolineJumping', 'ThrowDiscus', 'TennisSwing', 'TaiChi', 'Swing', 'SoccerJuggling', 
#  'Skijet', 'Skiing', 'SkateBoarding', 'SalsaSpin', 'Rowing', 'RopeClimbing', 'RockClimbingIndoor', 'PushUps', 'Punch', 'PullUps', 'PommelHorse', 
#  'PoleVault', 'PlayingViolin', 'PlayingTabla', 'PlayingPiano', 'PlayingGuitar', 'PizzaTossing', 'Nunchucks', 'Mixing', 'MilitaryParade', 'Lunges', 
#  'Kayaking', 'JumpRope', 'JumpingJack', 'JugglingBalls', 'JavelinThrow', 'HulaHoop', 'HorseRiding', 'HorseRace', 'HighJump', 'GolfSwing', 
#  'Fencing', 'Drumming', 'Diving', 'CleanAndJerk', 'BreastStroke', 'Billiards', 'Biking', 'BenchPress', 'Basketball', 'BaseballPitch']



    
# dir_path = '/nas/lrz/tuei/ldv/studierende/data/video/test-keyframe/PommelHorse'  
# file_list = [f for f in os.listdir(dir_path)]
  
# for file in file_list:
#   file_path = os.path.join(dir_path, file)   # "/content/drive/My Drive/train-keyframe/Basketball/v_Basketball_g01_c02/"
#   inputs = []
#   for img in os.listdir(file_path):
      
#     if img.endswith('.jpg'): 
#       img_path = os.path.join(file_path, img)
    
#       transform1 = transforms.Compose([
#           transforms.Grayscale(),
#           transforms.Resize(256),
#           transforms.CenterCrop(224),
#           transforms.ToTensor() , 
#           transforms.Normalize([0.485], [0.229])])

  
#       img = Image.open(img_path)
#       img1 = transform1(img)
#       x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)
#       y = x.data.numpy()
#       print (y.shape)
#       inputs.append(y)
#   inputs = np.array(inputs).reshape(np.size(inputs))
#   print(inputs.shape)
#   df.loc[len(df)] = [file, inputs]
#   print(df)


# def create_dataframe(dir_path):
#     df = pd.DataFrame(columns=['Label','Tensor'])
#     ucf_class = [cla for cla in os.listdir(dir_path)]

#     for cla in ucf_class:
#         sub_dir_path = os.path.join(dir_path, cla)
#         file_list = [f for f in os.listdir(sub_dir_path)]
        
#         for file in file_list:
#             file_path = os.path.join(sub_dir_path, file)   # "/content/drive/My Drive/train-keyframe/Basketball/v_Basketball_g01_c02/"
#             inputs = []
#             for img in os.listdir(file_path):
      
#                 if img.endswith('.jpg'): 
#                     img_path = os.path.join(file_path, img)  
#                     transform1 = transforms.Compose([
#                         transforms.Resize(256),
#                         transforms.CenterCrop(224),
#                         transforms.ToTensor() , 
#                         transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])

  
#                     img = Image.open(img_path)
#                     img1 = transform1(img)
#                     x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)
#                     y = x.data.numpy()

#             inputs.append(y)
#             inputs = np.array(inputs).reshape(np.size(inputs))
#             df.loc[len(df)] = [file, inputs]
#     return df

# inputs = [] 
# last_pts = None
# with open('/nas/lrz/tuei/ldv/studierende/data/video/keyframe/train-keyframe/YoYo/v_YoYo_g02_c03/v_YoYo_g02_c03_index.txt', 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         if line.startswith('frame'):
#             pts_index = line.find('pts:')
#             pts_time_index = line.find('pts_time:')
#             pts = int(line[pts_index+4:pts_time_index-1])
#             if last_pts is not None:
#                 diff = pts - last_pts
#                 inputs.append(diff)
#             last_pts = pts
            
#     inputs = np.array(inputs)
#     print(inputs.shape)

# df = pd.DataFrame(columns=['Label','Tensor'])
# # ucf_class = [cla for cla in os.listdir('/nas/lrz/tuei/ldv/studierende/data/video/test-keyframe')]
# ucf_class =  ['YoYo', 'WalkingWithDog', 'VolleyballSpiking', 'TrampolineJumping', 'ThrowDiscus', 'TennisSwing', 'TaiChi', 'Swing', 'SoccerJuggling','Skijet']
# for cla in ucf_class:
#   dir_path = os.path.join('/nas/lrz/tuei/ldv/studierende/data/video/keyframe/train-keyframe-0.1', cla)   # "/content/drive/My Drive/train-keyframe/Basketball"
#   file_list = [f for f in os.listdir(dir_path)]
  
#   for file in file_list:
#     file_path = os.path.join(dir_path, file)   # "/content/drive/My Drive/train-keyframe/Basketball/v_Basketball_g01_c02/"
#     inputs = []
#     for img in os.listdir(file_path):
      
#       if img.endswith('.txt'): 
#         last_pts = None
#         img_path = os.path.join(file_path, img)
#         with open(img_path, 'r') as f:
#           lines = f.readlines()
#           for line in lines:
#             if line.startswith('frame'):
#               pts_index = line.find('pts:')
#               pts_time_index = line.find('pts_time:')
#               pts = int(line[pts_index+4:pts_time_index-1])
#               if last_pts is not None:
#                 diff = pts - last_pts
#                 inputs.append(diff)
#               last_pts = pts
            
#     inputs = np.array(inputs)
#     print(inputs.shape)
#     inputs = np.array(inputs).reshape(np.size(inputs))
#     print(inputs.shape)
#     df.loc[len(df)] = [file, inputs]
# print(len(df))
# df.to_pickle('/nas/lrz/tuei/ldv/studierende/data/video/keyframe/train10-keyframe-time-0.1.pkl')

# def create_dataframe(dir_path):
#     df = pd.DataFrame(columns=['Label','Tensor'])
#     ucf_class = [cla for cla in os.listdir(dir_path)]

#     for cla in ucf_class:
#         sub_dir_path = os.path.join(dir_path, cla)
#         file_list = [f for f in os.listdir(sub_dir_path)]
        
#         for file in file_list:
#             file_path = os.path.join(sub_dir_path, file)   # "/content/drive/My Drive/train-keyframe/Basketball/v_Basketball_g01_c02/"
#             inputs = []
#             for img in os.listdir(file_path):
      
#                 if img.endswith('.jpg'): 
#                     img_path = os.path.join(file_path, img)  
#                     transform1 = transforms.Compose([
#                         transforms.Resize(256),
#                         transforms.CenterCrop(224),
#                         transforms.ToTensor() , 
#                         transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])

  
#                     img = Image.open(img_path)
#                     img1 = transform1(img)
#                     x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)
#                     y = x.data.numpy()

#             inputs.append(y)
#             inputs = np.array(inputs).reshape(np.size(inputs))
#             df.loc[len(df)] = [file, inputs]
#     return df

# df = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/random/train11-random-time.pkl')  
# df['Tensor'] = df['Tensor'].apply(lambda x: np.concatenate((x, [1])))
# print(df)
# df.to_pickle('/nas/lrz/tuei/ldv/studierende/data/video/random/train11-random-time.pkl')
    
df1 = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/random/train11-random-time.pkl')  
df1 = df1.rename(columns={'Tensor': 'Time'})
df2 = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/random/train11-random.pkl') 
df = pd.merge(df1, df2, on='Label') 
print(df)
df.to_pickle('/nas/lrz/tuei/ldv/studierende/data/video/random/train11-random.pkl')


# df = pd.DataFrame(columns=['Label','Tensor'])
# # ucf_class = [cla for cla in os.listdir('/nas/lrz/tuei/ldv/studierende/data/video/test-keyframe')]
# ucf_class =  ['YoYo', 'WalkingWithDog', 'VolleyballSpiking', 'TrampolineJumping', 'ThrowDiscus', 'TennisSwing', 'TaiChi', 'Swing', 'SoccerJuggling','Skijet']
# for cla in ucf_class:
#   dir_path = os.path.join('/nas/lrz/tuei/ldv/studierende/data/video/random/train11-random', cla)   # "/content/drive/My Drive/train-keyframe/Basketball"
#   file_list = [f for f in os.listdir(dir_path)]
  
#   for file in file_list:
#     file_path = os.path.join(dir_path, file)   # "/content/drive/My Drive/train-keyframe/Basketball/v_Basketball_g01_c02/"
#     inputs = []
#     for img in os.listdir(file_path):
      
#       if img.endswith('.txt'): 
#         img_path = os.path.join(file_path, img)
#         with open(img_path, 'r') as f:
#             num_list = [int(line.strip()) for line in f.readlines()]
#             for i in range(1, len(num_list)):
#                 inputs.append(num_list[i] - num_list[i-1])
            
#     inputs = np.array(inputs)
#     print(inputs.shape)
#     inputs = np.array(inputs).reshape(np.size(inputs))
#     print(inputs.shape)
#     df.loc[len(df)] = [file, inputs]
# print(df)
# df.to_pickle('/nas/lrz/tuei/ldv/studierende/data/video/random/train11-random-time.pkl')


