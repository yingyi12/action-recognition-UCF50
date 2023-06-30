import torch
import numpy as np
import torch 
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt 
import random
import cv2
import ffmpeg
import pandas as pd
import torch.nn as nn
import os
import copy
import subprocess
from shutil import rmtree, copy
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
import shutil

def mk_file(file_path):
    if os.path.exists(file_path):
      rmtree(file_path)
    os.mkdir(file_path)

# cla = 'BaseballPitch'
# path = '/nas/lrz/tuei/ldv/studierende/data/video/UCF50/test/'
# dir_path = os.path.join(path, cla)
# file_list = [f for f in os.listdir(dir_path)]

# cwd = '/nas/lrz/tuei/ldv/studierende/data/video/UCF50'

# for file in file_list:
#   mk_file((os.path.join(os.path.join('/nas/lrz/tuei/ldv/studierende/data/video/i-frame/test-i-frame/',  cla), os.path.splitext(file)[0])))

#   input_file = os.path.join(dir_path, file)
#   output = os.path.join('/nas/lrz/tuei/ldv/studierende/data/video/i-frame/test-i-frame/', cla)
#   output_path = os.path.join(output, os.path.splitext(file)[0])
#   output_file = os.path.join(output_path, os.path.splitext(file)[0]+'_i-frame-%2d.jpg')

#   cmd = ['ffmpeg', '-i', input_file, '-vf', 'select=eq(pict_type\,I)', '-vsync', 'vfr', output_file]

#   proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#   out, err = proc.communicate()

#   if proc.returncode != 0:
#         print(f'Error running FFmpeg command: {err.decode()}')
#   else:
#         print('FFmpeg command completed successfully.')
  
  
# cwd = '/nas/lrz/tuei/ldv/studierende/data/video/UCF50'

# origin_ucf_path = os.path.join(cwd, 'train')
# ucf_class = [cla for cla in os.listdir(origin_ucf_path)
#         if os.path.isdir(os.path.join(origin_ucf_path, cla))]      

 
# for cla in ucf_class:
#   dir_path = os.path.join('/nas/lrz/tuei/ldv/studierende/data/video/UCF50/train/', cla)
#   file_list = [f for f in os.listdir(dir_path)]
#   for file in file_list:
#     mk_file((os.path.join(os.path.join('/nas/lrz/tuei/ldv/studierende/data/video/i-frame/train-i-frame/', cla), os.path.splitext(file)[0])))

#     input_file = os.path.join(dir_path, file)
#     output = os.path.join('/nas/lrz/tuei/ldv/studierende/data/video/i-frame/train-i-frame/', cla)
#     output_path = os.path.join(output, os.path.splitext(file)[0])
#     output_file = os.path.join(output_path, os.path.splitext(file)[0]+'_i-frame-%2d.jpg')

#     cmd = ['ffmpeg', '-i', input_file, '-vf', 'select=eq(pict_type\,I)', '-vsync', 'vfr', output_file]


#     proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     out, err = proc.communicate()

#     if proc.returncode != 0:
#         print(f'Error running FFmpeg command: {err.decode()}')
#     else:
#         print('FFmpeg command completed successfully.')

# "C:\Users\yingy\Desktop\master\UCF50\BaseballPitch\v_BaseballPitch_g01_c01.avi"

input_file = "C:/Users/yingy/Desktop/master/UCF50/Biking/v_Biking_g01_c01.avi"
output_path = "C:/Users/yingy/Desktop/I"
output_file = os.path.join(output_path, 'i-frame-%2d.jpg')
txt_file = os.path.join(output_path, "index.txt")
cmd = ['ffmpeg', '-i', input_file, '-vf', 'select=eq(pict_type\,I)' , '-vsync', 'vfr', output_file]


proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = proc.communicate()

if proc.returncode != 0:
  print(f'Error running FFmpeg command: {err.decode()}')
else:
  print('FFmpeg command completed successfully.')
