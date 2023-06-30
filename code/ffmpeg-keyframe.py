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

print(torch.cuda.is_available())

def mk_file(file_path):
    if os.path.exists(file_path):
      rmtree(file_path)
    os.mkdir(file_path)
    
'''input_file = '/nas/lrz/tuei/ldv/studierende/data/video/test/BaseballPitch/v_BaseballPitch_g01_c05.avi'
output_pattern = '/nas/lrz/tuei/ldv/studierende/data/video/test-keyframe/BaseballPitch/v_BaseballPitch_g01_c05/yosemiteThumb%03d.png'

cmd = ['ffmpeg', '-i', input_file, '-vf', 'select=gt(scene\,0.05),metadata=print:file=/nas/lrz/tuei/ldv/studierende/data/video/test-keyframe/BaseballPitch/v_BaseballPitch_g01_c05/scenescores.txt', '-vsync', 'vfr', output_pattern]

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = proc.communicate()

if proc.returncode != 0:
    print(f'Error running FFmpeg command: {err.decode()}')
else:
    print('FFmpeg command completed successfully.')'''
    
    
# cwd = '/nas/lrz/tuei/ldv/studierende/data/video/random'

# origin_ucf_path = os.path.join('/nas/lrz/tuei/ldv/studierende/data/video/UCF50', 'val')
# ucf_class = [cla for cla in os.listdir(origin_ucf_path)
#         if os.path.isdir(os.path.join(origin_ucf_path, cla))] 

# # folder train, val, test
# train_root = os.path.join(cwd, 'train-random11')
# mk_file(train_root)
# for cla in ucf_class:
#    mk_file(os.path.join(train_root, cla))

# val_root = os.path.join(cwd, 'val-random11')
# mk_file(val_root)
# for cla in ucf_class:
#    mk_file(os.path.join(val_root, cla))

# test_root = os.path.join(cwd, 'test-random11')
# mk_file(test_root)
# for cla in ucf_class:
#    mk_file(os.path.join(test_root, cla))
   

# cla = 'BaseballPitch'
# path = '/nas/lrz/tuei/ldv/studierende/data/video/UCF50/test/'
# dir_path = os.path.join(path, cla)
# file_list = [f for f in os.listdir(dir_path)]

# for file in file_list:
#     mk_file((os.path.join(os.path.join('/nas/lrz/tuei/ldv/studierende/data/video/random/test-random/',  cla), os.path.splitext(file)[0])))

#     input_file = os.path.join(dir_path, file)
#     output = os.path.join('/nas/lrz/tuei/ldv/studierende/data/video/random/test-random/', cla)
#     output_path = os.path.join(output, os.path.splitext(file)[0])
#     output_file = os.path.join(output_path, os.path.splitext(file)[0]+'_random-%2d.jpg')
#     txt_file = os.path.join(output_path, os.path.splitext(file)[0]+"_index.txt")
#     cmd = ['ffmpeg', '-i', input_file, '-vcodec', 'copy', '-f', 'null', '-']
#     output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)

#     # 解析输出结果中的帧数信息
#     output = output.decode('utf-8')
#     frame_index = output.find('frame=')
#     if frame_index != -1:
#         frame_str = output[frame_index+len('frame='):].split()[0]
#         frame_count = int(frame_str)
#     frame_list = list(range(1, frame_count+1))

#     # 从中随机选择10个数字，并排序
#     index = sorted(random.sample(frame_list, 10)) 
    
#     frame_select = '+'.join(['eq(n\\,{})'.format(i) for i in index])
#     cmd = ['ffmpeg', '-i', input_file, '-vf',  "select='{}',metadata=print:file={}".format(frame_select, txt_file), '-vsync', '0', output_file]    
    
   
#     proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     out, err = proc.communicate()

#     if proc.returncode != 0:
#         print(f'Error running FFmpeg command: {err.decode()}')
#     else:
#         print('FFmpeg command completed successfully.')
#     with open(txt_file, "w") as f:
#         for idx in index:
#             f.write(str(idx) + "\n")    
        
#   files1 = os.listdir(output_path)     
#   num_keyframes  = len(files1)  
#   if num_keyframes == 1:
#     # 如果关键帧数量为 0，则重新处理视频并提取前 5 张关键帧
#     shutil.rmtree(output_path)
#     os.makedirs(output_path)
#     cmd = ['ffmpeg', '-i', input_file, '-vf', 'select=eq(n\,0)+eq(n\,1)+eq(n\,2)+eq(n\,3)+eq(n\,4),metadata=print:file=%s' % txt_file, '-vsync', 'vfr', output_file]
#     cmd = ['ffmpeg', '-i', input_file, '-vf', 'select=gt(scene\,0.02),metadata=print:file=%s' % txt_file, '-vsync', 'vfr', output_file]
#     cmd = ['ffmpeg', '-i', input_file, '-vf', 'select=not(mod(n\\,100)),metadata=print:file=%s' % txt_file, '-vsync', 0, output_file]

#     subprocess.call(cmd)
# else:
#     # 如果关键帧数量不为 0，则不做处理
#     pass
        
cwd = '/nas/lrz/tuei/ldv/studierende/data/video/UCF50'

origin_ucf_path = os.path.join(cwd, 'val')
ucf_class = [cla for cla in os.listdir(origin_ucf_path)
        if os.path.isdir(os.path.join(origin_ucf_path, cla))]      

 
for cla in ucf_class:
  dir_path = os.path.join('/nas/lrz/tuei/ldv/studierende/data/video/UCF50/val/', cla)
  file_list = [f for f in os.listdir(dir_path)]
  for file in file_list:
    mk_file((os.path.join(os.path.join('/nas/lrz/tuei/ldv/studierende/data/video/random/val11-random/', cla), os.path.splitext(file)[0])))

    input_file = os.path.join(dir_path, file)
    output = os.path.join('/nas/lrz/tuei/ldv/studierende/data/video/random/val11-random/', cla)
    output_path = os.path.join(output, os.path.splitext(file)[0])
    output_file = os.path.join(output_path, os.path.splitext(file)[0]+'_random-%2d.jpg')
    txt_file = os.path.join(output_path, os.path.splitext(file)[0]+"_index.txt")
    
    cmd1 = ['ffmpeg', '-i', input_file, '-vcodec', 'copy', '-f', 'null', '-']
    output = subprocess.check_output(cmd1, stderr=subprocess.STDOUT)

    output = output.decode('utf-8')
    frame_index = output.find('frame=')
    if frame_index != -1:
        frame_str = output[frame_index+len('frame='):].split()[0]
        frame_count = int(frame_str)
    frame_list = list(range(1, frame_count+1))
    index = sorted(random.sample(frame_list, 11)) 
    
    frame_select = '+'.join(['eq(n\\,{})'.format(i) for i in index])
    cmd2 = ['ffmpeg', '-i', input_file, '-vf',  "select='{}',metadata=print:file={}".format(frame_select, txt_file), '-vsync', '0', output_file]  
      

    proc = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()

    if proc.returncode != 0:
        print(f'Error running FFmpeg command: {err.decode()}')
    else:
        print('FFmpeg command completed successfully.')
    with open(txt_file, "w") as f:
        for idx in index:
            f.write(str(idx) + "\n")