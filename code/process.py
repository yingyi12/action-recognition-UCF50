import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

def process_images(dir_path):
    df = pd.DataFrame(columns=['Label', 'Tensor'])
    ucf_class = [cla for cla in os.listdir('/nas/lrz/tuei/ldv/studierende/data/video/UCF50/train')]
    
    for cla in ucf_class:
        class_dir_path = os.path.join(dir_path, cla)
        file_list = [f for f in os.listdir(class_dir_path)]
        
        for file in file_list:
            file_path = os.path.join(class_dir_path, file)
            inputs = []
            
            for img in os.listdir(file_path):
                if img.endswith('.jpg'):
                    img_path = os.path.join(file_path, img)
                    transform1 = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    img = Image.open(img_path)
                    img1 = transform1(img)
                    x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)
                    y = x.data.numpy()
                    inputs.append(y)
            
            inputs = np.array(inputs).reshape(np.size(inputs))
            df.loc[len(df)] = [file, inputs]
    
    return df


