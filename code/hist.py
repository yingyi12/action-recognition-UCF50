import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os


# data = pd.read_csv('/nas/lrz/tuei/ldv/studierende/data/video/keyframe/count-keyframe-0.1.csv')
# fig,ax = plt.subplots()
# ax.hist(data["number"],np.arange(0, 80, 10))
# ax.set_xticklabels('Number of Keyframes')
# ax.set_ylabel('Videos Count')
# ax.set_title('Histogram of Keyframe Counts with Changes Exceeding 10%')
# plt.savefig('/nas/lrz/tuei/ldv/studierende/data/video/keyframe/count-keyframe-0.1.png')


data1 = pd.read_csv('C:/Users/yingy/Desktop/count 50/count-uniform.csv')
data2 = pd.read_csv('C:/Users/yingy/Desktop/count 50/count-i-frame.csv')
data3 = pd.read_csv('C:/Users/yingy/Desktop/count 50/count-keyframe-0.1.csv')
data4 = pd.read_csv('C:/Users/yingy/Desktop/count 50/count-Katna.csv')

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15,15))

# 绘制第一个图
axs[0, 0].hist(data1["number"], np.arange(0, 90, 10))
axs[0, 0].set_xlabel('Number of CFR I-frames',fontsize=20)
axs[0, 0].set_ylabel('Videos Count',fontsize=20)
axs[0, 0].tick_params(axis='x', labelsize=20)
axs[0, 0].tick_params(axis='y', labelsize=20)
axs[0, 0].set_title('CFR I-frames Counts',fontsize=20)
axs[0, 0].set_ylim([0, 5000]) 

axs[0, 1].hist(data2["number"], np.arange(0, 90, 10))
axs[0, 1].set_xlabel('Number of VFR I-frames',fontsize=20)
axs[0, 1].set_ylabel('Videos Count',fontsize=20)
axs[0, 1].tick_params(axis='x', labelsize=20)
axs[0, 1].tick_params(axis='y', labelsize=20)
axs[0, 1].set_title('VFR I-frames Counts',fontsize=20)
axs[0, 1].set_ylim([0, 5000]) 

# 绘制第二个图
axs[1, 0].hist(data3["number"], np.arange(0, 90, 10))
axs[1, 0].set_xlabel('Number of Keyframes',fontsize=20)
axs[1, 0].set_ylabel('Videos Count',fontsize=20)
axs[1, 0].tick_params(axis='x', labelsize=20)
axs[1, 0].tick_params(axis='y', labelsize=20)
axs[1, 0].set_title('Keyframe Counts with FFmpeg',fontsize=20)
axs[1, 0].set_ylim([0, 5000]) 

# 绘制第三个图
axs[1, 1].hist(data4["number"], np.arange(0, 90, 10))
axs[1, 1].set_xlabel('Number of Keyframes',fontsize=20)
axs[1, 1].set_ylabel('Videos Count',fontsize=20)
axs[1, 1].tick_params(axis='x', labelsize=20)
axs[1, 1].tick_params(axis='y', labelsize=20)
axs[1, 1].set_title('Keyframe Counts with Katna',fontsize=20)
axs[1, 1].set_ylim([0, 5000]) 


# 调整子图之间的距离
plt.subplots_adjust(wspace=0.3)
plt.subplots_adjust(hspace=0.3)

# 保存图像
plt.savefig('C:/Users/yingy/Desktop/counts.pdf', dpi=300, bbox_inches='tight', pad_inches=0.2)
# 显示图像
plt.show()