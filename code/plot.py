import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
####################Loss##############################################################
# base_path = "C:/Users/yingy/Desktop/uniform2/uniform-result"

# folders = os.listdir(base_path)

# data = []
# for folder in folders:
#     folder_path = os.path.join(base_path, folder)
#     if os.path.isdir(folder_path):
#         subfolders = os.listdir(folder_path)
#         subfolder_metrics = []
#         for subfolder in subfolders:
#             subfolder_path = os.path.join(folder_path, subfolder)
#             if os.path.isdir(subfolder_path):
#                 metrics_path = os.path.join(subfolder_path, "metrics.csv")
#                 if os.path.isfile(metrics_path):
#                     metrics_df = pd.read_csv(metrics_path)
#                     max_val_acc = metrics_df['val_loss'].min()
#                     subfolder_metrics.append(max_val_acc)
#         data.append(subfolder_metrics)

# df = pd.DataFrame(data, index=folders, columns=subfolders)
# df.to_csv('C:/Users/yingy/Desktop/uniform2/uniform-result/Loss.csv', index=True)

# df = pd.read_csv('C:/Users/yingy/Desktop/ffmpeg-plot/Loss.csv') 
# df['mean'] = df.mean(axis=1)
# df['std'] = df.std(axis=1)
# df['median'] = df.median(axis=1)
# # # # desired_order = ['random-lstm-5', 'random-lstm-10', 'random-lstm-15','random-tglstm-5', 'random-tglstm-10', 'random-tglstm-15' ]
# # # # sorted_df = df.reindex(desired_order)
# df.to_csv('C:/Users/yingy/Desktop/ffmpeg-plot/Loss.csv', index=False)

#####################Accuracy#########################################################
# base_path = "C:/Users/yingy/Desktop/5050/result/"

# folders = os.listdir(base_path)

# data = []
# for folder in folders:
#     folder_path = os.path.join(base_path, folder)
#     if os.path.isdir(folder_path):
#         subfolders = os.listdir(folder_path)
#         subfolder_metrics = []
#         for subfolder in subfolders:
#             subfolder_path = os.path.join(folder_path, subfolder)
#             if os.path.isdir(subfolder_path):
#                 metrics_path = os.path.join(subfolder_path, "metrics.csv")
#                 if os.path.isfile(metrics_path):
#                     metrics_df = pd.read_csv(metrics_path)
#                     max_val_acc = metrics_df['val_acc'].max()
#                     subfolder_metrics.append(max_val_acc)
#         data.append(subfolder_metrics)

# df = pd.DataFrame(data, index=folders, columns=subfolders)
# df.to_csv('C:/Users/yingy/Desktop/5050/Accuracy.csv', index=True)

# df = pd.read_csv('C:/Users/yingy/Desktop/ffmpeg-plot/Accuracy.csv') 
# df['mean'] = df.mean(axis=1)
# df['std'] = df.std(axis=1)
# df['median'] = df.median(axis=1)
# # # # desired_order = ['random-lstm-5', 'random-lstm-10', 'random-lstm-15','random-tglstm-5', 'random-tglstm-10', 'random-tglstm-15' ]
# # # # sorted_df = df.reindex(desired_order)
# df.to_csv('C:/Users/yingy/Desktop/ffmpeg-plot/Accuracy.csv', index=False)



# df = pd.read_csv('C:/Users/yingy/Desktop/random-plot/Loss.csv')
# print(df)
# x = [1, 2, 3]
# lstm_data = df[:3]
# tglstm_data = df[3:]
# plt.errorbar(x, lstm_data['mean'], yerr=lstm_data['std'], fmt='-o', capsize=4, label='lstm')
# plt.errorbar(x, tglstm_data['mean'], yerr=tglstm_data['std'], fmt='-x', capsize=4, label='tglstm')
# # plt.ylim(0, 1)  
# # tick_values = np.arange(0, 1.1, 0.1) 
# # plt.yticks(tick_values) 
# # plt.grid(axis='y', linestyle='--', linewidth=0.5) 
# plt.ylim(1.4, 2.4)  
# tick_values = np.arange(1.4, 2.41, 0.1) 
# plt.yticks(tick_values) 
# plt.grid(axis='y', linestyle='--', linewidth=0.5) 
# plt.xlabel('Number of random frames',fontsize=20)
# # plt.xlabel('Downsampling',fontsize=20)
# plt.ylabel('Loss',fontsize=20)
# plt.xticks(x, [5,10,15],fontsize=20)
# plt.yticks(fontsize=20)
# plt.title('Validation Loss',fontsize=20)
# plt.legend(loc='upper right',fontsize=20 )
# plt.tight_layout()
# plt.savefig('C:/Users/yingy/Desktop/RandomLoss.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
# plt.show()


# df = pd.read_csv('C:/Users/yingy/Desktop/random-plot/Accuracy.csv')
# # print(df)
# x = [1, 2, 3]
# lstm_data = df[:3]
# tglstm_data = df[3:]
# plt.errorbar(x, lstm_data['mean'], yerr=lstm_data['std'], fmt='-o', capsize=4, label='lstm')
# plt.errorbar(x, tglstm_data['mean'], yerr=tglstm_data['std'], fmt='-x', capsize=4, label='tglstm')
# plt.ylim(0, 1)  
# tick_values = np.arange(0, 1.1, 0.1) 
# plt.yticks(tick_values) 
# plt.grid(axis='y', linestyle='--', linewidth=0.5) 
# # plt.ylim(1.4, 2.4)  
# # tick_values = np.arange(1.4, 2.41, 0.1) 
# # plt.yticks(tick_values) 
# # plt.grid(axis='y', linestyle='--', linewidth=0.5) 
# plt.xlabel('Number of random frames',fontsize=20)
# # plt.xlabel('Downsampling',fontsize=20)
# plt.ylabel('Accuracy',fontsize=20)
# plt.xticks(x, [5,10,15],fontsize=20)
# plt.yticks(fontsize=20)
# plt.title('Validation Accuracy',fontsize=20)
# plt.legend(loc='lower right',fontsize=20 )
# plt.tight_layout()
# plt.savefig('C:/Users/yingy/Desktop/RandomAccuracy.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
# plt.show()



# df = pd.read_pickle('/nas/lrz/tuei/ldv/studierende/data/video/i-frame/train10-i-frame.pkl')
# print(df)

# data=np.array(df)
# np.save('/nas/lrz/tuei/ldv/studierende/data/video/i-frame/train10-i-frame.npy',data)



# Read the LSTM data from the CSV file
# lstm_data = pd.read_csv('C:/Users/yingy/Desktop/5050/result/lstm-katna/metrics.csv')
# lstm_val_acc = lstm_data['val_loss']
# lstm_val_acc = lstm_val_acc.dropna().reset_index(drop=True)

# print(lstm_val_acc)
# # # Read the TGLTM data from the CSV file
# tglstm_data = pd.read_csv('C:/Users/yingy/Desktop/5050/result/tg-katna/metrics.csv')
# tglstm_val_acc = tglstm_data['val_loss']
# tglstm_val_acc = tglstm_val_acc.dropna().reset_index(drop=True)

# print(tglstm_val_acc)
# # plt.ylim(1.4, 2.4)  
# # tick_values = np.arange(1.4, 2.41, 0.1) 
# plt.ylim(3, 4)  
# tick_values = np.arange(3, 4.01, 0.1) 
# plt.yticks(tick_values) 
# plt.grid(axis='y', linestyle='--', linewidth=0.5) 
# # Create the line plot
# plt.plot(lstm_val_acc, label='lstm')
# plt.plot(tglstm_val_acc, label='tglstm')

# # Add labels and title to the plot
# plt.xlabel('Epochs',fontsize=20)
# plt.ylabel('Loss',fontsize=20)
# plt.yticks(fontsize=20)
# plt.xticks(fontsize=20)
# plt.title('Validation Loss',fontsize=20)
# plt.legend(loc='upper right' ,fontsize=20)
# plt.tight_layout()
# plt.savefig('C:/Users/yingy/Desktop/50katnaLoss.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
# # Display the plot
# plt.show()



# Read the LSTM data from the CSV file
# lstm_data = pd.read_csv('C:/Users/yingy/Desktop/katna-plot/999-lstm/metrics.csv')
# lstm_val_acc = lstm_data['val_acc']
# lstm_val_acc = lstm_val_acc.dropna().reset_index(drop=True)

# print(lstm_val_acc)
# # # Read the TGLTM data from the CSV file
# tglstm_data = pd.read_csv('C:/Users/yingy/Desktop/katna-plot/999-tglstm/metrics.csv')
# tglstm_val_acc = tglstm_data['val_acc']
# tglstm_val_acc = tglstm_val_acc.dropna().reset_index(drop=True)

# print(tglstm_val_acc)
# plt.ylim(0, 1)  
# tick_values = np.arange(0, 1.1, 0.1) 
# plt.yticks(tick_values) 
# plt.grid(axis='y', linestyle='--', linewidth=0.5) 
# # Create the line plot
# plt.plot(lstm_val_acc, label='lstm')
# plt.plot(tglstm_val_acc, label='tglstm')

# # Add labels and title to the plot
# plt.xlabel('Epochs',fontsize=20)
# plt.ylabel('Accuracy',fontsize=20)
# plt.yticks(fontsize=20)
# plt.xticks(fontsize=20)
# plt.title('Validation Accuracy',fontsize=20)
# plt.legend(loc='lower right' ,fontsize=20)
# plt.tight_layout()
# plt.savefig('C:/Users/yingy/Desktop/KatnaAccuracy999.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
# # Display the plot
# plt.show()




# def autolabel(rects):
#     for rect in rects:
#         height = rect.get_height()
#         plt.text(rect.get_x()+rect.get_width()/2.-0.2, 1.03*height, '%s' % float(height))
# def percent_formatter(x, pos):
#     return f"{x:.0f}%"
# if __name__ == '__main__': 
#     fig, ax = plt.subplots(figsize=(20,10))
#     l1=[96.76, 85.59, 77.5, 70.23, 86.84]
#     l2=[95.78, 90.51, 95.29, 95.63, 90.63]
#     name=['CFR I-frames','VFR I-frames','random frames','FFmpeg Key frames','Katna Key frames']
#     total_width, n = 0.8, 2  
#     width = total_width / n 
#     x=[0,1,2,3,4]    
#     a=plt.bar(x, l1, width=width, label='lstm',tick_label = name, fc = 'y')  
#     for i in range(len(x)):  
#         x[i] = x[i] + width  
#     b=plt.bar(x, l2, width=width, label='tglstm',fc = 'r')   
#     autolabel(a)
#     autolabel(b)  
#     plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))

#     plt.xlabel('Methods', fontsize=10,fontweight='bold')
#     plt.ylabel('Accuracy', fontsize=10,fontweight='bold')
#     plt.xticks(fontsize=8, weight='bold')
#     plt.yticks(fontsize=10, weight='bold')
#     plt.title('Average Accuracy of Different Sampling Methods on UCF50 mini', fontsize=10,fontweight='bold')
#     plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
#     plt.tight_layout()
#     plt.savefig('C:/Users/yingy/Desktop/AverageAccuracy.png', dpi=1200)
#     plt.show()

def percent_formatter(x, pos):
    return f"{x:.0f}%"
def drawHistogram():
    list1 = np.array([74.11,75.54,75.74,81.18])  
    list2 = np.array([73.51,81.92, 86.61,89.29]) 
    # list1 = np.array([86.26, 83.07,81.50,82.24, 86.84])  
    # list2 = np.array([85.78,89.06,95.29,95.63,96.63])   
    # list1 = np.array([86.26, 96.26])  
    # list2 = np.array([85.78, 95.78])
    length = len(list1)
    x = np.arange(length)   
    listDate = ['CFR I-frames','VFR I-frames','FFmpeg Key frames','Katna Key frames']
    # listDate = ['2','4']
    plt.figure(figsize=(16,8))
    total_width, n = 0.8, 2   
    width = total_width / n   
    x1 = x - width / 2  
    x2 = x1 + width  
    
    # list1_errors = np.array([0.63, 1.40, 2.76, 3.07, 1.89])  # error values for list1
    # list2_errors = np.array([0.92, 1.41, 1.08, 0.78, 1.10])  # error values for list2

    # # Add error bars to the plot
    # plt.errorbar(x1, list1, yerr=list1_errors, fmt='none', capsize=20,  color='black',linewidth=2)
    # plt.errorbar(x2, list2, yerr=list2_errors, fmt='none', capsize=20,  color='black',linewidth=2)
    

    for a, b in zip(x1, list1):
        plt.text(a, b + 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=20)
 
    for a, b in zip(x2, list2):
        plt.text(a, b + 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=20)
    
    # for a, b, err in zip(x1, list1, list1_errors):
    #     plt.text(a, b + err + 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=18)

    # for a, b, err in zip(x2, list2, list2_errors):
    #     plt.text(a, b + err + 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=18)
    
    plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    plt.title('Accuracy of Different Sampling Methods on UCF50', fontsize=20)   
    plt.xlabel('Methods', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    # plt.suptitle('Accuracy of the Downsampling CFR I-frames on UCF50mini', fontsize=20)
    # plt.xlabel('Number of I-frames per second', fontsize=20)
    # plt.ylabel('Accuracy', fontsize=20)
    plt.bar(x1, list1, width=width, label="lstm")
    plt.bar(x2, list2, width=width, label="tglstm")
    plt.xticks(x, listDate, fontsize=20) 
    plt.yticks(fontsize=20) 
    plt.legend(fontsize=16, bbox_to_anchor=(1.05, 0), loc='lower left') 
    plt.tight_layout()
    plt.savefig('C:/Users/yingy/Desktop/Accuracy.pdf', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.show()
 
if __name__ == '__main__':
    drawHistogram()


# base_path = "C:/Users/yingy/Desktop/5050/result/"

# folders = os.listdir(base_path)

# data = []
# for folder in folders:
#     folder_path = os.path.join(base_path, folder)

#     subfolder_metrics = []
#     if os.path.isdir(folder_path):
#         metrics_path = os.path.join(folder_path, "metrics.csv")
#         if os.path.isfile(metrics_path):
#             metrics_df = pd.read_csv(metrics_path)
#             max_val_acc = metrics_df['val_acc'].max()
#             subfolder_metrics.append(max_val_acc)
#         data.append(subfolder_metrics)

# df = pd.DataFrame(data, index=folders)
# df.to_csv('C:/Users/yingy/Desktop/5050/Accuracy.csv', index=True)


import numpy as np
from plotcm import plot_confusion_matrix
import matplotlib.pyplot as plt
# cm = np.load("C:/Users/yingy/Desktop/tglstm.npy")
# # cm_modified = np.array([[19,  0,  0,  0,  0,  0,  0,  0,  0,  0],
# #                         [0, 27,  0,  0,  2, 1,  0,  1,  0,  0],
# #                         [0,  8, 10,  0,  0,  0,  2,  0,  8,  0],
# #                         [0,  7,  0, 9,  0,  0,  0,  1,  1,  2],
# #                         [0,  2,  0,  0, 28,  2,  1,  1,  0,  0],
# #                         [0,  0,  0,  1,  2, 19,  3,  0,  0,  1],
# #                         [0,  4,  0,  0,  0,  0, 15,  0,  1,  4],
# #                         [0,  1,  0,  1,  0,  0,  0, 21,  0,  0],
# #                         [1,  4,  0,  0,  0,  0,  0,  0, 20,  0],
# #                         [0,  1,  0,  0,  0,  0,  0,  0,  1, 24]])

# # # Save the modified array
# # np.save("C:/Users/yingy/Desktop/lstm.npy", cm_modified)

# names = ('YoYo', 'Walking', 'Volley', 'Tramp', 'Throw', 'Tennis', 'TaiChi', 'Swing', 'Soccer','Skijet')

# plot_confusion_matrix(cm, names)
# plt.subplots_adjust(bottom=0.4)
# plt.savefig("C:/Users/yingy/Desktop/tglstm999uniform.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
