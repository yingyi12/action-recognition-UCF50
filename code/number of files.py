# https://blog.csdn.net/qq_36108664/article/details/107205942
import os
import pandas as pd
import cv2
# path1 = "/nas/lrz/tuei/ldv/studierende/data/video//keyframe/test-keyframe-0.1"                   # 输入一级文件夹地址
#                 # 输入一级文件夹地址
# # df = pd.DataFrame(columns=['name', 'number'])
# # files1 = os.listdir(path1)           # 读入一级文件夹
# # num1 = len(files1)                   # 统计一级文件夹中的二级文件夹个数
# # num2 = []                            # 建立空列表
# # num3 = []                            # 建立空列表

# # for i in range(num1):                # 遍历所有二级文件夹
# #     path2 = path1 + '//' + files1[i] # 某二级文件夹的路径
# #     files2 = os.listdir(path2)       # 读入二级文件夹
# #     num2.append(len(files2))         # 统计二级文件夹中的文件数量
    
# #     for j in range(len(files2)):     # 遍历所有三级文件夹
# #         path3 = path2 + '//' + files2[j]  # 某三级文件夹的路径
# #         files3 = os.listdir(path3)   # 读入三级文件夹
# #         num_files3 = len(files3)     # 统计三级文件夹中的文件数量
# #         num3.append(num_files3)      # 将统计结果添加到列表中
# #         new_row = {'name': files2[j], 'number': num_files3}
    
# #         # Add the new row to the dataframe
# #         df = df.append(new_row, ignore_index=True)
        
# def count_files(path):
#     df = pd.DataFrame(columns=['name', 'number'])
#     files1 =  os.listdir(path)        # Read in the first-level folder
#     num1 = len(files1)                   # Count the number of second-level folders
#     num2 = []                            # Create an empty list
#     num3 = []                            # Create an empty list

#     for i in range(num1):                # Loop through all the second-level folders
#         path2 = path + '//' + files1[i] # Path to a second-level folder
#         files2 = os.listdir(path2)       # Read in the second-level folder
#         num_files2 = len(files2)         # Count the number of files in the second-level folder
#         num2.append(num_files2)         # Add the count to the list of counts

#         for j in range(len(files2)):     # Loop through all the third-level folders
#             path3 = path2 + '//' + files2[j]  # Path to a third-level folder
#             files3 = os.listdir(path3)   # Read in the third-level folder
#             num_files3 = len(files3)     # Count the number of files in the third-level folder
#             num3.append(num_files3)      # Add the count to the list of counts
#             new_row = {'name': files2[j], 'number': num_files3}

#             # Add the new row to the dataframe
#             df = df.append(new_row, ignore_index=True)
#     return df


# df1 = count_files("C:/Users/yingy/Desktop/train-i-frame")
# df2 =  count_files("C:/Users/yingy/Desktop/val-i-frame")
# df3 =  count_files("C:/Users/yingy/Desktop/test-i-frame")

# result_df = pd.concat([df1, df2, df3], axis=0)

# # # Save the resulting dataframe to a CSV file
# result_df.to_csv('C:/Users/yingy/Desktop/iframe10.csv', index=False)

# Print the dataframe
# print(df)
        # print("三级文件夹名称：", files2[j])
        # print("三级文件夹中的文件数量：", num_files3)

# print("二级文件夹中的文件数量：", num2)
# print("三级文件夹中的文件数量：", num3)
    
# print("所有二级文件夹名:")
# print(files1)                        # 打印二级文件夹名称
# print("所有二级文件夹中的文件个数:")
# print(num2)                          # 打印二级文件夹中的文件个数

# print("对应输出:")
# xinhua = dict(zip(files1,num2))      # 将二级文件夹名称和所含文件个数组合成字典
# for key,value in xinhua.items():     # 将二级文件夹名称和所含文件个数对应输出
#     print('{key}:{value}'.format(key = key, value = value)) 
    
# import os
# path1 = "/nas/lrz/tuei/ldv/studierende/data/video/keyframe/test-keyframe/BaseballPitch"                           # 输入一级文件夹地址
# files1 = os.listdir(path1)           # 读入一级文件夹
# num1 = len(files1)                   # 统计一级文件夹中的二级文件夹个数
# num2 = []                            # 建立空列表
# for i in range(num1):                # 遍历所有二级文件夹
#     path2 = path1 +'//' +files1[i]   # 某二级文件夹的路径
#     files2 = os.listdir(path2)       # 读入二级文件夹
#     num2.append(len(files2))         # 二级文件夹中的文件个数
    
# print("所有二级文件夹名:")
# print(files1)                        # 打印二级文件夹名称
# print("所有二级文件夹中的文件个数:")
# print(num2)                          # 打印二级文件夹中的文件个数

# print("对应输出:")
# xinhua = dict(zip(files1,num2))      # 将二级文件夹名称和所含文件个数组合成字典
# for key,value in xinhua.items():     # 将二级文件夹名称和所含文件个数对应输出
#     print('{key}:{value}'.format(key = key, value = value))


df = pd.read_csv('C:/Users/yingy/Desktop/video.csv')
mean_value = df['frame_count'].mean()

print("mean:", mean_value)  


# def count_files(path):
#     df = pd.DataFrame(columns=['folder', 'name', 'frame_count'])

#     # Loop through the first-level folders
#     for folder_name in os.listdir(path):
#         folder_path = os.path.join(path, folder_name)  # Path to a first-level folder

#         # Check if the item is a folder
#         if os.path.isdir(folder_path):
#             # Loop through the video files in the second-level folder
#             for file_name in os.listdir(folder_path):
#                 file_path = os.path.join(folder_path, file_name)  # Path to a video file
#                 video = cv2.VideoCapture(file_path)  # Open the video file
#                 frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames in the video
#                 new_row = {'folder': folder_name, 'name': file_name, 'frame_count': frame_count}

#                 # Add the new row to the dataframe
#                 df = df.append(new_row, ignore_index=True)

#                 video.release()  # Release the video capture object

#     return df


# df1 = count_files("C:/Users/yingy/Desktop/master/UCF50")


# # # Save the resulting dataframe to a CSV file
# df1.to_csv('C:/Users/yingy/Desktop/video.csv', index=False)