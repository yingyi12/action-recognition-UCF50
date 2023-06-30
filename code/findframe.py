import cv2
import os
import glob
def find_frame(video_path, image_path):
    # 加载视频
    cap = cv2.VideoCapture(video_path)

    # 加载要查找的图像
    image = cv2.imread(image_path)

    # 初始化变量
    frame_count = 0
    found = False

    while True:
        # 逐帧读取视频
        ret, frame = cap.read()

        # 检查视频是否结束
        if not ret:
            break

        # 增加帧计数
        frame_count += 1

        # 在当前帧上查找图像
        result = cv2.matchTemplate(frame, image, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # 如果找到图像，停止循环
        if max_val >= 0.996:
            found = True
            break

    # 释放视频对象
    cap.release()

    if found:
        return frame_count
    else:
        return 1


# 示例用法
video_path = 'C:\\UCF50\\test\\'
image_path = 'C:\\test-i-frame\\'
ucf_class = [cla for cla in os.listdir(video_path)
        if os.path.isdir(os.path.join(video_path, cla))] 

for cla in ucf_class:
    vd_path = os.path.join(video_path, cla)
    img_path=os.path.join(image_path, cla)
    
    for num in os.listdir(img_path):
        last_folder_name = os.path.basename(num)
        v_name=last_folder_name+'.avi'
        v_path=os.path.join(vd_path, v_name)
        im_path=os.path.join(img_path,last_folder_name)
        
        jpeg_pattern = os.path.join(im_path, '*.jpeg')

        # 使用glob匹配所有的JPEG文件
        jpeg_files = glob.glob(jpeg_pattern)
        print(jpeg_files)
        numbers=[]
        for file_path in jpeg_files:
            
            frame_number = find_frame(v_path, file_path)
            txt_file_name=num+"_index.txt"
            txt_file=os.path.join(im_path,txt_file_name)
            numbers.append(str(frame_number))
            print(numbers)
        with open(txt_file, 'w+') as file:
            file.write('\n'.join(numbers))
            print(numbers)  # 打印当前数字（可选）
                
            
            
#frame_number = find_frame(video_path, image_path)

