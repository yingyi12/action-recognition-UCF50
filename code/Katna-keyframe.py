from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import os
from shutil import rmtree, copy
#import findframe

# initialize video module

def mk_file(file_path):
    if os.path.exists(file_path):
      rmtree(file_path)
    os.makedirs(file_path)
    


# extract keyframes and process data with diskwriter
if __name__ == '__main__':
    cwd = 'C:\\UCF50\\'

    origin_ucf_path = os.path.join(cwd, 'test')
    ucf_class = [cla for cla in os.listdir(origin_ucf_path)
            if os.path.isdir(os.path.join(origin_ucf_path, cla))] 


    print(ucf_class)
    for cla in ucf_class:
        dir_path = os.path.join(origin_ucf_path, cla)
        file_list = [f for f in os.listdir(dir_path)]
        for file in file_list:
            print(os.path.splitext(file)[0])
            mk_file((os.path.join(os.path.join('C:\\test-i-frame\\', cla), os.path.splitext(file)[0])))

            input_file = os.path.join(dir_path, file)
            output = os.path.join('C:\\test-i-frame\\', cla)
            output_path = os.path.join(output, os.path.splitext(file)[0])
            output_file = os.path.join(output_path, os.path.splitext(file)[0]+'_i-frame-%2d.jpg')
            vd = Video()
    
    
             # number of images to be returned
            no_of_frames_to_returned = 10
    
    # initialize diskwriter to save data at desired location
            disk_writer = KeyFrameDiskWriter(location=output_path)
    
    # Video file path
            video_file_path = input_file
            vd.extract_video_keyframes(
                no_of_frames=no_of_frames_to_returned, file_path=video_file_path,
                writer=disk_writer
            )
