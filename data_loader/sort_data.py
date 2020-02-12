import os
import shutil
#shutil.copy(os.path.join(src, filename), os.path.join(des))
import re


def get_folder(path):
    file_lists = os.listdir(path)
    for file in file_lists:
        if not os.path.isdir(path + '/' + file):  # 排除非文件夹的文件
            file_lists.remove(file)
    return file_lists

path = 'dataset'
time_folders = get_folder(path)


for time_folder in time_folders:
    cycle_folders = get_folder(path + '/'+time_folder)
    for cycle_folder in cycle_folders:
        data_files = os.listdir(path + '/'+time_folder+'/'+cycle_folder)
        for data_file in data_files:

            (filename, extension) = os.path.splitext(data_file)
            index = re.findall('\d+', filename)
            if index :
                a = os.path.join(path, time_folder, cycle_folder, data_file)
                print(time_folder,cycle_folder,data_file,index)
                shutil.copy(os.path.join(path, time_folder,cycle_folder,data_file)
                            , os.path.join('train',index[0],time_folder+cycle_folder+data_file))

