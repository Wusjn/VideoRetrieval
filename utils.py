import os
import re
from PIL import Image
import numpy as np

def get_longest_shot(avi_dir):
    max_len_shot_num = 0
    max_len = 0
    for filename in os.listdir(avi_dir):
        info = re.match(r"shot_([0-9])_len_([0-9]+)_(.+)", filename)
        shot_num, lenth, pos = info.groups()
        shot_num, lenth = int(shot_num), int(lenth)
        if lenth > max_len:
            max_len = lenth
            max_len_shot_num = shot_num
    base_name = avi_dir + "/" + "shot_{0}_len_{1}_".format(str(max_len_shot_num), str(max_len))
    pics = []
    pics.append(base_name + "head.jpg")
    pics.append(base_name + "medium.jpg")
    pics.append(base_name + "tail.jpg")
    return pics


def pic_norm(pic_path, size):
    img = Image.open(pic_path)
    img = img.resize(size)
    img = np.asarray(img) / 255
    img = img.transpose()
    return img

def get_split(splits_dir, pattern):
    split = []
    for i, split_file_name in enumerate(os.listdir(splits_dir)):
        split_file_path = splits_dir + "/" + split_file_name
        end = split_file_name.find("_test")
        split_type = split_file_name[:end]

        with open(split_file_path, "r", encoding="utf-8") as file:
            for line in file.readlines():
                name, split_num = line.split()
                match = re.match(pattern, split_num)
                if match != None:
                    split.append((i, split_type, name))
    return split