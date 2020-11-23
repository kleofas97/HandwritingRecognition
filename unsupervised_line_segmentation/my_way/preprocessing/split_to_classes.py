import os
from shutil import copyfile

import re

FILE_TO_READ = r'F:\Studia\pythonProject\unsupervised_line_segmentation\my_way\patches\val'
FILE_TO_SAVE_0 = r'F:\Studia\pythonProject\unsupervised_line_segmentation\my_way\patches\val_2\image_0'
FILE_TO_SAVE_1 = r'F:\Studia\pythonProject\unsupervised_line_segmentation\my_way\patches\val_2\image_1'


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def rename(directory):
    dir_list = os.listdir(directory)
    dir_list.sort(key=natural_keys)
    i = 0
    for imgp in dir_list:
        label = imgp[-6:-4]
        os.rename(os.path.join(directory, imgp),
                  os.path.join(directory, str(i) + label + ".png"))
        i += 1


if __name__ == "__main__":
    # dir_to_read = os.listdir(FILE_TO_READ)
    # for imgp in dir_to_read:
    #     if "_0_" in imgp:
    #         copyfile(os.path.join(FILE_TO_READ, imgp), os.path.join(FILE_TO_SAVE_0, imgp))
    #     if "_1_" in imgp:
    #         copyfile(os.path.join(FILE_TO_READ, imgp), os.path.join(FILE_TO_SAVE_1, imgp))
    rename(FILE_TO_SAVE_0)
    rename(FILE_TO_SAVE_1)
