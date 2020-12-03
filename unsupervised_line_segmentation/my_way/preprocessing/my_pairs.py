import os
import numpy as np
import cv2
from typing import List, Tuple
import itertools
import time

MARGIN = 20
TIMEOUT = 5


def get_position(img: np.ndarray, patch_size: int) -> Tuple:
    """Find random position for patches within acceptable location"""
    assert patch_size * 2 < img.shape[
        0], "Patch size to big, img vertical size is {}, while proposed patch {}. Reuce patch size".format(
        img.shape[0], patch_size)
    assert patch_size < img.shape[1], "Width of patch is to big"
    pos = [np.random.randint(low=0 + MARGIN, high=img.shape[0] - 2 * patch_size - MARGIN),
           np.random.randint(low=0 + MARGIN, high=img.shape[1] - patch_size - MARGIN)
           ]
    p1_pos = pos
    p2_pos = [pos[0] + patch_size, pos[1]]
    return p1_pos, p2_pos


def evaluate_s(p1: np.ndarray, p2: np.ndarray) -> float:
    """Based on two patches evaluate s value"""
    _, p_th1 = cv2.threshold(p1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, p_th2 = cv2.threshold(p2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pixels1 = cv2.countNonZero(p_th1)
    pixels2 = cv2.countNonZero(p_th2)
    if pixels1 != 0 or pixels2 != 0:
        s = min(pixels1, pixels2) / max(pixels1, pixels2)
        return s
    else:
        return 1


def get_patches(img: np.ndarray, patch_size: int) -> Tuple:
    """Generate patches from image"""
    p1_pos, p2_pos = get_position(img, patch_size)
    p1 = img[p1_pos[0]:p1_pos[0] + patch_size, p1_pos[1]:p1_pos[1] + patch_size]
    p2 = img[p2_pos[0]:p2_pos[0] + patch_size, p2_pos[1]:p2_pos[1] + patch_size]
    return p1, p2


def get_patches_similar_by_number_of_foreground_pixels(img: np.ndarray,
                                                       patch_size: int) -> Tuple:
    for _ in itertools.count():
        p1, p2 = get_patches(img=img, patch_size=patch_size)
        s = evaluate_s(p1, p2)
        if s >= 0.99:  # This might have to be improved
            label = 0
            return p1, p2, label
        else:
            continue


def get_patches_different_by_number_of_foreground_pixels(img: np.ndarray,
                                                         patch_size: int) -> Tuple:
    for _ in itertools.count():
        p1, p2 = get_patches(img=img, patch_size=patch_size)
        s = evaluate_s(p1, p2)
        if s < 0.99:  # This might have to be improved
            label = 1
            return p1, p2, label
        else:
            continue


def get_patches_different_by_background_area(img: np.ndarray,
                                             patch_size: int) -> Tuple:
    margin = 30
    start = time.time()
    for _ in itertools.count():
        if time.time() - start >= TIMEOUT:  # in case there is lack of white space in the image, after long time switch to different function
            p1, p2, label = get_patches_different_by_number_of_foreground_pixels(img, patch_size)
            return p1, p2, label
        p1, p2 = get_patches(img=img, patch_size=patch_size)
        numPixelsPatch = patch_size ** 2
        _, p1_th = cv2.threshold(p1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, p2_th = cv2.threshold(p2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if cv2.countNonZero(p1_th) > numPixelsPatch - margin or cv2.countNonZero(
                p2_th) > numPixelsPatch - margin:
            label = 1
            return p1, p2, label
        else:
            continue


def get_s_list(img: np.ndarray, patch_size: int, nb_of_patches: int) -> List:
    s_list = []
    for _ in range(1, nb_of_patches):
        p1, p2 = get_patches(img=img, patch_size=patch_size)
        s_list.append(evaluate_s(p1, p2))
    return s_list


def get_random_pair(images_path, patch_size):
    images = os.listdir(images_path)
    image_name = np.random.choice(images)
    img = cv2.imread(os.path.join(images_path, image_name), 0)

    # there are 3 possibilites on patches generation.
    # Patches Similiar by number of froeground pixels (s > 0.99)
    # Patches difrent by number of foreground pixels (s < 0.99)
    # Patches different by background area (nb of white > nb of black pixels)

    gen_func = np.random.choice([get_patches_similar_by_number_of_foreground_pixels,
                                 get_patches_similar_by_number_of_foreground_pixels,
                                 get_patches_different_by_number_of_foreground_pixels,
                                 get_patches_different_by_background_area])
    p1, p2, label = gen_func(img, patch_size)

    return p1, p2, label


def unsupervised_loaddata(folderName, set_size, patch_size):
    pairs = []
    labels = []
    percent = set_size / 100
    for i in range(set_size):
        # if i % percent  == 0:
        #     print('Set finished in {}%'.format(100 * i / set_size))
        p1, p2, label = get_random_pair(folderName, patch_size)
        p1 = p1.reshape(p1.shape[0], p1.shape[1], 1)
        p2 = p2.reshape(p2.shape[0], p2.shape[1], 1)
        pairs += [[p1, p2]]
        labels += [label]
    apairs = np.array(pairs, dtype=object)
    print(apairs.shape)
    alabels = np.array(labels)
    return apairs, alabels


import splitfolders
import os


def make_dirs(train_path: str, val_path: str) -> None:
    if os.path.isdir(train_path) is False:
        os.mkdir(train_path)
    if os.path.isdir(val_path) is False:
        os.mkdir(val_path)
    if os.path.isdir(os.path.join(val_path, "val_0")) is False:
        os.mkdir(os.path.join(val_path, "val_0"))
    if os.path.isdir(os.path.join(val_path, "val_1")) is False:
        os.mkdir(os.path.join(val_path, "val_1"))
    if os.path.isdir(os.path.join(train_path, "train_0")) is False:
        os.mkdir(os.path.join(train_path, "train_0"))
    if os.path.isdir(os.path.join(train_path, "train_1")) is False:
        os.mkdir(os.path.join(train_path, "train_1"))


def prepare_dataset(dataset_path_train: str,dataset_path_val: str, path_to_output: str, train_set_size: int,val_set_size:int, patch_size: int):
    """Function to prepare dataset for future generator use.
    :argument
        dataset_path_train (str) - path to folder with images for train patches
        dataset_path_val (str) - path to folder wth images for train patches
        path_to_output (str) - path to folder where patches from train and val sets will be stored
        train_set_size (int) - number of pairs of patches to be generated from images in train_folder
        val_set_size (int) - number of pairs of patches to be generated from images in val_folder
        patch_size (int) - size of patch to be generated
        """
    # 1 preparing path for train and validation dataset
    train_path = os.path.join(path_to_output, "train")
    val_path = os.path.join(path_to_output, "val")
    make_dirs(train_path, val_path)
    # 2 preparing train_set
    percent = int((train_set_size / 100))
    for i in range(train_set_size):
        if i % percent == 0:
            print('Train Set finished in {}%'.format(100 * i / train_set_size))
        p1, p2, label = get_random_pair(dataset_path_train, patch_size)
        p1 = p1.reshape(p1.shape[0], p1.shape[1], 1)
        p2 = p2.reshape(p2.shape[0], p2.shape[1], 1)
        cv2.imwrite(
            os.path.join(os.path.join(train_path, "train_0"), "{}_{}.png".format(i, label)), p1)
        cv2.imwrite(
            os.path.join(os.path.join(train_path, "train_1"), "{}_{}.png".format(i, label)), p2)
    # 3 preparing train_set
    percent = int(val_set_size / 100)
    for i in range(val_set_size):
        if i % percent == 0:
            print('Val Set finished in {}%'.format(100 * i / val_set_size))
        p1, p2, label = get_random_pair(dataset_path_val, patch_size)
        p1 = p1.reshape(p1.shape[0], p1.shape[1], 1)
        p2 = p2.reshape(p2.shape[0], p2.shape[1], 1)
        cv2.imwrite(os.path.join(os.path.join(val_path, "val_0"), "{}_{}.png".format(i, label)),
                    p1)
        cv2.imwrite(os.path.join(os.path.join(val_path, "val_1"), "{}_{}.png".format(i, label)),
                    p2)
