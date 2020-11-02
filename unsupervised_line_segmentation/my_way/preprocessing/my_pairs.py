import os
import numpy as np
import cv2
from typing import List, Tuple
import unsupervised_line_segmentation.my_way.preprocessing.pairs_testing as pair_test

MARGIN = 20


def get_position(img: np.ndarray, patch_size: int) -> Tuple:
    """Find random position for patches within acceptable location"""
    pos = [np.random.randint(low=0 + MARGIN, high=img.shape[1] - 2 * patch_size - MARGIN),
           np.random.randint(low=0 + MARGIN, high=img.shape[0] - patch_size - MARGIN)
           ]
    p1_pos = pos
    p2_pos = [pos[0] + patch_size, pos[1]]
    return p1_pos, p2_pos


def evaluate_s(p1: np.ndarray, p2: np.ndarray) -> float:
    """Based on two patches evaluate s value"""
    pixels1 = cv2.countNonZero(p1)
    pixels2 = cv2.countNonZero(p2)
    if pixels1 != 0 and pixels2 != 0:  # TODO to remove when in get_patches size is confirmed
        s = min(pixels1, pixels2) / max(pixels1, pixels2)
        return s


def get_patches(img: np.ndarray, patch_size: int) -> Tuple:
    """Generate patches from image"""
    p1_pos, p2_pos = get_position(img, patch_size)
    # TODO check why somethimes p1 or p2 can have (0,50) size which leads to mistake in evaluate_s
    p1 = img[p1_pos[0]:p1_pos[0] + patch_size, p1_pos[1]:p1_pos[1] + patch_size]
    p2 = img[p2_pos[0]:p2_pos[0] + patch_size, p2_pos[1]:p2_pos[1] + patch_size]
    return p1, p2


def get_patches_similar_by_number_of_foreground_pixels(img: np.ndarray,
                                                       patch_size: int):  # -> List:

    while True:
        # get random coords
        p1, p2 = get_patches(img=img, patch_size=patch_size)
        s = evaluate_s(p1, p2)
        # TODO based on s value label the patches and return them


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
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # there are 3 possibilites on patches generation.
    # Patches Similiar by number of froeground pixels (s > 0.7)
    # Patches difrent by number of foreground pixels (s < 0.4)
    # Patches different by background area (nb of white > nb of black pixels)

    # gen_func = np.random.choice([get_nearby_patches, get_backpaired_patches,
    #                              get_nearby_patches, get_different_area_patches])
    # p1, p2, label = gen_func(img, thresh)
    # get_patches_similar_by_number_of_foreground_pixels(img, 150)
    # for size in range(50,200,20):
    #     testing(img,size)
    # return p1, p2, label
