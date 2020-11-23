from __future__ import absolute_import
from __future__ import print_function

import numpy as np

import os
from keras.models import Model, load_model, Sequential
from sklearn.decomposition import PCA

import cv2
SLIDING_WINDOW_SIZE = 20

def get_intensity_value(value, min_val, max_val):
    if np.isnan(value):
        value = 0
    if np.isnan(max_val):
        max_val = 0.0001
    if np.isnan(min_val):
        min_val = 0
    nor_val=255 * ((value - min_val) / (max_val - min_val))
    if np.isnan(nor_val):
        nor_val=np.nan_to_num(nor_val)
    else:
        nor_val=int(nor_val)
    return nor_val



def predict(path_to_model,path_to_read,path_to_save,patch_size):
    outersize = 200
    # trimsize = 90
    trimsize = (patch_size - SLIDING_WINDOW_SIZE)/2
    model = load_model(path_to_model)
    output_layer = 2
    model = model.layers[output_layer]

    for imgp in os.listdir(path_to_read):
        print(imgp)
        page = cv2.imread('{}/{}'.format(path_to_read, imgp), 0)
        rows, cols = page.shape
        x = rows // SLIDING_WINDOW_SIZE
        y = cols // SLIDING_WINDOW_SIZE
        prows = (x + 1) * SLIDING_WINDOW_SIZE + 2 * trimsize
        pcols = (y + 1) * SLIDING_WINDOW_SIZE + 2 * trimsize
        ppage = np.zeros([prows, pcols])
        ppage[trimsize:rows + trimsize, trimsize:cols + trimsize] = page[:, :]
        predicted_patch = model.predict(np.zeros((1, outersize, outersize, 1)))
        predicted_img = np.zeros((x + 1, y + 1, predicted_patch.shape[1]), np.float32)

        for i in range(0, x + 1):
            for j in range(0, y + 1):
                patch = ppage[i * SLIDING_WINDOW_SIZE:i * SLIDING_WINDOW_SIZE + outersize,
                        j * SLIDING_WINDOW_SIZE:j * SLIDING_WINDOW_SIZE+ outersize]
                patch = np.expand_dims(patch, axis=0)
                predicted_patch = model.predict(patch)

                predicted_img[i, j] = predicted_patch


        # pca = PCA(n_components=predicted_img.shape[2])
        #
        # features = predicted_img.reshape(-1, predicted_img.shape[2])
        # pca_t_features = pca.fit(features).transform(features)
        # pca_t_features = pca_t_features[:, :3]
        #
        # rgb = [[get_intensity_value(pca_t_features[i, 0], pca_t_features[:, 0].min(),
        #                             pca_t_features[:, 0].max()),
        #         get_intensity_value(pca_t_features[i, 1], pca_t_features[:, 1].min(),
        #                             pca_t_features[:, 1].max()),
        #         get_intensity_value(pca_t_features[i, 2], pca_t_features[:, 2].min(),
        #                             pca_t_features[:, 2].max())]
        #        for i in range(pca_t_features.shape[0])]
        # black_white = [[get_intensity_value(pca_t_features[i, 0], pca_t_features[:, 0].min(),
        #                             pca_t_features[:, 0].max())]for i in range(pca_t_features.shape[0])]
        # rgb = np.asarray(rgb, dtype=np.uint8).reshape((*predicted_img.shape[:2], 3))
        # rgb_rows, rgb_cols, _ = rgb.shape
        # result = np.zeros([rows, cols])
        # for i in range(rgb_rows):
        #     for j in range(rgb_cols):
        #         pixel_value = rgb[i, j]
        #         result[i * SLIDING_WINDOW_SIZE:i *SLIDING_WINDOW_SIZE + SLIDING_WINDOW_SIZE,
        #         j * SLIDING_WINDOW_SIZE:j * SLIDING_WINDOW_SIZE + SLIDING_WINDOW_SIZE] = pixel_value
        #
        # # big_rgb=cv2.resize(rgb,(page.shape[1]-pad_h,page.shape[0]-pad_w))
        # # org_rgb=np.zeros([page.shape[0],page.shape[1],3])
        # # org_rgb[:-pad_w,:-pad_h]=big_rgb
        #
        #
        # cv2.imwrite('{}/{}'.format(os.path.join(path_to_save, imgp), imgp), result)