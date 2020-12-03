import numpy as np
import cv2
import os
# from keras.models import load_model
# from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from keras import backend as K
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_intensity_value(value, min_val, max_val):
    if np.isnan(value):
        value = 0
    if np.isnan(max_val):
        max_val = 0.0001
    if np.isnan(min_val):
        min_val = 0
    nor_val = 255 * ((value - min_val) / (max_val - min_val))
    if np.isnan(nor_val):
        nor_val = np.nan_to_num(nor_val)
    else:
        nor_val = int(nor_val)
    return nor_val


def pca():
    outersize = 128
    trimsize = 20
    innersize = outersize - 2 * trimsize
    model = load_model(r'F:\Studia\pythonProject\unsupervised_line_segmentation\my_way\learned_model_grayscale\bestmodel.h5py')
    # pages = 'complex_test'
    predict_folder = os.path.join(os.getcwd(),"test_folder","output",str(trimsize))
    test_folder = os.path.join(os.getcwd(),"test_folder","input")
    output_layer = 2
    model = model.layers[output_layer]

    os.makedirs(predict_folder, exist_ok=True)
    os.makedirs(os.path.join(predict_folder, 'cv2_vis1'), exist_ok=True)
    os.makedirs(os.path.join(predict_folder, 'cv2_vis2'), exist_ok=True)

    for imgp in os.listdir(test_folder):
        print(imgp)
        page = cv2.imread('{}/{}'.format(test_folder, imgp), 0)
        rows, cols = page.shape
        x = rows // innersize
        y = cols // innersize
        prows = (x + 1) * innersize + 2 * trimsize
        pcols = (y + 1) * innersize + 2 * trimsize
        ppage = np.zeros([prows, pcols])
        ppage[trimsize:rows + trimsize, trimsize:cols + trimsize] = page[:, :]
        predicted_patch = model.predict(np.zeros((1, outersize, outersize, 1)))
        predicted_img = np.zeros((x + 1, y + 1, predicted_patch.shape[1]), np.float32)

        for i in range(0, x + 1):
            for j in range(0, y + 1):
                patch = ppage[i * innersize:i * innersize + outersize,
                        j * innersize:j * innersize + outersize]
                patch = np.expand_dims(patch, axis=0)
                patch = np.expand_dims(patch, axis=3)
                predicted_patch = model.predict(patch)[0]

                predicted_img[i, j, :] = predicted_patch

        pca = PCA(n_components=predicted_img.shape[2])

        features = predicted_img.reshape(-1, predicted_img.shape[2])
        pca_t_features = pca.fit_transform(features)
        pca_t_features = pca_t_features[:, :3]

        rgb = [[get_intensity_value(pca_t_features[i, 0], pca_t_features[:, 0].min(),
                                    pca_t_features[:, 0].max()),
                get_intensity_value(pca_t_features[i, 1], pca_t_features[:, 1].min(),
                                    pca_t_features[:, 1].max()),
                get_intensity_value(pca_t_features[i, 2], pca_t_features[:, 2].min(),
                                    pca_t_features[:, 2].max())]
               for i in range(pca_t_features.shape[0])]

        rgb = np.asarray(rgb, dtype=np.uint8).reshape((*predicted_img.shape[:2], 3))
        rgb_rows, rgb_cols, _ = rgb.shape
        result = np.zeros([rows, cols, 3])
        for i in range(rgb_rows):
            for j in range(rgb_cols):
                pixel_value = rgb[i, j, :]
                result[i * innersize:i * innersize + innersize,
                j * innersize:j * innersize + innersize, :] = pixel_value

        # big_rgb=cv2.resize(rgb,(page.shape[1]-pad_h,page.shape[0]-pad_w))
        # org_rgb=np.zeros([page.shape[0],page.shape[1],3])
        # org_rgb[:-pad_w,:-pad_h]=big_rgb

        cv2.imwrite('{}/{}'.format(os.path.join(predict_folder, 'cv2_vis1'), imgp), rgb)
        cv2.imwrite('{}/{}'.format(os.path.join(predict_folder, 'cv2_vis2'), imgp), result)

def post_production():
    maintextfolder = 'complex_maintext_4'
    predfolder = 'complex_ptest'
    labelfolder = 'complex_ltest'

    for imgname in os.listdir(maintextfolder):
        mask = cv2.imread(os.path.join(maintextfolder, imgname), 0)
        imglabelname = imgname[:-4] + '.bmp'
        label = cv2.imread(os.path.join(labelfolder, imglabelname), 0)

        # get the contours
        ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)

        # find the biggest countour by area
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        x = x + 30
        y = y + 30
        w = w - 30
        h = h - 30

        # draw the biggest contour
        # cv2.rectangle(label,(x,y),(x+w,y+h),(0,255,0),2)

        # save the label
        # cv2.imwrite(os.path.join(predfolder,imgname),label)

        pred = label.copy()
        # get the maintext components
        mask = np.zeros(mask.shape)
        mask[y:y + h, x:x + w] = 255
        m = (mask == 255) & (pred < 255)
        pred[m] = 0
        # get the sidetext components
        mask = np.zeros(mask.shape)
        mask[:, 0:x] = 255
        mask[:, x + w:] = 255
        mask[:y, :] = 255
        mask[y + h:, :] = 255
        s = (mask == 255) & (pred < 255)
        pred[s] = 128

        # save the images
        cv2.imwrite(os.path.join(predfolder, imgname), pred)


if __name__ == "__main__":
    # pca()
    path = r"F:\Studia\pythonProject\unsupervised_line_segmentation\my_way\test_folder\output\54\cv2_vis2\a01-014u.png"
    I = cv2.imread(path,0)
    # cv2.imshow("test",I)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    ret, thresh = cv2.threshold(I, 84, 255, cv2.THRESH_BINARY_INV )
    thresh = cv2.resize(thresh,(int(thresh.shape[1]/2),int(thresh.shape[0]/2)))
    cv2.imshow("test",thresh)
    cv2.waitKey()
    cv2.destroyAllWindows()