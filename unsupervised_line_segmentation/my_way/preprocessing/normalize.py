import os
import cv2
import splitfolders
PATH_TO_IMAGES = os.path.join(os.path.dirname(os.getcwd()),'cutted_pages')
PATH_TO_SAVE = os.path.join(os.path.dirname(os.getcwd()),'normalized')

PATH_TO_IMAGES = r'F:\autocad\mapy\gornicze\JPG'
PATH_TO_SAVE = r'F:\autocad\mapy\gornicze\BINARY'



if __name__ == "__main__":

        img = cv2.imread('{}/{}'.format(PATH_TO_IMAGES, 'banhof_Katowitz.jpg'), 0)
        ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # th2 = cv2.normalize(th2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
        #                            dtype=cv2.CV_32F)
        cv2.imwrite(os.path.join(PATH_TO_SAVE, 'banhof_Katowitz.jpg'),th2)


