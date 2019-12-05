import os
import cv2
import numpy as np


LABEL_COLOR = [(0, 0, 0)
               # 0=background
               , (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128)
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               , (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0)
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               , (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128)
               # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
               , (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor


def convert_color(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    new_image = np.zeros((image.shape[0], image.shape[1], 3))
    for i in range(20):
        new_image[np.where(image == i)] = LABEL_COLOR[i]
    return new_image[:, :, ::-1]


if __name__ == '__main__':

    # Image List
    image_path = 'datalist/PascalVOC/val_id.txt'
    with open(image_path, 'r') as f:
        data = f.readlines()
    image_list = [name.strip() for name in data]
    final = []
    for img in image_list[4:8]:
        # Path for original image
        origin_path = '/srv/PascalVOC/VOCdevkit/VOC2012/JPEGImages/'
        # Path for your predicted image
        pred_path = 'train_log/caffe_trained/pred_images/'
        # Path for Colorized GT image (if you don't have this folder in your dataset, plz contact TAs)
        gt_path = '/srv/PascalVOC/VOCdevkit/VOC2012/SegmentationClass/'

        origin = os.path.join(origin_path, img+'.jpg')
        origin = cv2.imread(origin, cv2.IMREAD_COLOR)
        origin = cv2.resize(origin, (300, 300))

        pred = os.path.join(pred_path, img+'.png')
        new_pred = convert_color(pred)
        new_pred = cv2.resize(new_pred, (300, 300))

        gt = os.path.join(gt_path, img+'.png')
        gt = cv2.imread(gt, cv2.IMREAD_COLOR)
        gt = cv2.resize(gt, (300, 300))

        bar_h = np.ones((gt.shape[0], 4, 3)) * 255

        final.append(np.concatenate((origin, bar_h, new_pred, bar_h, gt), axis=1))
    final = np.concatenate(final, axis=0)
    cv2.imwrite(os.path.join('train_log/result_image/', 'result2.png'), final)

