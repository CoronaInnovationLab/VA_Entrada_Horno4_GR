import cv2 as cv
import os

input_path = 'data/train_img/800x600'
out_path = 'data/train_img'

paths = os.listdir(input_path)

for image_path in paths:
    image = cv.imread(os.path.join(input_path,image_path))
    image = cv.resize(image, (768, 576), interpolation=cv.INTER_CUBIC)
    cv.imwrite(os.path.join(out_path,image_path), image)