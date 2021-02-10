import cv2 as cv
import numpy as np
import time


print(1)

def sepia(src_image):
    gray = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY)
    normalized_gray = np.array(gray, np.float32) / 255
    # solid color
    sepia = np.ones(src_image.shape)
    sepia[:, :, 0] *= 153  # B
    sepia[:, :, 1] *= 204  # G
    sepia[:, :, 2] *= 255  # R
    # hadamard
    sepia[:, :, 0] *= normalized_gray  # B
    sepia[:, :, 1] *= normalized_gray  # G
    sepia[:, :, 2] *= normalized_gray  # R
    return np.array(sepia, np.uint8)


# номер 1
img = cv.imread('Thun-castle-long.jpg', cv.IMREAD_GRAYSCALE)
# print(img)
# cv.imshow('', img)
# cv.waitKey(0)

# номер 2
x, y = len(img), len(img[0])
print(x, 'пикселей в ширину,', y, 'пикселей в высоту')
cropped = img[::, 0:y // 2]
# cv.imshow('', cropped)
# cv.waitKey(0)
center = (y // 2, x // 2)
# M = cv.getRotationMatrix2D(center, 180, 1.0)
# rotated = cv.warpAffine(img, M, (y, x))
# cv.imshow('', rotated)
# cv.waitKey(0)
# image = cv.imread('Thun-castle-long.jpg')
# image2 = sepia(image)
# cv.imshow('', image2)
# cv.waitKey()


for i in range(1, 43):
    name = str(i) + '.jpg'
    img = cv.imread(name, cv.IMREAD_GRAYSCALE)
    x, y = len(img), len(img[0])
    print(x, x // 5)
    cv.imshow('', img)
    cv.waitKey()
