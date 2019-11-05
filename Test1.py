# -*- coding: utf-8 -*-

from MyHybridImages import myHybridImages

import matplotlib.pyplot as plt
import matplotlib.image as maping


if __name__=='__main__':
    low_Image=maping.imread('dog.bmp')
    high_Image=maping.imread('cat.bmp')
    hybridImage=myHybridImages(low_Image,8,high_Image,4)
    print(hybridImage)
    plt.imshow(hybridImage)
    plt.show()
# zll_image=maping.imread('IMG_2197.jpeg')
# print(zll_image.shape)
