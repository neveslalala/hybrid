import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as maping

def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    
    print("original image shape", image.shape)
    #print(image)
    
    h = kernel.shape[0]//2
    w = kernel.shape[1]//2
        # opencv filter2d 默认采用reflect填充方式
        # 前面测试了constant edge都不符合
    image_b = np.pad(image[:, :, 0], ((h, h), (w, w)), 'constant',constant_values = (0,0))
    image_g = np.pad(image[:, :, 1], ((h, h), (w, w)), 'constant',constant_values = (0,0))
    image_r = np.pad(image[:, :, 2], ((h, h), (w, w)), 'constant',constant_values = (0,0))
    image = np.dstack([image_b, image_g, image_r])
    #plt.imshow(image)
    #plt.show()
    
    print(image.shape)
    print(kernel.shape)
    
    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]
    
    conv_h = image.shape[0]-kernel.shape[0]+1
    conv_w = image.shape[1]-kernel.shape[1]+1
    
    conv_b = np.zeros((conv_h,conv_w),dtype = 'uint8')
    conv_g = np.zeros((conv_h,conv_w),dtype = 'uint8')
    conv_r = np.zeros((conv_h,conv_w),dtype = 'uint8')
    
    for i in range(conv_h):
         for j in range(conv_w):                 
             conv_b[i, j] = np.sum(image[i:i + kernel_h,j:j + kernel_w, 0 ]*kernel)
    
    for i in range(conv_h):
         for j in range(conv_w):                 
             conv_g[i, j] = np.sum(image[i:i + kernel_h,j:j + kernel_w, 1 ]*kernel)
        
    for i in range(conv_h):
         for j in range(conv_w):                 
             conv_r[i, j] = np.sum(image[i:i + kernel_h,j:j + kernel_w, 2 ]*kernel)

    
    dstack = np.dstack([conv_b, conv_g, conv_r])
    #print("chaged image shape", dstack.shape)
    #print(dstack)
    plt.imshow(dstack)
    plt.show()
    #plt.imshow(conv_b)
    #plt.show()
    return dstack