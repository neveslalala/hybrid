# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as maping

from MyConvolution import convolve


def myHybridImages(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma: float) -> np.ndarray:
    """
    Create hybrid images by combining a low-pass and high-pass filtered pair.
    
    :param lowImage: the image to low-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray
    
    :param lowSigma: the standard deviation of the Gaussian used for low-pass filtering lowImage
    :type float
    
    :param highImage: the image to high-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray
    
    :param highSigma: the standard deviation of the Gaussian used for low-pass filtering highImage before subtraction to create the high-pass filtered image
    :type float 
    
    :returns returns the hybrid image created
           by low-pass filtering lowImage with a Gaussian of s.d. lowSigma and combining it with 
           a high-pass image created by subtracting highImage from highImage convolved with
           a Gaussian of s.d. highSigma. The resultant image has the same size as the input images.
    :rtype numpy.ndarray
    """

    # Your code here.
    lowFilteredImage = convolve(lowImage, makeGaussianKernel(lowSigma))
    print("the picture should be below")
    plt.imshow(lowFilteredImage)
    #plt.show()
    print("the picture should be upper")
    
    highFilteredImage = highImage - convolve(highImage, makeGaussianKernel(highSigma)
    plt.imshow(highFilteredImage)
    plt.show()
    hybridImage = lowFilteredImage + highFilteredImage
    #print(lowFilteredImage)
    #print(highFilteredImage)
    #print(hybridImage)
    return hybridImage

def makeGaussianKernel(sigma: float) -> np.ndarray:
    """
    Use this function to create a 2D gaussian kernel with standard deviation sigma.
    The kernel values should sum to 1.0, and the size should be floor(8*sigma+1) or 
    floor(8*sigma+1)+1 (whichever is odd) as per the assignment specification.
    """

    # Your code here.
    kernel_size = 8*sigma+1
    kernel = np.zeros([kernel_size,kernel_size], dtype=float)
    center = kernel_size//2
    
    
    s = 2*(sigma**2)
    sum_val = 0
    for i in range(0,kernel_size):
        for j in range(0,kernel_size):
            x = i-center
            y = j-center
            kernel[i,j] = np.exp(-(x**2+y**2) / s)
            sum_val += kernel[i,j]
            #/(np.pi * s)
    sum_val = 1/sum_val
    print("here is the kernel", kernel*sum_val)
    return kernel*sum_val
    
    
   


