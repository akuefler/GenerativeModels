import os
import numpy as np
import scipy as sp
import scipy.misc
import scipy.ndimage as ndimage

import skimage
import skimage.transform

#from PIL import Image
from matplotlib import pyplot as plt

def prep_image(im, lw= 128):
    """
    From : https://github.com/ebenolson/Recipes/blob/master/examples/imagecaption/COCO%20Preprocessing.ipynb
    """
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)
    # Resize so smallest dim = 224, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (lw, w*lw/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*lw/w, lw), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-(lw/2):h//2+(lw/2), w//2-(lw/2):w//2+(lw/2)]
    
    rawim = np.copy(im).astype('uint8')
    
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # Convert to BGR
    #im = im[::-1, :, :]

    #im = im - MEAN_VALUES
    return rawim, (im[np.newaxis]).astype('float32')

class TT_object(object):
    def __init__(self, path = 'batch1/Bannanas/', angle= 'Bottom/', lw= 128):
        self.lw = lw
        self.ims = {}
        root = './data/CALTECH/'
        
        for im in os.listdir(root+path+angle):
            try:
                key = int(im.split('_')[1].split('-')[1])
            except:
                continue
            
            if key < 360:
                self.ims[key] = root+path+angle+im
                #self.ims[key] = ndimage.imread(root+path+angle+im)
            
    def retrieve_img(self, degree, downsample= 128):
        im1 = ndimage.imread(self.ims[degree])
        im2 = np.array(im1)
        im, cnn_im = prep_image(im2)
        #im = sp.misc.imresize(im, size= (downsample,downsample))
        return im
    
    def retrieve_example(self, degree1, degree2):
        assert degree1 < degree2
        
        x1 = self.retrieve_img(degree1)
        x2 = self.retrieve_img(degree2)
        y = self.retrieve_img(degree1 + int(round(((degree2 - degree1)/2)/5) * 5))
        
        return x1, x2, y
        
        
#tt = TT_object()
#x1, x2, y = tt.retrieve_example(270, 295)
#halt= True