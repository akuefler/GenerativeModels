import os
import numpy as np
import scipy as sp
import scipy.misc
import scipy.ndimage as ndimage

import skimage
import skimage.transform

import cPickle

import itertools
import copy
import random

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

def load_CIFAR():
    fo = open('./data/CIFAR/cifar-10-batches-py/data_batch_1', 'rb')
    dic = cPickle.load(fo)
    fo.close()

    X = dic['data']
    X = np.swapaxes(X.reshape((10000, 32, 32, 3), order='F'),1,2)
    return X

def im2vf(X):
    """
    channel 0: Angle
    channel 1: Magnitude
    """
    if X.shape[-1] == 3:
        X = np.delete(X, -1, 3)
    
    X = X / 255.0
    return X

def vf2im(X):
    """
    """
    X = np.repeat(X, (1,2), axis=3) #Copy a channel.
    
    X = X * 255
    return X

class TT_object(object):
    def __init__(self, path = 'batch1/Bannanas/', angle= 'Bottom/', scale= 7):
        self.scale = scale
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
            
    def retrieve_img(self, degree):
        im1 = ndimage.imread(self.ims[degree])
        im2 = np.array(im1)
        im, cnn_im = prep_image(im2, lw= 2 ** self.scale)
        #im = sp.misc.imresize(im, size= (downsample,downsample))
        return im
    
    def retrieve_example(self, degree1, degree2):
        if degree1 > degree2:
            temp = degree2
            degree2 = degree1
            degree1 = temp
        
        x1 = self.retrieve_img(degree1)
        x2 = self.retrieve_img(degree2)
        y = self.retrieve_img(degree1 + int(round((abs(degree2 - degree1)/2)/5) * 5))
        
        return x1, x2, y
    
class Rot_Data_Loader(object):
    def __init__(self, objects= [('batch1', 'Bannanas'), ('batch1', 'Base')],\
                 interval = 2, scale= 7):
        self.objects= [TT_object(path= folder+'/'+item+'/', scale= scale)\
                       for folder, item in objects]
        
        data = []
        
        for obj in self.objects:
            combos= list(itertools.combinations(obj.ims.keys(), 2))
            for combo in combos:
                data.append((obj,)+combo)
                
        self.train_data = [datapoint for i, datapoint in enumerate(data) if i % interval != 0]
        self.val_data = [datapoint for i, datapoint in enumerate(data) if i % interval == 0]
        
    def minibatch_2_Xy(self, minibatch):
        X1= []
        X2= []
        Y = []
        for datapoint in minibatch:
            x1, x2, y = datapoint[0].retrieve_example(datapoint[1], datapoint[2])
            X1.append(x1)
            X2.append(x2)
            Y.append(y)
            
        X1_b= np.stack(X1, axis= 0)
        X2_b= np.stack(X2, axis= 0)
        Y_b = np.stack(Y , axis= 0)
                   
        return X1_b, X2_b, Y_b
    
    def iterate_minibatches(self, batch_size= 10, mode= 'train', shuffle= True):

        
        if mode == "train":
            d = copy.copy(self.train_data)
        else:
            d = copy.copy(self.val_data)
        
        n = len(d)
        remainder= n - (batch_size * (n / batch_size))        

        if shuffle:
            random.shuffle(d)
            
        d = d[0:-remainder]
            
        minibatches = np.split(np.array(d), (len(d))/batch_size)
            
        return minibatches

halt= True

        
#tt = TT_object()
#x1, x2, y = tt.retrieve_example(270, 295)
#halt= True

#X = load_CIFAR()

#x = X[240]
##plt.imshow(x)

##plt.show()

#V = im2vf(X)
#X = vf2im(V)

#x = X[1000]
#plt.imshow(x)

#plt.show()

#halt = True