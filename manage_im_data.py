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

import cv2

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

def load_CIFAR(batch_num= 1):
    fo = open('./data/CIFAR/cifar-10-batches-py/data_batch_'+str(batch_num)+'', 'rb')
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
    def __init__(self, path = 'batch1/Bannanas/', angle= 'Bottom/', scale= 7, dataset= None):
        self.scale = scale
        self.ims = {}
        self.dataset= dataset
        root = './data/CALTECH/'
        
        for im in os.listdir(root+path+angle):
            try:
                key = int(im.split('_')[1].split('-')[1])
            except:
                continue
            
            if key < 360:
                self.ims[key] = root+path+angle+im
                #self.ims[key] = ndimage.imread(root+path+angle+im)
            
    def retrieve_img(self, degree, use_seg= True):
        if not use_seg:
            im1 = ndimage.imread(self.ims[degree])
            im2 = np.array(im1)
        else:
            f = self.ims[degree].replace('Bottom','Masks')
            path = ''.join(f.split('.')[:-1])
            im2 = np.load('.'+path+'.npz')['arr_0']
            
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
    
class CIFAR_Data_Loader(object):
    def __init__(self, max_data= 10000):
        if max_data is None:
            self.X_train = load_CIFAR(batch_num= 1)
            self.X_val = load_CIFAR(batch_num= 2)
            
        else:
            self.X_train = load_CIFAR(batch_num= 1)[:max_data]
            self.X_val = load_CIFAR(batch_num= 2)[:max_data]
        
    def iterate_minibatches(self, batch_size= 10, mode= 'train', shuffle= True):

        if mode == "train":
            X = self.X_train
        else:
            X = self.X_val
        
        n = len(X)
        remainder= n - (batch_size * (n / batch_size))        

        if shuffle:
            p = np.random.permutation(len(X))
            X = X[p]
            
        minibatches = np.split(np.array(X), (len(X))/batch_size)
            
        return minibatches
    
    def minibatch_2_Xy(self, minibatch):
        return minibatch
        
        
    
class Rot_Data_Loader(object):
    def __init__(self, objects= [('Bannanas','train'), ('Base','val')],\
                 interval = 2, scale= 7, max_train_data= None, max_val_data= None):
        self.objects= [TT_object(path= 'batch1'+'/'+item+'/', scale= scale, dataset= dataset)\
                       for item, dataset in objects]
        
        t_data = []
        v_data = []
        
        for obj in self.objects:
            combos= list(itertools.combinations(obj.ims.keys(), 2))
            for combo in combos:
                if abs(combo[0] - combo[1]) < 10:
                    #print "skipped!"
                    continue
                
                if obj.dataset == 'train':
                    t_data.append((obj,)+combo)
                elif obj.dataset == 'val':
                    v_data.append((obj,)+combo)
                else:
                    assert False
                        
        #self.train_data = [datapoint for i, datapoint in enumerate(t_data) if i % interval != 0]
        #self.val_data = [datapoint for i, datapoint in enumerate(v_data) if i % interval == 0]
        
        t_n = len(t_data)
        v_n = len(v_data)
        
        #np.random.seed(1)
        t_keep = np.random.choice(np.arange(0,t_n), t_n/2, replace= False)
        self.train_data = [datapoint for i, datapoint in enumerate(t_data) if i in t_keep]
        
        #np.random.seed(2)
        v_keep = np.random.choice(np.arange(0,v_n), v_n/2, replace= False)
        self.val_data = [datapoint for i, datapoint in enumerate(v_data) if i in v_keep]        
        
        if max_train_data is not None:
            self.train_data=self.train_data[0:max_train_data]
            
        if max_val_data is not None:
            self.val_data = self.val_data[0:max_val_data]
        
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
            
        if remainder != 0:
            d = d[0:-remainder]
            
        minibatches = np.split(np.array(d), (len(d))/batch_size)
            
        return minibatches

def segment(filename):
    """
    code adapted from: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
    """
    img = ndimage.imread(filename)
    #img = skimage.transform.resize(img, (64, 64))
    
    Z = img.reshape((-1,3))
    
    # convert to np.float32
    Z = np.float32(Z)
    
    # 	
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 7
    if filename.split('/')[4] in ["FlowerLamp","TeddyBear"]:
        K = 18
        
    ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    mask1= (res2[:,:,0] > 100).astype('int32') \
        * (res2[:,:,1] > 100).astype('int32') \
        * (res2[:,:,2] > 100).astype('int32')
    #mask2 = (res2.argmax(axis= 2) == 1).astype('int')
    
    mask = mask1
    mask= np.dstack((mask, mask, mask))
    
    #plt.imshow(img * mask)
    idx= (mask1 == 1)
    Y= img.copy()
    Y[idx] = 0
    
    return Y

def create_mask_data(path):
    print "CREATING MASKS"
    ldir= sorted(os.listdir(path+'/Bottom'))
    for f in ldir:
        if f.split('.')[-1] == 'JPG':
            if int(f.split('-')[1].split('_')[0]) > 360:
                continue
            
            try:
                seg = segment(path+'/Bottom/'+f)
                try:
                    np.savez(path+'/Masks/'+(''.join(f.split('.')[:-1])), seg)
                except IOError:
                    os.mkdir(path+'/Masks')
                    np.savez(path+'/Masks/'+(''.join(f.split('.')[:-1])), seg)
                    
            except ValueError:
                print "skipped: ", path+'/Bottom/'+f
         
    #halt= True
    
#for toy in ['FlowerLamp','TeddyBear']:
    #print toy
    #create_mask_data('./data/CALTECH/batch1/'+toy)
    
#halt= True
        
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