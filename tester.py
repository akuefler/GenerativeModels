####
#import os
#import numpy as np
#import scipy as sp
#import scipy.misc

#import skimage
#import skimage.transform

from PIL import Image
#from matplotlib import pyplot as plt
####

import getpass
import sys
import time

import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

from networks import RotNet, Solver

#####
#import os
#import numpy as np
#import scipy as sp
#import scipy.misc

#import skimage
#import skimage.transform

#from PIL import Image
#from matplotlib import pyplot as plt
#####

from manage_im_data import *

"""
X.shape= [batch, in_height, in_width, in_channels]
"""

#tt = TT_object()
#X1, X2 = tt.retrieve_example(170, 190)

tt = TT_object()
X1, X2, y = tt.retrieve_example(270, 295)

#with tf.device('/gpu:0'):
rn = RotNet()
feed= {rn.x1_input: X1[np.newaxis], rn.x2_input: X2[np.newaxis]}

if True:
    s = Solver(rn, lr= -1*1e-4)
    s.trainROT(X1[np.newaxis], X2[np.newaxis], y[np.newaxis], EPOCHS= 1000)

    with s.sess as sess:
        #sess.run(tf.initialize_all_variables())
        result = sess.run(rn.output_layer,
                          feed_dict= feed)
        
else:
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        result = sess.run(rn.output_layer,
                          feed_dict= feed)    
    

#G= rn.predict(feed)

print "Success"
plt.imshow(np.squeeze(result))

plt.show()

halt= True