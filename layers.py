import tensorflow as tf
import numpy as np

def conv_keepdim(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def multi_conv(x, num_conv, chan_in, chan_out, f, nonlin= tf.nn.elu):
    #chan_in = 3
    #chan_out= h_chan
    #nonlin = self.nonlin
    
    prev = x
    for i in range(num_conv):
            
        w_conv = tf.get_variable("w"+str(i), shape= [f,f,chan_in,chan_out],
                                  initializer= tf.contrib.layers.xavier_initializer_conv2d())
        b_conv = tf.get_variable("b"+str(i), shape= [chan_out])
        prev = nonlin(conv_keepdim(prev, w_conv) + b_conv)
        
        chan_in = chan_out
    return prev

def deconv(x, W, num_channels):
    
    #batch_size = x.get_shape()[0].value
    #h = (x.get_shape()[1] + W.get_shape()[0] - 1).value
    #w = (x.get_shape()[2] + W.get_shape()[1] - 1).value
    #output_shape = (batch_size, h, w, num_channels)
    
    #stride: 2, pad: 0
    batch_size = x.get_shape()[0].value
    h = (W.get_shape()[0].value + 2*((x.get_shape()[1] - 1).value))
    w = (W.get_shape()[1].value + 2*((x.get_shape()[2] - 1).value))
    
    output_shape = (batch_size, h, w, num_channels)
    
    return tf.nn.conv2d_transpose(x, W,
                       output_shape,
                       strides=[1, 2, 2, 1],
                       padding= 'VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                          strides=[1, 4, 4, 1], padding='SAME')

def img_clamp(x):
    return tf.maximum(tf.minimum(x, 255), 0)

def img_normalize(x, top= 255):
    return top * (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))

def gather_indices(x, idx):
    """
    Differentiable version of gather_nd from: https://github.com/tensorflow/tensorflow/issues/206
    """
    #x = tf.constant([[1, 2, 3],
                     #[4, 5, 6],
                     #[7, 8, 9]])
    #idx = tf.constant([1, 0, 2])
    #idx_flattened = tf.range(0, x.get_shape()[0]) * x.get_shape()[1] + idx
    #y = tf.gather(tf.reshape(x, [-1]),  # flatten input
                  #idx_flattened)
    
    idx_flattened = tf.range(0, x.get_shape()[0]) * x.get_shape()[1] + idx
    y = tf.gather(tf.reshape(x, [-1]),  # flatten input
                  idx_flattened)  # use flattened indices
    return y    

def updown_module(x, index, convolve, c_up, c_down, up= 'deconv', f_up= 2, f_down= 7, t= tf.nn.elu):
    [batch_size, H, W, c_in] = x.get_shape().as_list()
    
    if up == 'deconv':
        w_convT0 = tf.get_variable("wT"+str(index), shape= [f_up, f_up, c_up, c_in],
                                   initializer= tf.contrib.layers.xavier_initializer_conv2d())
        b_convT0 = tf.get_variable("bT"+str(index), shape= [c_up])
        h_convT0 = t(deconv(x, w_convT0, c_up) + b_convT0)
        
        conv_c_in = c_up
        
    elif up == 'resize':
        conv_c_in = c_in
        
        H2 = 2 ** (np.log2(H) + 1)
        W2 = 2 ** (np.log2(W) + 1)
        
        h_convT0 = tf.image.resize_images(x, H2, W2)

    if convolve:
        w_conv0 = tf.get_variable("w"+str(index), shape= [f_down,f_down, conv_c_in, c_down],
                                  initializer= tf.contrib.layers.xavier_initializer_conv2d())
        b_conv0 = tf.get_variable("b"+str(index), shape= [c_down])        
        h_conv0 = t(conv_keepdim(h_convT0, w_conv0) + b_conv0)

        prev_layer = h_conv0
    else:
        prev_layer = h_convT0
        
    return prev_layer

def MSE_loss(predicted, target):
    batch_size = predicted.get_shape()[0].value
    
    x= tf.reshape(predicted, [batch_size, -1])
    y= tf.reshape(target, [batch_size, -1])
    
    return tf.reduce_mean(tf.sqrt(tf.matmul((x-y), tf.transpose(x-y))))

def neg_cosine_loss(predicted, target):
    batch_size = predicted.get_shape()[0].value
    
    x= tf.reshape(predicted, [batch_size, -1])
    y= tf.reshape(target, [batch_size, -1])
    
    #WARNING: This will break for larger batch sizes!!
    loss = -(tf.matmul(x, tf.transpose(y)) / (tf.nn.l2_loss(x) * tf.nn.l2_loss(y)))
    return loss

def cosine_loss_old(predicted, target):
    batch_size = predicted.get_shape()[0].value
    
    x= tf.reshape(predicted, [batch_size, -1])
    y= tf.reshape(target, [batch_size, -1])
    
    #WARNING: This will break for larger batch sizes!!
    #loss = tf.reduce_sum(1.0/(tf.matmul(x, tf.transpose(y)) / (tf.nn.l2_loss(x) * tf.nn.l2_loss(y))))
    
    #loss = tf.reduce_sum(1.0/(tf.matmul(x, tf.transpose(y)) / (tf.nn.l2_loss(x) * tf.nn.l2_loss(y))))
    loss = tf.reduce_sum((tf.nn.l2_loss(x) * tf.nn.l2_loss(y))/(tf.matmul(x, tf.transpose(y))))
    return loss
    
def cosine_loss(predicted, target):
    batch_size = predicted.get_shape()[0].value
    
    x= tf.reshape(predicted, [batch_size, -1])
    y= tf.reshape(target, [batch_size, -1])
    
    num = tf.reduce_sum(x*y, 1)
    
    #norm_x = tf.sqrt(tf.reduce_sum(tf.square(x),1))
    #norm_y = tf.sqrt(tf.reduce_sum(tf.square(y),1))
    
    norm_x = tf.reduce_sum(x ** 2,1) / 2
    norm_y = tf.reduce_sum(y ** 2,1) / 2
    
    return tf.reduce_sum( ((norm_x * norm_y)+1)/ num+2 )

def combined_loss(predicted, target):
    batch_size = predicted.get_shape()[0].value
    
    x= tf.reshape(predicted, [batch_size, -1])
    y= tf.reshape(target, [batch_size, -1])
    
    #WARNING: This will break for larger batch sizes!!
    loss = tf.sqrt(tf.nn.l2_loss(predicted-target))\
        + tf.reduce_sum(1.0/(tf.matmul(x, tf.transpose(y)) / (tf.nn.l2_loss(x) * tf.nn.l2_loss(y))))
    return loss

def cosine_retain_norm_loss(predicted, target):
    batch_size = predicted.get_shape()[0].value
    
    x= tf.reshape(predicted, [batch_size, -1])
    y= tf.reshape(target, [batch_size, -1])
    
    #WARNING: This will break for larger batch sizes!!
    loss = tf.reduce_sum( tf.abs(tf.sqrt(tf.nn.l2_loss(x)) - tf.sqrt(tf.nn.l2_loss(y))) + \
                          1.0/(tf.matmul(x, tf.transpose(y)) / (tf.nn.l2_loss(x) * tf.nn.l2_loss(y))) )
    return loss

#def dot_loss(predicted, target):
    #batch_size = predicted.get_shape()[0].value
    
    #x= tf.nn.l2_normalize(tf.reshape(predicted, [batch_size, -1]), 1) 
    #y= tf.nn.l2_normalize(tf.reshape(target, [batch_size, -1]), 1)
    
    ##WARNING: This will break for larger batch sizes!!
    #loss = tf.reduce_sum( 1.0/tf.matmul(x, tf.transpose(y)) )
    #return loss

def dot_loss(predicted, target):
    batch_size = predicted.get_shape()[0].value
    
    #x= tf.nn.l2_normalize(tf.reshape(predicted, [batch_size, -1]), 1) 
    #y= tf.nn.l2_normalize(tf.reshape(target, [batch_size, -1]), 1)
    
    x= tf.reshape(predicted, [batch_size, -1])
    y= tf.reshape(target, [batch_size, -1])  
    
    #WARNING: This will break for larger batch sizes!!
    loss = tf.reduce_sum( 1.0/(tf.abs(tf.matmul(x, tf.transpose(y))) + 1e-5 ) )
    return loss
    
#def cosine_loss(predicted, target):
    #batch_size = predicted.get_shape()[0].value
    
    #x= tf.reshape(predicted, [batch_size, -1])
    #y= tf.reshape(target, [batch_size, -1])
    
    ##WARNING: This will break for larger batch sizes!!
    #loss = (tf.matmul(x, tf.transpose(y)) / (tf.nn.l2_loss(x) * tf.nn.l2_loss(y)))
    #return loss
    
#def cosine_loss(predicted, target):
    #batch_size = predicted.get_shape()[0].value
    
    #x= tf.reshape(predicted, [batch_size, -1])
    #y= tf.reshape(target, [batch_size, -1])
    
    ##WARNING: This will break for larger batch sizes!!
    #loss =  (1.0/(tf.matmul(x, tf.transpose(y)) / (tf.nn.l2_loss(x) * tf.nn.l2_loss(y)))) + tf.nn.l2_loss(x)
    
    ##print "x norm: ", 
    
    #return loss
