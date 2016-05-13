import tensorflow as tf

def conv_keepdim(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

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

def img_normalize(x):
    return 255 * (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))

def updown_module(x, index, convolve, c_in, c_out, f_up= 2, f_down= 7, t= tf.nn.elu):
    w_convT0 = tf.get_variable("wT"+str(index), shape= [f_up, f_up, c_out, c_in],
                               initializer= tf.contrib.layers.xavier_initializer_conv2d())
    b_convT0 = tf.get_variable("bT"+str(index), shape= [c_out])
    h_convT0 = t(deconv(x, w_convT0, c_out) + b_convT0)

    if convolve:
        w_conv0 = tf.get_variable("w"+str(index), shape= [f_down,f_down, c_out, c_out],
                                  initializer= tf.contrib.layers.xavier_initializer_conv2d())
        b_conv0 = tf.get_variable("b"+str(index), shape= [c_out])        
        h_conv0 = t(conv_keepdim(h_convT0, w_conv0) + b_conv0)

        prev_layer = h_conv0
    else:
        prev_layer = h_convT0
        
    return prev_layer

#def cosine_loss(predicted, target):
    #batch_size = predicted.get_shape()[0].value
    
    #x= tf.reshape(predicted, [batch_size, -1])
    #y= tf.reshape(target, [batch_size, -1])
    
    ##WARNING: This will break for larger batch sizes!!
    #loss = 1/(tf.matmul(x, tf.transpose(y)) / (tf.nn.l2_loss(x) * tf.nn.l2_loss(y)))
    #return loss
    
def cosine_loss(predicted, target):
    batch_size = predicted.get_shape()[0].value
    
    x= tf.reshape(predicted, [batch_size, -1])
    y= tf.reshape(target, [batch_size, -1])
    
    #WARNING: This will break for larger batch sizes!!
    loss = (tf.matmul(x, tf.transpose(y)) / (tf.nn.l2_loss(x) * tf.nn.l2_loss(y)))
    return loss
    
#def cosine_loss(predicted, target):
    #batch_size = predicted.get_shape()[0].value
    
    #x= tf.reshape(predicted, [batch_size, -1])
    #y= tf.reshape(target, [batch_size, -1])
    
    ##WARNING: This will break for larger batch sizes!!
    #loss =  (1.0/(tf.matmul(x, tf.transpose(y)) / (tf.nn.l2_loss(x) * tf.nn.l2_loss(y)))) + tf.nn.l2_loss(x)
    
    ##print "x norm: ", 
    
    #return loss
