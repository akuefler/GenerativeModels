import tensorflow as tf
import numpy as np

def conv_keepdim(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def multi_conv(x, num_conv, chan_in, chan_out, f, dropout= 1.0, nonlin= tf.nn.elu):
    
    prev = x
    for i in range(num_conv):
            
        w_conv = tf.get_variable("w"+str(i), shape= [f,f,chan_in,chan_out],
                                  initializer= tf.contrib.layers.xavier_initializer_conv2d())
        b_conv = tf.get_variable("b"+str(i), shape= [chan_out])
        a = nonlin(conv_keepdim(prev, w_conv) + b_conv)
        
        prev = tf.nn.dropout(a, dropout)
        
        chan_in = chan_out
        
    return prev

def deconv(x, W, num_channels):
    
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

def channel_normalize(x, top= 255):
    maxi = tf.reduce_max(x, reduction_indices= [1,2])
    mini = tf.reduce_min(x, reduction_indices= [1,2])
    return top * ( (x - mini) / (maxi - mini) )

def gather_indices(x, idx):
    """
    Differentiable version of gather_nd from: https://github.com/tensorflow/tensorflow/issues/206
    """
    
    idx_flattened = tf.range(0, x.get_shape()[0]) * x.get_shape()[1] + idx
    y = tf.gather(tf.reshape(x, [-1]),  # flatten input
                  idx_flattened)  # use flattened indices
    return y

def extract_last_relevant(outputs, length, bs):
    """
    Args:
        outputs: [Tensor(batch_size, output_neurons)]: A list containing the output
            activations of each in the batch for each time step as returned by
            tensorflow.models.rnn.rnn.
        length: Tensor(batch_size): The used sequence length of each example in the
            batch with all later time steps being zeros. Should be of type tf.int32.

    Returns:
        Tensor(batch_size, output_neurons): The last relevant output activation for
            each example in the batch.
            
    code solution from: https://github.com/tensorflow/tensorflow/issues/206
    """
    output = tf.transpose(tf.pack(outputs), perm=[1, 0, 2])
    # Query shape.
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    num_neurons = int(output.get_shape()[2])
    
    # Index into flattened array as a workaround.
    #index = tf.range(0, batch_size) * max_length + (length - 1)
    index = tf.range(0, batch_size) * max_length + (length - 1)
    #flat = tf.reshape(output, [-1, num_neurons])
    flat = tf.reshape(output, [bs*max_length, num_neurons])
    
    relevant = tf.gather(flat, index)
    return relevant

def drop_fc(x, W, b, p, nonlin):
    """
    Fully connected layer with built in dropout
    """
    a= nonlin( tf.matmul(x, W) + b )
    y = tf.nn.dropout(a, 1.0-p)
    return y
    

def updown_module(x, convolve, c_up, c_down, up= 'deconv', f_up= 2, f_down= 7, t= tf.nn.elu, dropout= 1.0):
    [batch_size, H, W, c_in] = x.get_shape().as_list()
    
    if up == 'deconv':
        w_convT0 = tf.get_variable("wT", shape= [f_up, f_up, c_up, c_in],
                                   initializer= tf.contrib.layers.xavier_initializer_conv2d())
        b_convT0 = tf.get_variable("bT", shape= [c_up])
        h_convT0 = t(deconv(x, w_convT0, c_up) + b_convT0)
        a = tf.nn.dropout(h_convT0, dropout)
        
        conv_c_in = c_up
        
    elif up == 'resize':
        conv_c_in = c_in
        
        H2 = 2 ** (np.log2(H) + 1)
        W2 = 2 ** (np.log2(W) + 1)
        
        a = tf.image.resize_images(x, H2, W2)

    if convolve:
        w_conv0 = tf.get_variable("w", shape= [f_down,f_down, conv_c_in, c_down],
                                  initializer= tf.contrib.layers.xavier_initializer_conv2d())
        b_conv0 = tf.get_variable("b", shape= [c_down])        
        h_conv0 = t(conv_keepdim(a, w_conv0) + b_conv0)
        a = tf.nn.dropout(h_conv0, dropout)

        prev_layer = a
    else:
        try:
            prev_layer = tf.nn.top_k(a, c_down)[0]
        except ValueError: #c_down is greater than the number of channels.
            prev_layer = a
        
    return prev_layer

def MSE_loss(predicted, target):
    batch_size = predicted.get_shape()[0].value
    
    x= tf.reshape(predicted, [batch_size, -1])
    y= tf.reshape(target, [batch_size, -1])
    
    #return tf.reduce_mean(tf.sqrt(tf.matmul(tf.transpose(x-y),(x-y))))
    return tf.reduce_sum( tf.reduce_mean(0.5 *  tf.square(x - y), 1 ) )

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
    """
    Cosine loss with Laplace smoothing.
    """
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
