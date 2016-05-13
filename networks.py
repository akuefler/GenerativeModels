import tensorflow as tf
from layers import *

"""
GAN code from:
http://evjang.com/articles/genadv1

G(z) = x'
Truth ~ x
Max: D_1(x)
Min: D_2(x')


"""
batch_size= 10
EPOCHS= 10

def mlp(input, output_dim):
    # construct learnable parameters within local scope
    w1=tf.get_variable("w0", [input.get_shape()[1], 6], initializer=tf.random_normal_initializer())
    b1=tf.get_variable("b0", [6], initializer=tf.constant_initializer(0.0))
    w2=tf.get_variable("w1", [6, 5], initializer=tf.random_normal_initializer())
    b2=tf.get_variable("b1", [5], initializer=tf.constant_initializer(0.0))
    w3=tf.get_variable("w2", [5,output_dim], initializer=tf.random_normal_initializer())
    b3=tf.get_variable("b2", [output_dim], initializer=tf.constant_initializer(0.0))
    # nn operators
    fc1=tf.nn.tanh(tf.matmul(input,w1)+b1)
    fc2=tf.nn.tanh(tf.matmul(fc1,w2)+b2)
    fc3=tf.nn.tanh(tf.matmul(fc2,w3)+b3)
    return fc3, [w1,b1,w2,b2,w3,b3]
    

class Solver(object):
    def __init__(self, model, lr= 1e-4):
        self.model = model
        optimizer = tf.train.AdamOptimizer(learning_rate= lr)
        self.sess = tf.Session()
        self.opt = optimizer.minimize(model.J)
    
    def trainROT(self, X1, X2, Y, EPOCHS= 100):
        model = self.model
        #sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        
        print "Epochs: ", EPOCHS
        
        for i in range(EPOCHS):
            g_feed= {model.x1_input: X1, model.x2_input: X2, model.y_input: Y}
            loss, optout =self.sess.run([model.J, self.opt], g_feed) # update generator
            
            if i % (EPOCHS//10) == 0:
                print("completed: ", float(i)/float(EPOCHS))
                print("loss: ", loss)
    
    def sample_noise(self):
        return np.linspace(-5.0,5.0,batch_size)+np.random.random(batch_size)*0.01 # sample noise prior
        
    def sample_batch(self):
        return np.random.normal(mu,sigma,batch_size)
    
    def trainGAN(self, k= 1):
        # Algorithm 1 of Goodfellow et al 2014
        histd, histg= np.zeros(EPOCHS), np.zeros(EPOCHS)
        
        for i in range(EPOCHS):
            for j in range(k):
                #x= np.random.normal(mu,sigma,batch_size) # sampled m-batch from p_data
                x = sample_batch()
                x.sort()
                
                #z= np.linspace(-5.0,5.0,batch_size)+np.random.random(batch_size)*0.01  # sample m-batch from noise prior
                z = self.sample_noise()
                d_feed= {x_input: np.reshape(x,(batch_size,1)),\
                            z_input: np.reshape(z,(batch_size,1))}
                
                histd[i],_=sess.run([obj_d,opt_d], d_feed)
                
            #z= np.linspace(-5.0,5.0,batch_size)+np.random.random(batch_size)*0.01 # sample noise prior
            z = self.sample_noise()
            
            g_feed= {z_input: np.reshape(z,(batch_size,1))}
            histg[i],_=sess.run([obj_g,opt_g], g_feed) # update generator
            
            if i % (EPOCHS//10) == 0:
                print(float(i)/float(EPOCHS))        
        
class NeuralNet(object):
    def __init__(self):
        #self.sess = tf.Session()
        self.batch_size= 1
    
    def predict(self, feed):
        #with tf.Session as sess:
        sess= tf.Session()
        sess.run(tf.initialize_all_variables())
        result = sess.run(self.output_layer,
                          feed_dict= feed)
            
        return result

class GAN(object):
    def __init__(self):
        
        with tf.variable_scope("G"):
            z_input= tf.placeholder(tf.float32, shape= (batch_size,1))
            G,theta_g=mlp(z_input,1) # generate normal transformation of Z
            G=tf.mul(5.0,G) # scale up by 5 to match range        
            
        with tf.variable_scope("D") as scope:
            # D(x)
            x_input=tf.placeholder(tf.float32, shape=(batch_size,1)) # input M normally distributed floats
            fc,theta_d=mlp(x_input,1) # output likelihood of being normally distributed
            D1=tf.maximum(tf.minimum(fc,.99), 0.01) # clamp as a probability
            
            # make a copy of D that uses the same variables, but takes in G as input
            scope.reuse_variables()
            fc,theta_d=mlp(G,1)
            D2=tf.maximum(tf.minimum(fc,.99), 0.01)
                    
        # What's this for?
        obj_d=tf.reduce_mean(tf.log(D1)+tf.log(1-D2))
        obj_g=tf.reduce_mean(tf.log(D2))        
            
        # set up optimizer for G,D
        opt_d= momentum_optimizer(1-obj_d, theta_d)
        opt_g= momentum_optimizer(1-obj_g, theta_g) # maximize log(D(G(z)))
        
class RotNet(NeuralNet):
    def __init__(self, concat_ave= True):
        NeuralNet.__init__(self)
        
        #self.batch_size= 10
        self.C= 3
        self.H= 128
        self.W= 128
        
        #Dimensions after merging:
        self.C_merge = 1
        self.H_merge = 2 ** 4
        self.W_merge = 2 ** 4
        
        C = self.C
        H = self.H
        W = self.W
        
        self.filter_size = 3
        merge_dim = self.H_merge * self.W_merge * self.C_merge
        
        #with tf.device('/gpu:0'):
        with tf.variable_scope("params"):
            self.y_input= tf.placeholder(tf.float32, shape= (self.batch_size, H, W, C))
            
        # Layers corresponding to perspective 1
        with tf.variable_scope("P1"):
            self.x1_input= tf.placeholder(tf.float32, shape= (self.batch_size, H, W, C), name= 'x1')
            P1 = self.extractor(self.x1_input, merge_dim)
            
        # Layers corresponding to perspective 2
        with tf.variable_scope("P2"):
            self.x2_input= tf.placeholder(tf.float32, shape= (self.batch_size, H, W, C), name= 'x2')
            P2 = self.extractor(self.x2_input, merge_dim)
            
        # Layers for generating output from merge of 1 and 2
        with tf.variable_scope("GEN"):
            M = tf.mul(P1, P2)
            if concat_ave:
                m = self.ave_extractor(self.x1_input, self.x2_input)
                M = tf.concat(3, [m, M], name='concat')
            self.output_layer = self.generator(M)
            
        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, self.labels_placeholder)) + reg
        #loss = tf.sqrt(tf.reduce_sum(tf.square(self.output_layer - self.y_input)))
        
        #self.J = tf.sqrt(tf.reduce_sum(tf.square(self.output_layer - self.y_input)))
        #self.J = tf.nn.l2_loss(self.output_layer - self.y_input)
        self.J = cosine_loss(self.output_layer, self.y_input)
        
    def objective(self, predicted, truth):
        """
        Loss from: http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf
        """
        
        loss = tf.nn.l2_loss(predicted - truth)
        
    def extractor(self, x, output_dim):
        """
        Call this within a scope.
        """
        # construct learnable parameters within local scope
        
        #self.output_layer = tf.add(x, tf.ones_like(x))        
        
        #Convolution 0
        h_chan0 = 50
        w_conv0 = tf.get_variable("w0", shape= [self.filter_size,self.filter_size,3,h_chan0],
                                  initializer= tf.contrib.layers.xavier_initializer_conv2d())
        b_conv0 = tf.get_variable("b0", shape= [h_chan0])        
        h_conv0 = tf.nn.elu(conv_keepdim(x, w_conv0) + b_conv0)
        h_pool0 = max_pool_2x2(h_conv0)
        
        # Shrink
        h_chan2 = 2
        w_conv2 = tf.get_variable("w2", shape= [self.filter_size,self.filter_size,h_chan0,h_chan2],
                                  initializer= tf.contrib.layers.xavier_initializer_conv2d())
        b_conv2 = tf.get_variable("b2", shape= [h_chan2])        
        h_conv2 = tf.nn.elu(conv_keepdim(h_pool0, w_conv2) + b_conv2)
        h_pool2 = max_pool_4x4(h_conv2)        
        
        # Fully-Connected 1
        h_pool1_flat = tf.reshape(h_pool2, [self.batch_size, -1])
        
        w_fc0 = tf.get_variable("w3", [h_pool1_flat.get_shape()[1], output_dim],
                                initializer= tf.contrib.layers.xavier_initializer())
        b_fc0 = tf.get_variable("b3", [output_dim])         
        h_fc0 = tf.nn.elu(tf.matmul(h_pool1_flat, w_fc0) + b_fc0)
        
        # Fully-Connected 2
        w_fc1 = tf.get_variable("w4", [h_fc0.get_shape()[1], output_dim],
                                initializer= tf.contrib.layers.xavier_initializer())
        b_fc1 = tf.get_variable("b4", [output_dim])         
        h_fc1 = tf.nn.elu(tf.matmul(h_fc0, w_fc1) + b_fc1)
        
        h_fc1 = tf.reshape(h_fc1, shape= (self.batch_size, self.H_merge, self.W_merge, self.C_merge))

        return h_fc1
    
    def ave_extractor(self, x1, x2):
        m = (x1 + x2)/2.0
        y = tf.image.resize_images(m, self.H_merge, self.W_merge)
        return y
        
        
    def generator(self, x, convolve= True):
        #x_volume = tf.reshape(x, shape= (self.batch_size, self.H_merge, self.W_merge, self.C_merge))
        output_shape= (self.batch_size, self.H, self.W, 3)
        
        f = 2
        f_down = 3
        #C_hid = 10
        C_hid = 50
        C_merge = x.get_shape()[-1].value

        ##Deconv 1:        
        updown0 = updown_module(x, 0, convolve= True, 
                            c_in= C_merge, c_out= C_hid,
                            f_up=2, f_down= f_down, t=tf.nn.elu)

        ##Deconv 2:
        updown1 = updown_module(updown0, 1, convolve= True, 
                            c_in= C_hid, c_out= C_hid,
                            f_up=2, f_down= f_down, t=tf.nn.elu)
        
        ##Deconv 3:
        updown2 = updown_module(updown1, 2, convolve= True, 
                                    c_in= C_hid, c_out= 3,
                                    f_up=2, f_down=7, t=tf.nn.elu)                
        
        ###Convolve x1
        #w_conv_x1 = tf.get_variable("w_x1", shape= [self.filter_size,self.filter_size,3,3],
                                  #initializer= tf.contrib.layers.xavier_initializer_conv2d())
        #b_conv_x1 = tf.get_variable("b_x1", shape= [3])        
        #h_conv_x1 = tf.nn.elu(conv_keepdim(self.x1_input, w_conv_x1) + b_conv_x1)
        
        ###Convolve x2
        #w_conv_x2 = tf.get_variable("w_x2", shape= [self.filter_size,self.filter_size,3,3],
                                  #initializer= tf.contrib.layers.xavier_initializer_conv2d())
        #b_conv_x2 = tf.get_variable("b_x2", shape= [3])        
        #h_conv_x2 = tf.nn.elu(conv_keepdim(self.x2_input, w_conv_x2) + b_conv_x2)        
        
        #out = img_normalize(updown2 * h_conv_x1 * h_conv_x2)
        out = img_normalize(updown2)
        
        return out
        #return out, [w_convT0, b_convT0, w_convT1, b_convT1]
        
        
