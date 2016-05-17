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
        
class NeuralNet(object):
    def __init__(self, batch_size= 1):
        #self.sess = tf.Session()
        self.batch_size= batch_size
    
    def predict(self, feed):
        #with tf.Session as sess:
        sess= tf.Session()
        sess.run(tf.initialize_all_variables())
        result = sess.run(self.output_layer,
                          feed_dict= feed)
            
        return result

class GAN(object):
    def __init__(self, data):
        #self.L = tf.constant(data.L, name= "L", dtype= tf.float32)
        self.word2ix = data.word2ix
        self.vocab_size = len(self.word2ix.keys())
        
        self.batch_size = 1
        self.max_steps = 35
        self.seq_width = 50
        
        self.L = tf.placeholder(tf.float32, shape= (self.vocab_size, self.seq_width))
        
        with tf.variable_scope("G"):
            self.z_input = tf.placeholder(tf.int32,[self.batch_size, self.max_steps])
            
            self.z_seq = tf.nn.embedding_lookup(self.L, self.z_input)
            self.z_eos = tf.placeholder(tf.int32, shape= (self.batch_size,))
            
            G, theta_g = self.generator(self.z_seq)
            
        with tf.variable_scope("D") as scope:
            self.x_input=tf.placeholder(tf.int32,[self.batch_size, self.max_steps]) # input M normally distributed floats
            
            #D1 = self.discriminator2(self.x_input)
            self.x_seq = tf.nn.embedding_lookup(self.L, self.x_input)
            self.x_eos = tf.placeholder(tf.int32, shape= (self.batch_size,))
            D1= self.discriminator(self.x_seq, self.x_eos)
            
            scope.reuse_variables()
            
            #gMat = tf.reshape(G, (self.batch_size * self.max_steps, self.vocab_size))
            #self.g_seq = tf.reshape( tf.matmul(gMat, self.L),\
                                     #(self.batch_size, self.max_steps, self.seq_width) ) #Embedding lookup with one hot vectors.
            
            D2= self.discriminator(G, self.z_eos)
            
            theta_d = [w for w in tf.all_variables() if w.name[0] == 'D']
                    
        self.obj_d=tf.reduce_mean(tf.log(D1)+tf.log(1-D2))
        self.obj_g=tf.reduce_mean(tf.log(D2))
        
        adam = tf.train.AdamOptimizer()
            
        # set up optimizer for G,D        
        self.opt_d= adam.minimize(1-self.obj_d, var_list= theta_d)
        self.opt_g= adam.minimize(1-self.obj_g, var_list= theta_g) # maximize log(D(G(z)))
        
    
    def trainGAN(self, k= 1, EPOCHS= 10):
        # Algorithm 1 of Goodfellow et al 2014
        histd, histg= np.zeros(EPOCHS), np.zeros(EPOCHS)
        
        for i in range(EPOCHS):
            for j in range(k):
                x = sample_batch()
                x.sort()
                
                z = self.sample_noise()
                d_feed= {x_input: np.reshape(x,(batch_size,1)),\
                            z_input: np.reshape(z,(batch_size,1))}
                
                histd[i],_=sess.run([obj_d,opt_d], d_feed)
                
            z = self.sample_noise()
            
            g_feed= {z_input: np.reshape(z,(batch_size,1))}
            histg[i],_=sess.run([obj_g,opt_g], g_feed) # update generator
            
            if i % (EPOCHS//10) == 0:
                print(float(i)/float(EPOCHS))     
        
        
    def generator(self, z):
        
        inputs = [tf.reshape(i, (self.batch_size, self.seq_width))\
                  for i in tf.split(1, self.max_steps, z)]
        
        lstm_size= 100
        
        cell = tf.nn.rnn_cell.LSTMCell(lstm_size, input_size= self.seq_width)
        
        initial_state= cell.zero_state(self.batch_size, tf.float32)
        
        #eos_ix = tf.reshape( self.z_eos, shape= (self.batch_size,))
        lstm_outputs, states = tf.nn.rnn(cell, inputs,
                                    initial_state= initial_state,
                                    sequence_length= self.z_eos ) #ISSUE: Squeezing
        
        #Variables:
        W1 = tf.get_variable('W1', shape= (lstm_size, self.seq_width))
        b1 = tf.get_variable('b1', shape= (self.seq_width,))
        
        W2 = tf.get_variable('W2', shape= (self.seq_width, self.seq_width))
        b2 = tf.get_variable('b2', shape= (self.seq_width,))        
    
        outputs= [None]*len(lstm_outputs)
        for t, h1 in enumerate(lstm_outputs):
            h2 = tf.nn.elu( tf.matmul(h1, W1) + b1 )
            outputs[t] = ( tf.matmul(h2, W2) + b2 )
            
        
        G = tf.reshape(tf.concat(0,outputs) , shape= (self.batch_size, self.max_steps, self.seq_width) )
        
        return G, [W1, b1, W2, b2] #ISSUE: Remember LSTM weights.
    
    #def discriminator2(self, x):
        #U = tf.get_variable('W', shape= (self.max_steps, 1))
        #b = tf.get_variable('bs', shape= (1,))
        
        #return tf.nn.sigmoid( tf.matmul(tf.cast(x,tf.float32), U) + b )
        
    
    def discriminator(self, x, eos):
        
        inputs = [tf.reshape(i, (self.batch_size, self.seq_width))\
                  for i in tf.split(1, self.max_steps, x)]
        
        lstm_size= 100
        
        cell = tf.nn.rnn_cell.LSTMCell(lstm_size, input_size= self.seq_width)
        
        initial_state= cell.zero_state(self.batch_size, tf.float32)
        
        #eos_ix = tf.reshape( eos, shape= (self.batch_size,))
        lstm_outputs, states = tf.nn.rnn(cell, inputs,
                                    initial_state= initial_state,
                                    sequence_length= eos) #What to do with z_eos vs. x_eos?
        
        U = tf.get_variable('W', shape= (lstm_size, 1))
        b = tf.get_variable('bs', shape= (1,))
            
        outputs= [None]*len(lstm_outputs)
        for t, h in enumerate(lstm_outputs):
            outputs[t] = tf.nn.sigmoid( tf.matmul(h, U) + b )
            
        predictions = tf.concat(1, outputs, name= 'preds')
        
        #Slice off the last prediction from the lstm.
        
        #wons= tf.cast( tf.ones((self.batch_size, 1)) ,dtype= tf.int32)       
        #cutoff= tf.concat(1, [tf.reshape( eos, shape= wons.get_shape() ), wons]) )
        #cutoff= tf.concat(1, [tf.transpose( tf.range(0, limit= 10) ), eos])
        
        ran= tf.expand_dims( ( tf.range(0, limit= self.batch_size) ), 1)
        cutoff= tf.squeeze( tf.concat(1, [ran, tf.expand_dims(eos, 1)]) ) #ISSUE: Squeeze is weird.
        
        output = tf.slice(predictions, cutoff, [self.batch_size, 1]) #x_eos or x_eos-1
            
        return output      
        
                         
class RotNet(NeuralNet):
    def __init__(self, concat_ave= False, concat_ins= False):
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
                
            if concat_ins:
                m1 = tf.image.resize_images(self.x1_input, self.H_merge, self.W_merge)
                m2 = tf.image.resize_images(self.x2_input, self.H_merge, self.W_merge)
                M = tf.concat(3, [m1, m2, M], name='concat')                
                
            self.output_layer = self.generator(M)
            
        self.J = cosine_loss(self.output_layer, self.y_input)
        #self.J = dot_loss(self.output_layer, self.y_input)
        
        
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
        
        
    def generator(self, x, convolve= True, skip_connect= True):
        #x_volume = tf.reshape(x, shape= (self.batch_size, self.H_merge, self.W_merge, self.C_merge))
        output_shape= (self.batch_size, self.H, self.W, 3)
        
        f = 2
        f_down = 3
        #C_hid = 10
        C_hid = 50
        C_merge = x.get_shape()[-1].value

        ##Deconv 1:        
        updown0 = updown_module(x, 0, convolve= True, 
                            c_up= C_hid, c_down= 3, up= 'deconv',
                            f_up=2, f_down= f_down, t=tf.nn.elu)
        if skip_connect:
            g_0 = tf.nn.sigmoid(tf.get_variable("gate_0", shape=(1)))
            g_x1_0 = tf.nn.sigmoid(tf.get_variable("gate_x1_0", shape=(1)))
            g_x2_0 = tf.nn.sigmoid(tf.get_variable("gate_x2_0", shape=(1)))             

            x1_0 = tf.image.resize_images(self.x1_input, updown0.get_shape()[1], updown0.get_shape()[2])
            x2_0 = tf.image.resize_images(self.x2_input, updown0.get_shape()[1], updown0.get_shape()[2])
            updown0 = updown0 * g_0 + (g_x1_0) * x1_0 + (g_x2_0) * x2_0        

        ##Deconv 2:
        updown1 = updown_module(updown0, 1, convolve= True, 
                            c_up= C_hid, c_down= 3, up= 'deconv',
                            f_up=2, f_down= f_down, t=tf.nn.elu)
        if skip_connect:
            g_1 = tf.nn.sigmoid(tf.get_variable("gate_1", shape=(1)))
            g_x1_1 = tf.nn.sigmoid(tf.get_variable("gate_x1_1", shape=(1)))
            g_x2_1 = tf.nn.sigmoid(tf.get_variable("gate_x2_1", shape=(1)))
            
            x1_1 = tf.image.resize_images(self.x1_input, updown1.get_shape()[1], updown1.get_shape()[2])
            x2_1 = tf.image.resize_images(self.x2_input, updown1.get_shape()[1], updown1.get_shape()[2])
            updown1 = updown1 * g_1 + ( (g_x1_1)*x1_1 + (g_x2_1)*x2_1 )
        
        ##Deconv 3:
        updown2 = updown_module(updown1, 2, convolve= True, 
                                    c_up= C_hid, c_down= 3, up= 'deconv',
                                    f_up=2, f_down=3, t=tf.nn.elu)
        if skip_connect:
            g_2 = tf.nn.sigmoid(tf.get_variable("gate_2", shape=(1)))
            g_x1_2 = tf.nn.sigmoid(tf.get_variable("gate_x1_2", shape=(1)))
            g_x2_2 = tf.nn.sigmoid(tf.get_variable("gate_x2_2", shape=(1)))
            
            x1_2 = tf.image.resize_images(self.x1_input, updown2.get_shape()[1], updown2.get_shape()[2])
            x2_2 = tf.image.resize_images(self.x2_input, updown2.get_shape()[1], updown2.get_shape()[2])
            
            updown2 = updown2 * g_2 + ( (g_x1_2)*x1_2 + (g_x2_2)*x2_2 ) 
        
        if True:
            h_chan0 = 3
            
            w_conv0 = tf.get_variable("w_final1", shape= [5,5,3,C_hid],
                                      initializer= tf.contrib.layers.xavier_initializer_conv2d())
            b_conv0 = tf.get_variable("b_final1", shape= [C_hid])        
            h_conv0 = tf.nn.elu(conv_keepdim(updown2, w_conv0) + b_conv0)
            
            w_conv1 = tf.get_variable("w_final2", shape= [1,1,C_hid,3],
                                      initializer= tf.contrib.layers.xavier_initializer_conv2d())
            b_conv1 = tf.get_variable("b_final2", shape= [3])        
            h_conv1 = tf.nn.elu(conv_keepdim(h_conv0, w_conv1) + b_conv1)            
            
            penul = h_conv1
        else:
            penul = updown2
        
        out = img_normalize(penul)
        
        return out
        #return out, [w_convT0, b_convT0, w_convT1, b_convT1]
        
        
