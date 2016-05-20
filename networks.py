import tensorflow as tf
from layers import *

import time
import tqdm
import os

"""
GAN code from:
http://evjang.com/articles/genadv1

G(z) = x'
Truth ~ x
Max: D_1(x)
Min: D_2(x')

"""
#batch_size= 10
#EPOCHS= 10

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
    
    def trainROT(self, data, save= True, EPOCHS= 100, interval= 10):
        model = self.model
        self.sess.run(tf.initialize_all_variables())
        path= ''
        
        if save:
            saver = tf.train.Saver()
            
            folder_name = time.strftime("%d-%m-%Y")+'_'+(time.strftime("%H-%M-%S"))
            path = './img_weights'+folder_name
            
            os.mkdir(path)
                        
            f = open(path+'/'+'results.txt','w')
            for hp_key, hp_val in model.hyperparams.iteritems():
                f.write(hp_key+": "+str(hp_val)+'\n')            
        
        print "Epochs: ", EPOCHS
        
        for i in range(EPOCHS):
            losses = []
            for minibatch in tqdm.tqdm(data.iterate_minibatches(\
                mode= 'train', batch_size= model.batch_size, shuffle= True)):
                X1, X2, Y= data.minibatch_2_Xy(minibatch)
                
                g_feed= {model.x1_input: X1, model.x2_input: X2, model.y_input: Y}
                loss, optout =self.sess.run([model.J, self.opt], g_feed) # update generator
                losses.append(loss)
                
            if save:
                save_path = saver.save(self.sess,\
                                       'img_weights/'+folder_name+"/epoch"+str(i)+".ckpt")
            if i % (EPOCHS // interval) == 0:
                ave_loss = np.mean(losses)
                print("completed: ", float(i)/float(EPOCHS))
                print("loss: ", ave_loss)
                
                if save:
                    f.write("Epoch: "+str(i)+'\n')
                    f.write("Train Loss: "+str(ave_loss)+'\n')
                
        if save:
            f.close
            
        return path + 'epoch'+str(i-1)+'.ckpt'
                
    def trainGAN(self, DATA, k= 1, EPOCHS= 100):
        # Algorithm 1 of Goodfellow et al 2014
        #histd, histg= np.zeros(EPOCHS), np.zeros(EPOCHS)
        model= self.model
        
        self.sess.run(tf.initialize_all_variables())
        
        for i in range(EPOCHS):
            for j in range(k):
                x, xeos = DATA.sample_batch(model.batch_size)
                z, zeos = DATA.sample_batch(model.batch_size, noise= True)
                
                #d_feed= {x_input: np.reshape(x,(batch_size,1)),\
                            #z_input: np.reshape(z,(batch_size,1))}
                d_feed= {model.z_input : z,
                        model.z_eos : zeos,
                        model.x_input : x,
                        model.x_eos : xeos,
                        }
                
                loss_d, _ = self.sess.run([model.obj_d, model.opt_d], d_feed)
                
            z, zeos = DATA.sample_batch(model.batch_size, noise= True)
            
            g_feed= {model.z_input: z, model.z_eos: zeos}
            
            loss_g, _ = self.sess.run([model.obj_g, model.opt_g], g_feed)
            
            if i % (EPOCHS//10) == 0:
                print "loss g: ", loss_g
                print(float(i)/float(EPOCHS))     
          
        
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
    def __init__(self, data, batch_size= 1, max_steps= 35, seq_width= 50):

        self.word2ix = data.word2ix
        self.vocab_size = len(self.word2ix.keys())
        
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.seq_width = seq_width
        
        self.L = tf.constant(data.L, name= 'embed', dtype= tf.float32)
        
        with tf.variable_scope("G"):
            self.z_input = tf.placeholder(tf.int32,[self.batch_size, self.max_steps])
            
            self.z_seq = tf.nn.embedding_lookup(self.L, self.z_input)
            self.z_eos = tf.placeholder(tf.int32, shape= (self.batch_size,))
            
            G, theta_g = self.generator(self.z_seq)
            
            self.g_outs= G
            
        with tf.variable_scope("D") as scope:
            self.x_input=tf.placeholder(tf.int32,[self.batch_size, self.max_steps]) # input M normally distributed floats
            
            #D1 = self.discriminator2(self.x_input)
            self.x_seq = tf.nn.embedding_lookup(self.L, self.x_input)
            self.x_eos = tf.placeholder(tf.int32, shape= (self.batch_size,))
            D1, _ = self.discriminator(self.x_seq, self.x_eos)
            
            self.d1_outs= D1
            
            scope.reuse_variables()
            
            D2, theta_d= self.discriminator(G, self.z_eos)
            
            self.d2_outs= D2
            
            #theta_d = [w for w in tf.all_variables() if w.name[0] == 'D']
                    
        self.obj_d=tf.reduce_mean(tf.log(D1)+tf.log(1-D2))
        self.obj_g=tf.reduce_mean(tf.log(D2))
        
        adam = tf.train.AdamOptimizer()
            
        # set up optimizer for G,D        
        self.opt_d= adam.minimize(1-self.obj_d, var_list= theta_d)
        self.opt_g= adam.minimize(1-self.obj_g, var_list= theta_g) # maximize log(D(G(z)))
        
        
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
        
        return G, [w for w in tf.all_variables() if w.name[0] == 'G']
        
    
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
        
        output= gather_indices(predictions, eos)
            
        return output, [w for w in tf.all_variables() if w.name[0] == 'D']     
        
                         
class RotNet(NeuralNet):
    def __init__(self, hyperparams, concat_ave= False, concat_ins= False):
        NeuralNet.__init__(self, batch_size= hyperparams['batch_size'])
        
        self.hyperparams= hyperparams
        
        self.batch_size= self.hyperparams['batch_size']
        self.scale = self.hyperparams['scale']
        self.num_ups= self.hyperparams['num_ups']
        
        #self.batch_size= 10
        self.C= 3
        self.H= 2 ** self.scale
        self.W= 2 ** self.scale
        
        #Dimensions after merging:
        self.C_merge = 1
        self.H_merge = 2 ** (self.scale - self.num_ups)
        self.W_merge = 2 ** (self.scale - self.num_ups)
        
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
            
        if self.hyperparams['loss'] == 'cosine':
            self.J = cosine_loss(self.output_layer, self.y_input)
        
        
    def objective(self, predicted, truth):
        """
        Loss from: http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf
        """
        
        loss = tf.nn.l2_loss(predicted - truth)
        
    def extractor(self, x, output_dim, num_conv= 2, chan_out= 50):
        """
        Call this within a scope.
        """        
        f = 3
        
        #Convolution 0
        
        #h_chan0 = 50
        #w_conv0 = tf.get_variable("w0", shape= [self.filter_size,self.filter_size,3,h_chan0],
                                  #initializer= tf.contrib.layers.xavier_initializer_conv2d())
        #b_conv0 = tf.get_variable("b0", shape= [h_chan0])        
        #h_conv0 = tf.nn.elu(conv_keepdim(x, w_conv0) + b_conv0)
        #h_pool0 = max_pool_2x2(h_conv0)
        
        h= multi_conv(x, num_conv, 3, chan_out, f, nonlin= tf.nn.elu)
        
        # Shrink
        h_chan2 = 2
        w_conv2 = tf.get_variable("w_shrink", shape= [self.filter_size,self.filter_size,chan_out,h_chan2],
                                  initializer= tf.contrib.layers.xavier_initializer_conv2d())
        b_conv2 = tf.get_variable("b_shrink", shape= [h_chan2])        
        h_conv2 = tf.nn.elu(conv_keepdim(h, w_conv2) + b_conv2)
        h_pool2 = max_pool_4x4(h_conv2)        
        
        # Fully-Connected 1
        h_pool1_flat = tf.reshape(h_pool2, [self.batch_size, 2*((2 ** (self.scale - 2))**2)])
        
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
        #output_shape= (self.batch_size, self.H, self.W, 3)
        
        c_up = self.hyperparams['c_up']
        up= self.hyperparams['up']

        f_down=self.hyperparams['f_down']
        f_out =self.hyperparams['f_out']
        
        convolve_out = self.hyperparams['convolve_out']
        convolve_up = self.hyperparams['convolve_up']
        
        skip_connect= self.hyperparams['skip_connect']
        
        if skip_connect:
            c_down= 3
        else:
            c_down = self.hyperparams['c_down']
        
        f = 2        
        updown = x
        for i in range(self.num_ups):
            with tf.variable_scope("UP%i"%i):
                
                tf.get_variable_scope()
                ##Deconv 1:        
                updown = updown_module(updown, 0, convolve= convolve_up, 
                                    c_up= c_up, c_down= c_down, up= 'deconv',
                                    f_up=2, f_down= f_down, t=tf.nn.elu)
                if skip_connect:
                    g = tf.nn.sigmoid(tf.get_variable("gate", shape=(1)))
                    g_x1 = tf.nn.sigmoid(tf.get_variable("gate_x1", shape=(1)))
                    g_x2 = tf.nn.sigmoid(tf.get_variable("gate_x2", shape=(1)))             
        
                    x1 = tf.image.resize_images(self.x1_input, updown.get_shape()[1], updown.get_shape()[2])
                    x2 = tf.image.resize_images(self.x2_input, updown.get_shape()[1], updown.get_shape()[2])
                    updown = updown * g + (g_x1) * x1 + (g_x2) * x2             
        
        if convolve_out:
            w_conv0 = tf.get_variable("w_final1", shape= [f_down,f_down,c_down,c_up],
                                      initializer= tf.contrib.layers.xavier_initializer_conv2d())
            b_conv0 = tf.get_variable("b_final1", shape= [c_up])        
            h_conv0 = tf.nn.elu(conv_keepdim(updown, w_conv0) + b_conv0)
            
            w_conv1 = tf.get_variable("w_final2", shape= [f_out,f_out,c_up,3],
                                      initializer= tf.contrib.layers.xavier_initializer_conv2d())
            b_conv1 = tf.get_variable("b_final2", shape= [3])        
            h_conv1 = tf.nn.elu(conv_keepdim(h_conv0, w_conv1) + b_conv1)
            #h_conv1 = tf.nn.relu(conv_keepdim(h_conv0, w_conv1) + b_conv1)
            
            penul = h_conv1
        else:
            penul = updown2
        
        out = img_normalize(penul)
        
        return out
        
        
