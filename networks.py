import tensorflow as tf
from layers import *

import numpy as np

import time
import tqdm
import os

from sklearn.metrics import confusion_matrix

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

class Solver(object):
    def __init__(self, model):
        self.model = model
        #optimizer = tf.train.AdamOptimizer(learning_rate= lr)
        #self.opt = optimizer.minimize(model.J)
        self.sess = tf.Session()
    
    def train(self, mode, data, load= None, save= True, EPOCHS= 100, interval= 1):
        model = self.model
        saver = tf.train.Saver(max_to_keep= EPOCHS)
        path= ''
        
        ##Set up optimizer with decayed learning rate.
        global_step = tf.Variable(0, trainable=False)        

        if model.hyperparams['decay'] is not None:
            decay_step, decay_rate = model.hyperparams['decay']
            
            learning_rate = tf.train.exponential_decay(\
                model.hyperparams['lr'], global_step,\
                decay_step, decay_rate, staircase=True)
        else:
            learning_rate = model.hyperparams['lr']
        
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        #optimizer = tf.train.AdamOptimizer(learning_rate= model.hyperparams['lr'])
        #optimizer = tf.train.RMSPropOptimizer(learning_rate)
        self.opt = optimizer.minimize(model.J)         
        
        if load is None:
            self.sess.run(tf.initialize_all_variables())
            start = 0
        else:
            (start, weights_path) = load
            saver.restore(self.sess, weights_path)
        
        if save:
            folder_name = time.strftime("%d-%m-%Y")+'_'+(time.strftime("%H-%M-%S"))
            if mode == 'rot':
                path = './img_weights/'+folder_name
            elif mode == 'sent':
                path = './sent_weights/'+folder_name
            elif mode == 'wind':
                path = './wind_weights/'+folder_name
                
            os.mkdir(path)
                        
            f = open(path+'/'+'results.txt','w')
            f.write(str(model.hyperparams))
            
            #for hp_key, hp_val in model.hyperparams.iteritems():
                #f.write(hp_key+": "+str(hp_val)+'\n')            
        
        print "Epochs: ", EPOCHS
        
        prev_train_loss= np.inf
        for i in range(start,EPOCHS):
            train_losses = []
            for minibatch in tqdm.tqdm(data.iterate_minibatches(\
                mode= 'train', batch_size= model.batch_size, shuffle= True)):
                
                ###
                if mode == 'rot':
                    X1, X2, Y= data.minibatch_2_Xy(minibatch)
                    feed= {model.x1_input: X1, model.x2_input: X2, model.y_input: Y}
                    
                    if np.isnan(X1.sum()) or np.isnan(X2.sum()) or np.isnan(Y.sum()):
                        print "WARNING: NaN found."
                        continue
                    
                    if np.isinf(X1.sum()) or np.isinf(X2.sum()) or np.isinf(Y.sum()):
                        print "WARNING: Inf found."
                        continue
                    
                elif mode == 'sent':
                    X, Y, EOS = data.minibatch_2_Xy(minibatch)
                    feed= {model.x_input: X, model.y_input: Y, model.eos_input: EOS}
                elif mode == 'wind':
                    T = data.minibatch_2_Xy(minibatch)
                    feed= {model.T_input: T}
                    
                loss, optout =self.sess.run([model.J, self.opt], feed) # update generator
                train_losses.append(loss)
                ###
                
            global_step += 1
            if (i % interval) == 0 or i == EPOCHS - 1:
                print("completed: ", float(i)/float(EPOCHS))
                
                train_loss = np.mean(train_losses)
                print("train loss: ", train_loss)
                
                #Compute training /validation accuracy
                if mode == 'sent':
                    train_acc= self.compute_accuracy(model, data, 'train')
                    val_acc= self.compute_accuracy(model, data, 'val')
                    
                    print("train acc: ", train_acc)
                    print("val acc: ", val_acc)
                    
                elif mode == 'rot' or mode == 'wind':
                    val_loss= self.compute_val_loss(model, data, mode)
                    print("val loss: ", val_loss)
                    
                #Save parameters
                if save:
                    last_ix= str(i)
                    
                    save_path = saver.save(self.sess,path+"/epoch"+last_ix+".ckpt")
                    f.write("Epoch: "+str(i)+'\n')
                    f.write("train_loss: "+str(train_loss)+'\n')
                    
                    if mode == 'sent':
                        f.write("train_acc: "+str(train_acc)+'\n')
                        f.write("val_acc: "+str(val_acc)+'\n')
                    elif mode == 'rot' or mode == 'wind':
                        f.write("val_loss: "+str(val_loss)+'\n')
                        
                if train_loss == 0 or np.isnan(train_loss) or abs(train_loss-prev_train_loss) < 1e-5:
                    print "converged early!"
                    break
                prev_train_loss = train_loss
                        
        if save:
            f.close
            return save_path
        else:
            return
                
    #def trainGAN(self, DATA, k= 1, EPOCHS= 100, interval= 1):
        #model= self.model
        #start = 0
        
        ##Optimizers
        #self.opt_d = tf.train.AdamOptimizer(model.hyperparams['lr']) \
            #.minimize(model.obj_d, var_list= model.theta_d)
        
        #self.opt_g = tf.train.AdamOptimizer(model.hyperparams['lr']) \
            #.minimize(model.obj_g, var_list= model.theta_g)          
        
        #self.sess.run(tf.initialize_all_variables())
        
        #for i in range(start,EPOCHS):
            #g_losses = []
            #d_losses = []
            #for j, minibatch in enumerate(tqdm.tqdm(DATA.iterate_minibatches(\
                ##Update discriminator
                #mode= 'train', batch_size= model.batch_size, shuffle= True)) ):
                #x, xeos = DATA.minibatch_2_XEos(minibatch)
                #z, zeos = DATA.minibatch_2_XEos(minibatch, noise= True)                
                
                #d_feed= {model.z_input : z,
                        #model.z_eos : zeos,
                        #model.x_input : x,
                        #model.x_eos : xeos,
                        #}
                
                #loss_d, _ = self.sess.run([model.obj_d, self.opt_d], d_feed)
                #d_losses.append(loss_d)
                
                #if (j + 1) % k == 0:
                    ##Update generator
                    #z, zeos = DATA.minibatch_2_XEos(minibatch, noise= True)                            
                    #g_feed= {model.z_input: z, model.z_eos: zeos}
                    #loss_g, _ = self.sess.run([model.obj_g, self.opt_g], g_feed)
                    #g_losses.append(loss_g)                    
            
            #if (i % interval) == 0:              
    
                ##ave_g_loss = np.mean(g_losses)
                #ave_d_loss = np.mean(d_losses)
                
                #print("completed: ", float(i)/float(EPOCHS))
                ##print("loss (G): ", ave_g_loss)
                #print("loss (D): ", ave_d_loss)
                
                #if g_losses != []:
                    #ave_g_loss = np.mean(g_losses)
                    #print("loss (G): ", ave_g_loss)
                    
    def compute_accuracy(self, model, data, mode):
        accs= []
        for minibatch in tqdm.tqdm(data.iterate_minibatches(\
                        mode= mode, batch_size= model.batch_size, shuffle= True)):
            X, Y, EOS = data.minibatch_2_Xy(minibatch)
            feed= {model.x_input: X, model.y_input: Y, model.eos_input: EOS}
            
            softouts =self.sess.run([model.output_layer], feed)
            assert type(softouts) == list
            assert len(softouts) == 1
            
            so = softouts[0]
            Ypred = np.argmax(so, 1)
            #import pdb; pdb.set_trace()
            
            accs.append( float(np.count_nonzero( Ypred - Y)) / model.hyperparams['batch_size'] )
            
        return 1.0 - np.mean(accs)
    
    def compute_confusion(self, model, data, mode):
        accs= []
        M = np.zeros((model.num_classes, model.num_classes))
        
        for minibatch in tqdm.tqdm(data.iterate_minibatches(\
                        mode= mode, batch_size= model.batch_size, shuffle= True)):
            X, Y, EOS = data.minibatch_2_Xy(minibatch)
            feed= {model.x_input: X, model.y_input: Y, model.eos_input: EOS}
            
            softouts =self.sess.run([model.output_layer], feed)
            assert type(softouts) == list
            assert len(softouts) == 1
            
            so = softouts[0]
            Ypred = np.argmax(so, 1)
            #import pdb; pdb.set_trace()
            
            M += confusion_matrix(Y, Ypred, labels=[0,1,2,3,4,5,6,7])
            
            #accs.append( float(np.count_nonzero( Ypred - Y)) / model.hyperparams['batch_size'] )
            
        return M   
    
    def compute_val_loss(self, model, data, mode):
        losses= []
        for minibatch in tqdm.tqdm(data.iterate_minibatches(\
            mode= 'val', batch_size= model.batch_size, shuffle= True)):
            ###
            if mode == 'rot':
                X1, X2, Y= data.minibatch_2_Xy(minibatch)
                feed= {model.x1_input: X1, model.x2_input: X2, model.y_input: Y}
            elif mode == 'sent':
                X, Y, EOS = data.minibatch_2_Xy(minibatch)
                feed= {model.x_input: X, model.y_input: Y, model.eos_input: EOS}
            elif mode == 'wind':
                T = data.minibatch_2_Xy(minibatch)
                feed= {model.T_input: T}
                
            loss =self.sess.run([model.J], feed)
            losses.append(loss)
            ###
            
        return np.mean(losses)     
                    
                    
          
        
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

##Deprecated.
#class GAN(object):
    #def __init__(self, data, hyperparams):

        #self.word2ix = data.word2ix
        #self.vocab_size = len(self.word2ix.keys())
        
        #self.hyperparams= hyperparams
        #self.batch_size = hyperparams['batch_size']
        #self.max_steps = hyperparams['max_steps']
        #self.seq_width = hyperparams['seq_width']
        
        #self.L = tf.constant(data.L, name= 'embed', dtype= tf.float32)
        
        #with tf.variable_scope("G"):
            #self.z_input = tf.placeholder(tf.int32,[self.batch_size, self.max_steps])
            
            #self.z_seq = tf.nn.embedding_lookup(self.L, self.z_input)
            #self.z_eos = tf.placeholder(tf.int32, shape= (self.batch_size,))
            
            #G, self.theta_g = self.generator(self.z_seq)
            
            #self.g_outs= G
            
        #with tf.variable_scope("D") as scope:
            #self.x_input=tf.placeholder(tf.int32,[self.batch_size, self.max_steps]) # input M normally distributed floats
            
            ##D1 = self.discriminator2(self.x_input)
            #self.x_seq = tf.nn.embedding_lookup(self.L, self.x_input)
            #self.x_eos = tf.placeholder(tf.int32, shape= (self.batch_size,))
            #D1, D1_logits, _ = self.discriminator(self.x_seq, self.x_eos)
            
            #self.d1_outs= D1
            
            #scope.reuse_variables()
            
            #D2, D2_logits, self.theta_d= self.discriminator(G, self.z_eos)
            
            #self.d2_outs= D2
            
            ##theta_d = [w for w in tf.all_variables() if w.name[0] == 'D']
                    
        ##self.obj_d=tf.reduce_mean(tf.log(D1)+tf.log(1-D2))
        ##self.obj_g=tf.reduce_mean(tf.log(D2))
        
        ##adam = tf.train.AdamOptimizer()
            
        ## set up optimizer for G,D
        
        ##self.opt_d= adam.minimize(1-self.obj_d, var_list= theta_d)
        ##self.opt_g= adam.minimize(1-self.obj_g, var_list= theta_g) # maximize log(D(G(z)))
        
        #self.d_loss_real = tf.reduce_mean(
            #tf.nn.sigmoid_cross_entropy_with_logits(D1_logits, tf.ones_like(D1)))
        
        #self.d_loss_fake = tf.reduce_mean(
            #tf.nn.sigmoid_cross_entropy_with_logits(D2_logits, tf.zeros_like(D2)))
        
        #self.obj_d = self.d_loss_fake + self.d_loss_real
        
        #self.obj_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #D2_logits, tf.ones_like(D2)))
        
        ##Optimizers.
        ##self.opt_d = tf.train.AdamOptimizer(self.hyperparams['lr']) \
            ##.minimize(self.obj_d, var_list= theta_d)
        
        ##self.opt_g = tf.train.AdamOptimizer(self.hyperparams['lr']) \
            ##.minimize(self.obj_g, var_list= theta_g)        
        
        
        
    #def generator(self, z):
        
        #inputs = [tf.reshape(i, (self.batch_size, self.seq_width))\
                  #for i in tf.split(1, self.max_steps, z)]
        
        #lstm_size= 100
        
        #cell = tf.nn.rnn_cell.LSTMCell(lstm_size, input_size= self.seq_width)
        
        #initial_state= cell.zero_state(self.batch_size, tf.float32)
        
        ##eos_ix = tf.reshape( self.z_eos, shape= (self.batch_size,))
        #lstm_outputs, states = tf.nn.rnn(cell, inputs,
                                    #initial_state= initial_state,
                                    #sequence_length= self.z_eos ) #ISSUE: Squeezing
        
        ##Variables:
        #W1 = tf.get_variable('W1', shape= (lstm_size, self.seq_width))
        #b1 = tf.get_variable('b1', shape= (self.seq_width,))
        
        #W2 = tf.get_variable('W2', shape= (self.seq_width, self.seq_width))
        #b2 = tf.get_variable('b2', shape= (self.seq_width,))        
    
        #outputs= [None]*len(lstm_outputs)
        #for t, h1 in enumerate(lstm_outputs):
            #h2 = tf.nn.elu( tf.matmul(h1, W1) + b1 )
            #outputs[t] = ( tf.matmul(h2, W2) + b2 )
            
        
        #G = tf.reshape(tf.concat(0,outputs) , shape= (self.batch_size, self.max_steps, self.seq_width) )
        
        #return G, [w for w in tf.all_variables() if w.name[0] == 'G']
        
    
    #def discriminator(self, x, eos):
        
        #inputs = [tf.reshape(i, (self.batch_size, self.seq_width))\
                  #for i in tf.split(1, self.max_steps, x)]
        
        #lstm_size= 100
        
        #cell = tf.nn.rnn_cell.LSTMCell(lstm_size, input_size= self.seq_width)
        
        #initial_state= cell.zero_state(self.batch_size, tf.float32)
        
        ##eos_ix = tf.reshape( eos, shape= (self.batch_size,))
        #lstm_outputs, states = tf.nn.rnn(cell, inputs,
                                    #initial_state= initial_state,
                                    #sequence_length= eos) #What to do with z_eos vs. x_eos?
        
        #U = tf.get_variable('W', shape= (lstm_size, 1))
        #b = tf.get_variable('bs', shape= (1,))
            
        #outputs= [None]*len(lstm_outputs)
        #for t, h in enumerate(lstm_outputs):
            #outputs[t] = ( tf.matmul(h, U) + b )
            
        #predictions = tf.concat(1, outputs, name= 'preds')
        
        #logits= gather_indices(predictions, eos)
        #output= tf.nn.sigmoid(logits)
            
        #return output, logits, [w for w in tf.all_variables() if w.name[0] == 'D']     
        
                  
class RotNet(NeuralNet):
    def __init__(self, hyperparams, concat_ave= False, concat_ins= False):
        NeuralNet.__init__(self, batch_size= hyperparams['batch_size'])
        
        self.hyperparams= hyperparams
        
        self.batch_size= self.hyperparams['batch_size']
        self.scale = self.hyperparams['scale']
        self.num_ups= self.hyperparams['num_ups']
        share_weights= self.hyperparams['share_weights']
        
        reg = self.hyperparams['reg']
        
        #self.batch_size= 10
        self.C= 3
        self.H= 2 ** self.scale
        self.W= 2 ** self.scale
        
        #Dimensions after merging:
        self.C_merge = 3
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
        if share_weights:
            name_scope= "P1"
        else:
            name_scope= "P2"
            
        with tf.variable_scope(name_scope) as scope:
            if share_weights:
                scope.reuse_variables()
                
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
            
        J_reg = tf.add_n([tf.nn.l2_loss(t) for t in tf.trainable_variables()])
            
        if self.hyperparams['loss'] == 'cosine':
            self.J = cosine_loss(self.output_layer, self.y_input) + reg * J_reg
            
        elif self.hyperparams['loss'] == 'MSE':
            self.J = MSE_loss(self.output_layer, self.y_input) + reg * J_reg
            
        else:
            assert False
            
        #regloss = tf.add_n([tf.nn.l2_loss(t) for t in tf.trainable_variables()])
        #self.J += hyperparams['reg'] * regloss
        
        
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
        
        h= multi_conv(x, num_conv, 3, chan_out, f, dropout= self.hyperparams['dropout'], nonlin= tf.nn.elu)
        
        # Shrink
        h_chan2 = 2
        w_conv2 = tf.get_variable("w_shrink", shape= [self.filter_size,self.filter_size,chan_out,h_chan2],
                                  initializer= tf.contrib.layers.xavier_initializer_conv2d())
        b_conv2 = tf.get_variable("b_shrink", shape= [h_chan2])        
        h_conv2 = tf.nn.elu(conv_keepdim(h, w_conv2) + b_conv2)
        h_pool2 = max_pool_4x4(h_conv2)
        
        drop0 = tf.nn.dropout(h_pool2, self.hyperparams['dropout'])
        
        # Fully-Connected 1
        h_pool1_flat = tf.reshape(drop0, [self.batch_size, 2*((2 ** (self.scale - 2))**2)])
        
        w_fc0 = tf.get_variable("w3", [h_pool1_flat.get_shape()[1], output_dim],
                                initializer= tf.contrib.layers.xavier_initializer())
        b_fc0 = tf.get_variable("b3", [output_dim])         
        h_fc0 = tf.nn.elu(tf.matmul(h_pool1_flat, w_fc0) + b_fc0)
        
        drop1 = tf.nn.dropout(h_fc0, self.hyperparams['dropout'])        
        
        # Fully-Connected 2
        w_fc1 = tf.get_variable("w4", [drop1.get_shape()[1], output_dim],
                                initializer= tf.contrib.layers.xavier_initializer())
        b_fc1 = tf.get_variable("b4", [output_dim])         
        h_fc1 = tf.nn.elu(tf.matmul(drop1, w_fc1) + b_fc1)
        
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
        use_gates = self.hyperparams['use_gates']
        
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
                updown = updown_module(updown, convolve= convolve_up, 
                                    c_up= c_up, c_down= c_down, up= up,
                                    f_up=2, f_down= f_down, t=tf.nn.elu,
                                    dropout = self.hyperparams['dropout'])
                if skip_connect:            
                    x1 = tf.image.resize_images(self.x1_input, updown.get_shape()[1], updown.get_shape()[2])
                    x2 = tf.image.resize_images(self.x2_input, updown.get_shape()[1], updown.get_shape()[2])
                    
                    if use_gates:
                        g = tf.nn.sigmoid(tf.get_variable("gate", shape=(1)))
                        g_x1 = tf.nn.sigmoid(tf.get_variable("gate_x1", shape=(1)))
                        g_x2 = tf.nn.sigmoid(tf.get_variable("gate_x2", shape=(1)))                         
                        updown = updown * g + (g_x1) * x1 + (g_x2) * x2
                    else:
                        updown = updown + x1 + x2
        
        if convolve_out:
            w_conv0 = tf.get_variable("w_final1", shape= [f_down,f_down,c_down,c_up],
                                      initializer= tf.contrib.layers.xavier_initializer_conv2d())
            b_conv0 = tf.get_variable("b_final1", shape= [c_up])        
            h_conv0 = tf.nn.elu(conv_keepdim(updown, w_conv0) + b_conv0)
            a = tf.nn.dropout(h_conv0, keep_prob= self.hyperparams['dropout'])
            
            w_conv1 = tf.get_variable("w_final2", shape= [f_out,f_out,c_up,3],
                                      initializer= tf.contrib.layers.xavier_initializer_conv2d())
            b_conv1 = tf.get_variable("b_final2", shape= [3])        
            h_conv1 = conv_keepdim(a, w_conv1) + b_conv1
            
            penul = h_conv1
        else:
            penul = updown
            
        if self.hyperparams['output_nonlin'] == 'elu':
            penul= tf.nn.elu(penul)
        elif self.hyperparams['output_nonlin'] == 'rectify':
            penul= tf.nn.relu(penul)            
        elif self.hyperparams['output_nonlin'] == 'linear':          
            pass
        else:
            assert False
                        
        if self.hyperparams['normalize_output']:
            out = img_normalize(penul)
        else:
            out = penul
        
        return out


class SentNet(object):
    def __init__(self, data, hyperparams):

        self.word2ix = data.word2ix
        self.vocab_size = len(self.word2ix.keys())
        
        self.hyperparams= hyperparams
        self.batch_size = hyperparams['batch_size']
        self.max_steps = hyperparams['max_steps']
        self.seq_width = hyperparams['seq_width']
        
        if self.hyperparams['nonlin'] == 'elu':
            self.nonlin = tf.nn.elu
        elif self.hyperparams['nonlin'] == 'rectify':
            self.nonlin = tf.nn.relu        
        
        if self.hyperparams['labels'] == 'multi':
            self.num_classes = 8
        elif self.hyperparams['labels'] == 'binary':
            self.num_classes = 2
        else:
            assert False
        
        self.L = tf.constant(data.L, name= 'embed', dtype= tf.float32)
            
        with tf.variable_scope("params") as scope:
            self.x_input=tf.placeholder(tf.int32,[self.batch_size, self.max_steps])
            self.y_input= tf.placeholder(tf.int32,shape= (self.batch_size,))
            self.eos_input = tf.placeholder(tf.int32, shape= (self.batch_size,))
            
        with tf.variable_scope("network") as scope:
            self.x_seq = tf.nn.embedding_lookup(self.L, self.x_input)
            output_layer, logits = self.discriminator(self.x_seq, self.eos_input)
            
            self.output_layer= output_layer
            
        J_reg = self.create_reg_tensor()
            
        if self.hyperparams['loss'] == 'softmax':
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,self.y_input)
            self.J = tf.reduce_mean( loss ) + J_reg
        
    def discriminator(self, x, eos):
        
        #200 elem List of (batch_size, glove_length) tensors
        inputs = [tf.reshape(i, (self.batch_size, self.seq_width))\
                  for i in tf.split(1, self.max_steps, x)]
        
        fc_units = self.hyperparams['fc_units']
        
        #Convolution
        if self.hyperparams['conv_type'] is not None:
            with tf.variable_scope('conv'):
                num_filters = self.hyperparams['num_filters']
                if self.hyperparams['conv_type'] == 'inception':
                    conv_out = num_filters * 3
                else:
                    conv_out = num_filters
                
                a = self.convolution(x, self.hyperparams['num_filters'])
                W = tf.get_variable('W', shape= (conv_out, fc_units),
                                    initializer= tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('b', shape= (fc_units,))
                
                c = drop_fc(a, W, b, self.hyperparams['drop'], self.nonlin)
        
        #Forward Direction:
        if not self.hyperparams['conv_only']:
            rnn_units= self.hyperparams['rnn_units']
        
            with tf.variable_scope('left_right'):
                rnn_outputs = self.recurrence(inputs, eos, rnn_units)
                
                W = tf.get_variable('W', shape= (rnn_units, fc_units),
                                    initializer= tf.contrib.layers.xavier_initializer())
                b = tf.get_variable('b', shape= (fc_units,))
                    
                outputs= [None]*len(rnn_outputs)
                
                #200 elem List of (batch_size, hid dim) tensors
                for t, h in enumerate(rnn_outputs):
                    outputs[t] = drop_fc(h, W, b, 
                                         self.hyperparams['drop'],
                                         self.nonlin)
                    
                r= extract_last_relevant(outputs, eos, self.hyperparams['batch_size'])
                       
            #Backward Direction:
            if self.hyperparams['bidirection']:
                #assert False #Padding is going to end up on the wrong end. #Note: Inputs just a list
                inputs.reverse()
                
                with tf.variable_scope('right_left'):
                    rnn_outputs = self.recurrence(inputs, None, rnn_units) #No sequence length, because pad at beginning now.
        
                    W = tf.get_variable('W', shape= (rnn_units, fc_units),
                                        initializer= tf.contrib.layers.xavier_initializer())
                    b = tf.get_variable('b', shape= (fc_units,),)
                        
                    outputs_flip= [None]*len(rnn_outputs)
                    for t, h in enumerate(rnn_outputs):
                        outputs_flip[t] = drop_fc(h, W, b, 
                                                  self.hyperparams['drop'],
                                                  self.nonlin)
                        
                    #r_flip= extract_last_relevant(outputs, eos, self.hyperparams['batch_size'])
                    r_flip= outputs_flip[-1]
                    
                    if self.hyperparams['merge'] == 'elem':
                        r = tf.mul(r, r_flip)
                    elif self.hyperparams['merge'] == 'concat':
                        r = tf.concat(1, [r, r_flip])
                        
            #Merge convolutions
            if self.hyperparams['conv_type'] is not None:
                if self.hyperparams['merge'] == 'elem':
                    r = tf.mul(r, c)
                elif self.hyperparams['merge'] == 'concat':
                    r = tf.concat(1, [r, c])
        else:
            r = c
                    
        logits= self.feedforward(r)
        output= tf.nn.softmax(logits)
            
        return output, logits
    
    def feedforward(self, inputs):
        num_layers= self.hyperparams['num_layers']
        num_units = self.hyperparams['fc_units']
        if self.hyperparams['merge'] == 'concat':
            if self.hyperparams['conv_type'] == None:
                num_units *= 2
            else:
                num_units *= 3
            
        h = inputs
        for l in range(num_layers - 1):
            W = tf.get_variable('W_fc'+str(l), shape= (num_units, num_units),
                                initializer= tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b_fc'+str(l), shape= (num_units,))
            h = drop_fc(h, W, b, self.hyperparams['drop'], self.nonlin)
            
        #return logits
        W = tf.get_variable('W_logits', shape= (num_units, self.num_classes),
                            initializer= tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b_logits', shape= (self.num_classes,))        
        y = drop_fc(h, W, b, self.hyperparams['drop'], self.nonlin)
        return y
    
    def convolution(self, inputs, num_units):
        x = tf.expand_dims(inputs, 3)
        chan_in = 1
        
        #Bigram
        w_bigram = tf.get_variable("w_bigram", shape= [2,50,chan_in,num_units],
                                 initializer= tf.contrib.layers.xavier_initializer_conv2d())
        b_bigram = tf.get_variable("b_bigram", shape= [num_units])
        y_bigram = self.nonlin(tf.nn.conv2d(x, w_bigram, strides= [1,1,1,1], padding='VALID') + b_bigram)
        h_bigram = tf.reduce_max(tf.squeeze(y_bigram) , 1)
        
        #Trigram
        w_trigram = tf.get_variable("w_trigram", shape= [3,50,chan_in,num_units],
                                 initializer= tf.contrib.layers.xavier_initializer_conv2d())
        b_trigram = tf.get_variable("b_trigram", shape= [num_units])
        y_trigram = self.nonlin(tf.nn.conv2d(x, w_trigram, strides= [1,1,1,1], padding='VALID') + b_trigram)
        h_trigram = tf.reduce_max(tf.squeeze(y_trigram) , 1)
        
        #Quin-gram
        w_quingram = tf.get_variable("w_quingram", shape= [3,50,chan_in,num_units],
                                 initializer= tf.contrib.layers.xavier_initializer_conv2d())
        b_quingram = tf.get_variable("b_quingram", shape= [num_units])
        y_quingram = self.nonlin(tf.nn.conv2d(x, w_trigram, strides= [1,1,1,1], padding='VALID') + b_trigram)
        h_quingram = tf.reduce_max(tf.squeeze(y_quingram) , 1)
        
        if self.hyperparams['conv_type'] == 'bigram':
            h = h_bigram
        elif self.hyperparams['conv_type'] == 'trigram':
            h = h_trigram
        elif self.hyperparams['conv_type'] == 'quingram':
            h = h_quingram            
        elif self.hyperparams['conv_type'] == 'inception':
            h = tf.concat(1, [h_bigram, h_trigram, h_quingram])
            
        return h
            
    
    def recurrence(self, inputs, eos, num_units):
        
        if self.hyperparams['recur'] == 'lstm':
            cell = tf.nn.rnn_cell.LSTMCell(num_units, input_size= self.seq_width)
        elif self.hyperparams['recur']== 'gru':
            cell = tf.nn.rnn_cell.GRUCell(num_units, input_size = self.seq_width)
        elif self.hyperparams['recur']== 'rnn':
            cell = tf.nn.rnn_cell.BasicRNNCell(num_units,input_size=self.seq_width)
        
        initial_state= cell.zero_state(self.batch_size, tf.float32)
        
        rnn_outputs, states = tf.nn.rnn(cell, inputs,
                                    initial_state= initial_state,
                                    sequence_length= eos)
        return rnn_outputs
    