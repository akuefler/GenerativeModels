import os
import numpy as np
import re

from sklearn import neighbors as neigh

from nltk.tokenize import word_tokenize as wtok
from nltk.tokenize import sent_tokenize as stok

from itertools import chain

import codecs
import string

punc = set(string.punctuation)
dig = set(string.digits)

def create_vocab(mode= 'train'):
    assert mode == 'train' or mode == 'test'
    
    vocab = set([])
    dirname = './aclImdb/'+mode
    sentiment = ['pos', 'neg']
    if mode== 'train':
        sentiment += ['unsup']
    
    for s in sentiment:
        dname = dirname+'/'+s+'/'
        for txt in os.listdir(dname):
            with codecs.open(dname+txt,'r', 'utf-8-sig') as f:
                for line in f:
                    sent = line.lower()
                    tokens= wtok(sent)
                    #tokens2= [tok.split('-') for tok in tokens1]
                    
                    [vocab.add(tok) for tok in tokens]
                    
    vlis = np.array(list(vocab))
    np.savez(mode+'_vocab1', vlis)
    
def refine_vocab(path= 'train_vocab1'):
    M = np.load(path+'.npz')
    v= M['arr_0']
    
    ref_vocab= []
    for term in v:
        s= ''.join([ch for ch in term if ch not in punc and ch not in dig])
        ref_vocab += [s]
            
    S = set(ref_vocab)
    ref_vocab = np.sort(list(S))
    
    np.savez(path[:-1]+'2', ref_vocab)
    
def merge_vocabs():
    M = np.load('train_vocab2'+'.npz')
    v_train= M['arr_0'] 
    
    M = np.load('test_vocab2'+'.npz')
    v_test= M['arr_0']
    
    v = np.concatenate((v_train,v_test))
    v = np.sort(list(set(v)))
    
    np.savez('train_test_vocab', v)
    
def extract_glove(filename= './data/gloves/glove.6B.50d.txt'):
    M = np.load('train_test_vocab'+'.npz')
    vocab_lis= M['arr_0']
    
    valid_words= []
    word_vecs= []
    
    c = 0
    
    with open(filename) as f:
        for line in f:
            header= line.split(' ')[0]
            try:
                if header in vocab_lis:
                    vec = np.array(line.split(' ')[1:], dtype= float)
                    
                    valid_words.append(header)
                    word_vecs.append(vec)
                    
            except UnicodeDecodeError:
                pass
            
    word2ix = {word : ix for (word, ix) in zip(valid_words,range(len(valid_words)))}
    embedding= np.row_stack([word_vecs])
    
    np.savez('embed_matrix', embedding)
    np.savez('word2ix_dic', word2ix)
    
    #n = len(vocab_dic.keys())
    #np.savez('vdic_'+str(n)+'n_50d',vocab_dic)
                         
#refine_vocab(path= 'test_vocab1')                
#refine_vocab(path= 'train_vocab1')
#merge_vocabs()
#extract_glove()

class IMDB_Dataset():
    
    def __init__(self):
        M = np.load('embed_matrix.npz')
        self.L = M['arr_0']
        
        self.L_KNN=\
            neigh.NearestNeighbors(\
                n_neighbors=1, algorithm='ball_tree').fit(self.L)
        
        N = np.load('word2ix_dic.npz')
        self.word2ix = N['arr_0'].all()      

        halt= True
        
    def translate(self, X, eos):
        """
        X = (batch_size, max_step, vector_dim)
        eos=(batch_size,)
        """
        batch_size, max_step, dim = X.shape
        lang= np.array(self.word2ix.keys())
        sentences= []
        
        for i, steps in enumerate(X):
            #for j in range(eos[i]):   
            distances, matches= self.L_KNN.kneighbors(steps)
            sent= ' '.join( list( lang[matches[0:eos[i]].flatten()] ) )
            sentences.append(sent)
            
        return sentences
                
        
    def getReviewSentences(self, path= None, mode= 'train', sentiment= 'pos', ix= '0_2'):
        sentences= []
        
        if path is None:
            path = './aclImdb/'+mode+'/'+sentiment+'/'+ix+'.txt'
            
        with codecs.open(path,'r', 'utf-8-sig') as f:
            for line in f:
                sent = line.lower()
                tokens= wtok(sent)
                #tokens2= [tok.split('-') for tok in tokens1]
                
                sentence= []
                for tok in tokens:
                    
                    try:
                        sentence.append(self.word2ix[tok])
                    except KeyError:
                        pass
                    
                    if tok == '.' and len(sentence) >= 3:
                        x = np.array(sentence)
                        sentences.append(x)
                        sentence= []
                     
        return sentences            
            
        
        

#imdb = IMDB_Dataset()
#imdb.getReviewSentences('./aclImdb/train/neg/0_3.txt')

halt= True