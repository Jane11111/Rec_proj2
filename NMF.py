# -*- coding: utf-8 -*-
# @Time    : 2020-04-20 20:47
# @Author  : zxl
# @FileName: NMF.py

import sys
import random
import numpy as np
import tensorflow as tf
from datetime import datetime

class NMF:

    def __init__(self,learning_rate=0.001,batch_size=10000,iteration=20):
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.iteration=iteration

    def myShuffle(self,U,I,y):
        """
        对数据进行shuffle
        :param X:
        :param y1:
        :param y2:
        :return:
        """
        idx=[i for i in range(len(U))]
        random.shuffle(idx)
        U_=U[idx]
        I_=I[idx]
        y_=y[idx]
        return (U_,I_,y_)

    def fit(self,U,I,Y):
        """
        :param U: one-hot向量
        :param I:
        :param Y:0/1 本身包含正例与负例
        :return:
        """
        m=len(set(U))#用户数目
        n=len(set(I))#物品数目
        h_mf=8
        h_mlp=16
        h_nmf=8

        U_ = tf.placeholder(dtype=tf.int64, shape=[None, ], name='U_')
        I_ = tf.placeholder(dtype=tf.int64, shape=[None, ], name='I_')

        #Encoding
        init_random = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
        Wu_mf=tf.get_variable('Wu_mf',shape=[m,h_mf],initializer=init_random,dtype=tf.float64)
        Wi_mf=tf.get_variable('Wi_mf',shape=[n,h_mf],initializer=init_random,dtype=tf.float64)
        Wu_mlp=tf.get_variable('Wu_mlp',shape=[m,h_mlp],initializer=init_random,dtype=tf.float64)
        Wi_mlp=tf.get_variable('Wi_mlp',shape=[n,h_mlp],initializer=init_random,dtype=tf.float64)

        Hu_mf = tf.nn.embedding_lookup(Wu_mf, ids=U_)
        Hi_mf = tf.nn.embedding_lookup(Wi_mf, ids=I_)
        Hu_mlp = tf.nn.embedding_lookup(Wu_mlp, ids=U_)
        Hi_mlp = tf.nn.embedding_lookup(Wi_mlp, ids=I_)

        #Gmf
        H_gmf = tf.multiply(Hu_mf, Hi_mf)  # element-wise 乘

        #mlp
        H_mlp = tf.concat([Hu_mlp, Hi_mlp], 1)

        num_layers=[32,16,8]
        last_vec=H_mlp
        for i in range(len(num_layers)):
            layer=tf.layers.dense(inputs=last_vec,units=num_layers[i],activation=tf.nn.relu,use_bias=True)
            last_vec=layer

        #NeuMF
        # W_nmf = tf.get_variable('W_nmf', shape=[num_layers[-1] + h_mf, h_nmf], initializer=init_random, dtype=tf.float64)
        # b_nmf = tf.get_variable('b_nmf', shape=[h_nmf], initializer=init_random, dtype=tf.float64)

        H_con=tf.concat([H_gmf,last_vec],1)
        # H_nmf=tf.matmul(H_con,W_nmf)+b_nmf
        H_nmf=tf.layers.dense(inputs=H_con,units=h_nmf,activation=tf.nn.relu,use_bias=True)
        # pred_y=tf.nn.sigmoid(H_nmf)
        pred_y=tf.layers.dense(inputs=H_nmf,units=1,activation=tf.nn.sigmoid,use_bias=True)

        tf.add_to_collection('pred_y',pred_y)

        #loss
        y_=tf.placeholder(dtype=tf.float64,shape=[None,1])

        entropy_loss=tf.reduce_mean(-(y_*tf.log(pred_y)+(1-y_)*tf.log(1-pred_y)))
        reg_loss=tf.reduce_mean(tf.nn.l2_loss(Wu_mf))+tf.reduce_mean(tf.nn.l2_loss(Wu_mlp))+tf.reduce_mean(tf.nn.l2_loss(Wi_mf))+tf.reduce_mean(tf.nn.l2_loss(Wi_mlp))
                 # tf.reduce_mean(tf.nn.l2_loss(W_nmf))+tf.reduce_mean(tf.nn.l2_loss(b_nmf))
        # for i in range(len(num_layers)):
        #     reg_loss+=tf.reduce_mean(tf.nn.l2_loss(W_lst[i])+tf.nn.l2_loss(b_lst[i]))

        loss=entropy_loss+0.001*reg_loss

        optimizer=tf.train.AdamOptimizer(self.learning_rate)
        train=optimizer.minimize(loss)

        saver = tf.train.Saver()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            for e in range(self.iteration):#迭代次数
                #TODO 要写成batch的形式

                cur_loss=0

                new_U,new_I,new_y=self.myShuffle(U,I,Y)
                # new_U,new_I,new_y=U,I,Y
                i=0
                count=0
                while i<len(U):
                    count+=1
                    cur_U=new_U[i:min(i+self.batch_size,len(U)),]
                    cur_I=new_I[i:min(i+self.batch_size,len(U)),]
                    cur_Y=new_y[i:min(i+self.batch_size,len(U)),]

                    sess.run(train,feed_dict={U_:cur_U,I_:cur_I,y_:cur_Y})
                    aa=sess.run(loss,feed_dict={U_:cur_U,I_:cur_I,y_:cur_Y})
                    cur_loss+=aa
                    # print('---epoch: %d, cur_loss: %f'%(e,aa))
                    i+=self.batch_size
                print("epoch: %d, loss: %f"%(e,cur_loss/count))
                print(datetime.now())

            self.model_path='model/nmf'
            saver.save(sess,self.model_path)


    def predict(self,U,I):


        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("model/nmf.meta")
            saver.restore(sess,'model/nmf')
            U_=tf.get_default_graph().get_tensor_by_name("U_:0")
            I_=tf.get_default_graph().get_tensor_by_name("I_:0")
            y_=tf.get_collection("pred_y")[0]

            pred_y=sess.run(y_,feed_dict={U_:U,I_:I})
            return pred_y


