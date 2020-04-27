# -*- coding: utf-8 -*-
# @Time    : 2020-04-21 17:36
# @Author  : zxl
# @FileName: main.py

import sys
import random
import numpy as np
from NMF import NMF
from GMF import GMF
from MLP import MLP
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
def split_data(R):
    """
    把数据分为训练集，验证集，测试集
    训练集：每个正例随机采4个负例
    验证集：随机选1个
    测试集：最后一个交互+100个无交互
    :param R:评分矩阵
    :return:train，validation，test，三元组矩阵
    """
    train_neg=4
    test_neg=100

    user_lst=R[:,0]
    item_lst=R[:,1]

    train_lst=[]
    validation_lst=[]
    test_lst=[]

    last_idx=0
    for i in range(len(R)):
        if R[i][0]==R[last_idx][0] and not i==len(R)-1:
            continue
        cur_idx=i-1
        if i==len(R)-1:
            cur_idx=i

        cur_item_lst=R[last_idx:cur_idx+1,1]
        arr=list(set(item_lst).difference(set(cur_item_lst)))
        n1=train_neg*(cur_idx-last_idx+1)
        n2=test_neg
        neg_item=random.sample(arr,min(n1+n2,len(arr)))
        print('n1: %d'%n1)
        print('n2: %d'%n2)
        print('len(arr): %d'%len(arr))
        print('---------------')

        u = R[last_idx][0]
        for tup in R[last_idx+1:cur_idx-1]:
            train_lst.append([tup[0],tup[1],1])
        validation_lst.append([R[last_idx][0],R[last_idx][1],1])#第一个当作validation
        test_lst.append([R[cur_idx][0],R[cur_idx][1],1])#最后一个当成测试


        for j in range(min(n1,len(neg_item))):#为训练集和测试集添加负样本
            train_lst.append([u,neg_item[j],0])
        for j in np.arange(len(neg_item)-1,len(neg_item)-1-n2,-1):
            test_lst.append([u,neg_item[j],0])
        last_idx=i#下一个用户的数据

    train=np.array(train_lst)
    validation=np.array(validation_lst)
    test=np.array(test_lst)

    train_user=len(set(train[:,0]))
    train_item=len(set(train[:,1]))
    validation_user=len(set(validation[:,0]))
    validation_item=len(set(validation[:,1]))
    test_user=len(set(test[:,0]))
    test_item=len(set(test[:,1]))
    print("训练集： %d, %d"%(train_user,train_item))
    print("验证集： %d, %d"%(validation_user,validation_item))
    print("测试集： %d, %d"%(test_user,test_item))

    return train,validation,test

def write_arr(dataset,file_path):
    with open(file_path,'w') as w:
        for arr in dataset:
            w.write(str(arr[0])+','+str(arr[1])+','+str(arr[2])+'\n')
def load_arr(file_path):
    lst=[]
    with open(file_path,'r') as f:
        l=f.readline()
        while l:
            l=l.replace('\n','')
            arr=l.split(',')
            lst.append([int(arr[0]),int(arr[1]),int(arr[2])])
            l=f.readline()
    return np.array(lst)

def hating_rate(pred_y,m,k):
    """
    hating rate@k
    :param pred_y:
    :param m: 每m个为一组
    :param k: 取前k个
    :return:
    """
    pred_y=pred_y.flatten()
    res=0
    count=0
    for i in np.arange(0,len(pred_y),m):
        count+=1
        arr=pred_y[i:i+m]
        idx=np.argsort(-arr)
        # print(idx[0])
        if idx[0]<k:
            res+=1
    return res/count


def ndcg(pred_y,m,k):
    """

    :param pred_y: 预测的值
    :param m: 每m个为一组
    :param k: 取前10个
    :return:
    """
    pred_y=pred_y.flatten()
    res=0
    count=0
    for i in np.arange(0,len(pred_y),m):
        count += 1
        arr = pred_y[i:i + m]
        idx = np.argsort(-arr)
        if idx[0]<k:
            res+=1/(np.log2(idx[0]+1+1))
    return res/count





if __name__ == "__main__":
    root = "/Users/jane/Documents/ECNU/研一下/推荐系统/作业/作业1/ml-1m/ml-1m/"
    out_root="/Users/jane/Documents/ECNU/研一下/推荐系统/作业/作业2/data/"
    train_path="train.txt"
    validation_path="validation.txt"
    test_path="test.txt"
    rating_file = "ratings.txt"

    train=load_arr(train_path)
    test=load_arr(test_path)
    validation=load_arr(validation_path)

    usr_encoder=LabelEncoder()
    item_encoder=LabelEncoder()
    usr_encoder.fit(train[:,0].flatten())
    item_encoder.fit(train[:,1].flatten())

    train_U=usr_encoder.transform(train[:,0].reshape(-1,1))
    train_I=item_encoder.transform(train[:,1].reshape(-1,1))
    train_y=train[:,2].reshape(-1,1)

    test_U=usr_encoder.transform(test[:,0].reshape(-1,1))
    test_I=item_encoder.transform(test[:,1].reshape(-1,1))
    test_y=test[:,2].reshape(-1,1)

    validation_U=usr_encoder.transform(validation[:,0].reshape(-1,1))
    validation_I=item_encoder.transform(validation[:,1].reshape(-1,1))
    validation_y=validation[:,2].reshape(-1,1)

    #参数
    args=sys.argv
    learning_rate=float(args[1])
    batch_size=int(sys.argv[2])
    iteration=int(sys.argv[3])
    print('start training , learning_rate: %f, batch_size: %d, iter: %d'%(learning_rate,batch_size,iteration))


    print('-------------'+'satrt time '+str(datetime.now())+'--------------')
    model=NMF(learning_rate=learning_rate,batch_size=batch_size,iteration=iteration)
    # model=MLP(learning_rate=learning_rate,batch_size=batch_size,iteration=iteration)
    # model=GMF(learning_rate=learning_rate,batch_size=batch_size,iteration=iteration)

    model.fit(train_U,train_I,train_y)
    res=model.predict(test_U,test_I)
    m=101
    k=10
    hr=hating_rate(res,m,k)
    ndcg=ndcg(res,m,k)
    print('hating rate: %f'%hr)
    print('ndcg: %f'%ndcg)
    with open('res-nmf.txt','w') as w:
        w.write('learning_rate: %f, batch_size: %d, iter: %d\n' % (learning_rate, batch_size, iteration))
        w.write('hating rate: %f\n' % hr)
        w.write('ndcg: %f\n' % ndcg)
        w.write('---------------\n')
    print('-------------'+'end time '+str(datetime.now())+'--------------')


    # R=[]
    # with open(rating_file,'r') as f:
    #     l=f.readline()
    #     # c=1000
    #     while l :
    #         # c-=1
    #         l=l.replace('\n','')
    #         arr=l.split('::')
    #         R.append([int(arr[0]),int(arr[1]),int(arr[2])])
    #         l=f.readline()
    # R=np.array(R)
    # print(len(set(R[:,0])))
    # print(len(set(R[:,1])))
    # train,validation,test=split_data(R)
    # write_arr(train,train_path)
    # write_arr(validation,validation_path)
    # write_arr(test,test_path)

