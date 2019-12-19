#!/usr/bin/env python3
import numpy as np 
import pandas as pd
import csv
import random 
from random import shuffle
from math import log
def data(filename,ratio=0.8,test=0):
    """"
    Arguments: file name, ratio of spliting the data, bool: 0 for training, test for anthing else
    output: 
    train data, y_train data, validation data, y valid data, test data 
    """
    if(test==0):
        print("this is a data, we need to shuffle")
        # read the csv file 
        line=csv.reader(open(filename))
        # skip the header ( col names)
        # next(line)
        data=list(line)
        # shuffle the data
        shuffle(data)
        # size of the training set 
        tsize=int(len(data)*ratio)
        # split data into training and validation set
        train=[]
        valid=[]
        data_copy=data.copy()
        # split data to a training and validation set 
        for i in range(tsize):
            train.append(data_copy[i])
        for i in range(tsize,len(data)):
            valid.append(data_copy[i])
        traindf=pd.DataFrame(train)
        # remove the reponse column from the training data 
        mytrain=traindf.drop([0],axis=1)
        train=np.asarray(mytrain) 
        y_train=np.asarray(traindf.iloc[:,0])
        validdf=pd.DataFrame(valid)
        valid=validdf.drop([0],axis=1)
        valid=np.asarray(valid)
        y_valid=np.asarray(validdf.iloc[:,0])
        for i in range(len(valid)):
            for j in range(len(valid[0])):
                valid[i][j]=int(valid[i][j])
        return train, y_train, valid, y_valid
    else:
        line=csv.reader(open(filename))
        # next(line)
        print("this is test, no shuffling !")
        test=list(line)
        testdf=pd.DataFrame(test)
        test=np.asarray(testdf) 
        for i in range(len(test)):
            for j in range(len(test[0])):
                test[i][j]=int(test[i][j])
        return test
def prob_y(response):
    """
    arguments: response vecttor 
    output: probability of the reposne 
    """
    n_obs=len(response)
    sum_1=0
    for i  in range(n_obs):
        if response[i]=='1':
            sum_1=sum_1+1
    prob_y1=sum_1/n_obs
    return prob_y1
def accuracy(y_predicted,y):
    """
    argument: y predicted and the real y 
    output: accuracy rate
    """
    sum_y=0
    for i in range(len(y)):
        if int(y_predicted[i])==int(y[i]):
            sum_y=sum_y+1
    accuracy=sum_y/len(y)
    return accuracy
def cond_prob(train, response,class_y):
    """
    argument: train data, reponse, class of y{ if binary class is {0,1}}
    out: conditionaly probabilty on class y 
    """
    nrow=len(response)
    ncol=len(train[0])
    sum_1=0
    for i  in range(nrow):
        if int(response[i])==class_y:
            sum_1=sum_1+1
    theta_j=np.zeros((ncol,1))
    for i in range(nrow):
        for j in range(ncol):
            if int(train[i][j])==1 and int(response[i])==class_y:
                theta_j[j][0]=theta_j[j][0]+1
    for k in range(ncol):
        theta_j[k]=(theta_j[k]+1)/float(sum_1+2)
    return theta_j
def predict(theta1,theta_j_1,theta_j_0,x):
    """
    Argument: conditional probabilities, class probability 
    Output : prediction using the Naive Bayes rule 
    """
    w_c=log(theta1/(1-theta1))
    w_j_0=0    
    for i in range (len(theta_j_0)):
        w_j_0=w_j_0+log((1-theta_j_1[i][0])/(1-theta_j_0[i][0]))
    w_0=w_c+w_j_0
    w=np.zeros(len(theta_j_1))
    for i in range(len(theta_j_1)):
        w[i]=w[i]+(log(theta_j_1[i][0]/theta_j_0[i][0])-log((1-theta_j_1[i][0])/(1-theta_j_0[i][0])))
    odd=np.dot(x,w)
    y_predicted=np.zeros(len(x))
    for i in range(len(odd)):
        odd[i]=odd[i]+w_0
    for i in range(len(odd)):
        if odd[i]>=0: 
            y_predicted[i]=1
        else:
            y_predicted[i]=0
    for i in range(len(y_predicted)):
        y_predicted[i]=int(y_predicted[i])
    return(y_predicted)
def naive_bayes(x_train,y_train,x_val):
    """
    Argument: training and validation set 
    output: predicted y 
    """
    theta_1=prob_y(y_train)
    theta_j_1=cond_prob(x_train,y_train,1)
    theta_j_0=cond_prob(x_train,y_train,0)
    y_fitted=predict(theta_1,theta_j_1,theta_j_0,x_val)
    return y_fitted
def main ():
    # splits training and validation set
    train, y_train,valid,y_valid=data("training_w_y.csv",0.8)
    # test set 
    test=data("test_bin.csv",test=1)
    # predicted values for test set
    y_test=naive_bayes(train,y_train,test)
    # id list: 
    id_list=np.loadtxt('test_bin_IDs.txt',dtype=int)
    # create a submission csv file : ID Category 
    submission_file=pd.DataFrame({'Id':id_list,'Category':y_test })
    submission_file.to_csv("predictions.csv",index=False)
main()
