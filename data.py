import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_images(path,categories):
    data_list = list()
    labels=list()          
    for label in  sorted(os.listdir(path)):  
        y=os.path.join(path,label)
        for filename in os.listdir(y):
            pixels = cv2.imread(os.path.join(y,filename))
            pixels=cv2.resize(pixels,(299,299))
            data_list.append(pixels)
            labels.append(categories[label])
    data_list=np.array(data_list,dtype='float32')/255.0
    labels=keras.utils.to_categorical(labels, 36)
    return data_list,labels

def get_dataset_partitions(ds, ds_size,val_split=0.2, shuffle=True, shuffle_size=1000):
    train_split = 1- val_split
    X,Y=ds
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=val_split, random_state=42)
    return (X_train,y_train),(X_test, y_test)

def data_builder(args):
    train_path=args.train_path
    categories = {
        "0":0,
        "1":1,
        "2":2,
        "3":3,
        "4":4,
        "5":5,
        "6":6,
        "7":7,
        "8":8,
        "9":9,
        "a":10,
        "b":11,
        "c":12,
        "d":13,
        "e":14,
        "f":15,
        "g":16,
        "h":17,
        "i":18,
        "j":19,
        "k":20,
        "l":21,
        "m":22,
        "n":23,
        "o":24,
        "p":25,
        "q":26,
        "r":27,
        "s":28,
        "t":29,
        "u":30,
        "v":31 ,
        "w":32,
        "x":33,
        "y":34,
        "z":35,
    }
    train_X,train_Y=load_images(train_path,categories)
    if args.mode=='test':
        return train_X,train_Y
    if args.val_path=='None':
        size_=train_X.shape[0]
        (train_X,train_Y),(test_x,test_y)=get_dataset_partitions(ds=(train_X,train_Y),ds_size=size_,val_split=args.val_split)
    else:
        (test_x,test_y)=load_images(args.val_path,categories)    

    return (train_X,train_Y),(test_x,test_y)
    