import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist

## Task 2.1.1
# Accuracy begin: 0.2649

## Task 2.1.2
# Accuracy L2: 0.19

def predict_L2(X):
    num_test=X.shape[0]
    Lpred=np.zeros(num_test, dtype=Ltr_set.dtype)
    
    for i in range(num_test):
        distances=np.sqrt(np.sum(np.square(Tr_set-X[i,:]),axis=1))
        
        min_index= np.argmin(distances)
        Lpred[i]=Ltr_set[min_index]
    return Lpred

## Task 2.1.3

# The Datatype of distances was uint8, which lead to data overflows and thus nonsense distances

def predict_L1(X):
    num_test=X.shape[0]
    Lpred=np.zeros(num_test, dtype=Ltr_set.dtype)

    X = X.astype(np.float32)
    Tr = Tr_set.astype(np.float32)

    for i in range(num_test):
        distances= np.sum(np.abs(Tr-X[i]),axis=1)

        min_index= np.argmin(distances)
        Lpred[i]=Ltr_set[min_index]
    return Lpred

def predict_L2(X):

    num_test=X.shape[0]
    Lpred=np.zeros(num_test, dtype=Ltr_set.dtype)

    X = X.astype(np.float32)
    Tr = Tr_set.astype(np.float32)
    
    for i in range(num_test):
        distances=np.sqrt(np.sum(np.square(Tr-X[i]), axis=1))
        min_index= np.argmin(distances)
        Lpred[i]=Ltr_set[min_index]

    return Lpred

# New Accuracy: 0.811 for L1 and 0.8294 for L2


## Task 2.1.4

def majority_vote(array):
    # returns the element found most often in the array. 
    # if two elements are equally rare, it chooses the one that has it first in the array
    count = np.zeros(array.size)
    for i in range (0, array.size):
        count[i] = np.count_nonzero(array == array[i])
    return array[np.argmax(count)]


def predict_L1(X, k):
    num_test=X.shape[0]
    Lpred=np.zeros(num_test, dtype=Ltr_set.dtype)

    X = X.astype(np.float32)
    Tr = Tr_set.astype(np.float32)
    
    for i in range(num_test):
        distances= np.sum(np.abs(Tr-X[i]),axis=1)

        min_indexes = np.argpartition(distances, k)[:k]
        Lpred[i]= majority_vote(Ltr_set[min_indexes])
    return Lpred

def predict_L2(X, k):

    num_test=X.shape[0]
    Lpred=np.zeros(num_test, dtype=Ltr_set.dtype)

    X = X.astype(np.float32)
    Tr = Tr_set.astype(np.float32)
    
    for i in range(num_test):
        distances=np.sqrt(np.sum(np.square(Tr-X[i]), axis=1))
        min_indexes = np.argpartition(distances, k)[:k]
        Lpred[i]= majority_vote(Ltr_set[min_indexes])

    return Lpred
