# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:16:05 2020

@author: jongkim
"""


import numpy as np
from keras import backend as K
from itertools import product



# 20201103 
def Custom_weighted_categorical_crossentropy( y_true, y_pred, nWeight = 1 ):
    '''
    m_weights=np.array([[0.01,  0.02,  0.04,  0.16],
                    [0.02,  0.01,  0.02,  0.04],
                    [0.04,  0.02,  0.01,  0.02],
                    [0.16,  0.04,  0.02,  0.01]])
    '''
    nWeight = 6  # used this until 20211231
    nWeight = 8  # used this from 20220408 to retrain DenseNet121" --ep 100  --tl 0 --ex "86161 and  ResNet152"  --tl 0    --ex "861610"  --ep 100
    
    print( "Wrighted_Categorial CrossEmtropy: {} ".format(nWeight) )
    
    if( nWeight == 1 ):
        weights=np.array([[1.0,  1.0,  1.0, 1.0],
                          [1.0,  1.0,  1.0, 1.0],
                          [1.0,  1.0,  1.0, 1.0],
                          [1.0,  1.0,  1.0, 1.0]])
    elif( nWeight == 2 ):
        weights=np.array([[0.1,  0.2,  0.4, 1.6],
                          [0.2,  0.1,  0.2, 0.4],
                          [0.4,  0.2,  0.1, 0.2],
                          [1.6,  0.4,  0.2, 0.1]])   
    
    elif( nWeight == 3 ):
        # (i-j)*(i-j)/(n-1)*(n-1) Quadric weight  n=4
        weights=np.array([[0.0000, 0.1111, 0.4444, 1.0000 ],
                          [0.1111, 0.0000, 0.1111, 0.4444 ],
                          [0.4444, 0.1111, 0.0000, 0.1111 ],
                          [1.0000, 0.4444, 0.1111, 0.0000 ]])   
    
    elif( nWeight == 4 ):
        # (i-j)*(i-j)/(n-1)*(n-1)  Quadric weight n=5
        weights=np.array([[0.0000, 0.0625, 0.2500, 0.5625 ],
                          [0.0625, 0.0000, 0.0625, 0.2500 ],
                          [0.2500, 0.0625, 0.0000, 0.0625 ],
                          [0.5625, 0.2500, 0.0625, 0.0000 ]])   
    
    elif( nWeight == 5 ):
        # (i-j)*(i-j)/(n-1)*(n-1)  Quadric weight n=5 + two class concepts
        weights=np.array([[0.0100, 0.0625, 0.2500, 0.2500 ],
                          [0.0625, 0.0100, 0.2500, 0.2500 ],
                          [0.2500, 0.2500, 0.0100, 0.0625 ],
                          [0.2500, 0.2500, 0.0625, 0.0100 ]])   
    
    elif( nWeight == 6 ):
        # Use Total Training error (error/total error for each class)
        weights=np.array([[0.1000, 0.2148, 0.6542, 0.1311 ],
                          [0.3915, 0.1000, 0.0439, 0.5646 ],
                          [0.6418, 0.0278, 0.1000, 0.3304 ],
                          [0.0701, 0.3617, 0.5682, 0.1000 ]])    

    elif( nWeight == 7 ):
        # Use Total Training error (error/total error for all classes)
        weights=np.array([[0.0000, 0.0718, 0.2185, 0.0438 ],
                          [0.0604, 0.0000, 0.0068, 0.0872 ],
                          [0.2229, 0.0097, 0.0000, 0.1147 ],
                          [0.0115, 0.0594, 0.0933, 0.0000 ]])    

    elif( nWeight == 8 ):
        # Use Total Training error (error/total error for each class)
        weights=np.array([[0.0100, 0.2148, 0.6542, 0.1311 ],
                          [0.3915, 0.0100, 0.0439, 0.5646 ],
                          [0.6418, 0.0278, 0.0100, 0.3304 ],
                          [0.0701, 0.3617, 0.5682, 0.0100 ]])    
        
    else:
        weights=np.array([[1.0,  1.0,  1.0, 1.0],
                          [1.0,  1.0,  1.0, 1.0],
                          [1.0,  1.0,  1.0, 1.0],
                          [1.0,  1.0,  1.0, 1.0]])
    

    
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    #y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p] ,K.floatx()) * K.cast(y_true[:, c_t],K.floatx()))
    out_loss = K.categorical_crossentropy(y_true, y_pred) * final_mask 
    
    return  out_loss 



#def Combined_categorical_crossentropy( y_true, y_pred, nWeight = 1, dAlpha = 0.5 ):
def Combined_categorical_crossentropy( y_true, y_pred  ):
    
    nType = 1  # 20220410 for ResNet152-Ca-Cl0-Ro224-Co224-Ch3-Vs20-Do50-Au0-Pr0-Tl0-sgd-Ep100-861610F-w6C10050E5F-Ensemble
    nType = 0  # 20220410 for DenseNet121-Ca-Cl0-Ro224-Co224-Ch3-Vs20-Do50-Au0-Pr0-Tl0-sgd-Ep100-86161F-w6C7525E5F-Ensemble
    
    if( nType == 0 ):
        dAlpha = 0.75    
        print( '\n === Combined_categorical_crossentropy dAlpha:  {}-{}'.format(dAlpha, (1.0-dAlpha)) )    
        return dAlpha*K.categorical_crossentropy(y_true, y_pred) + (1.0-dAlpha)*Custom_weighted_categorical_crossentropy( y_true, y_pred)
    else:    
        dAlpha = 0.50
        print( '\n === Combined_categorical_crossentropy dAlpha:  {}'.format(dAlpha) )    
        return K.categorical_crossentropy(y_true, y_pred) + dAlpha*Custom_weighted_categorical_crossentropy( y_true, y_pred)




"""
Here is a dice loss for keras which is smoothed to approximate a linear (L1) loss.
It ranges from 1 to 0 (no error), and returns results similar to binary crossentropy
"""
'''
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true /= 255.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
'''

def pixelwise_l2_loss(y_true, y_pred):
    y_true /= 255.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return K.mean(K.square(y_true_f - y_pred_f))

def pixelwise_binary_ce(y_true, y_pred):
    y_true /= 255.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return K.mean(K.binary_crossentropy(y_pred_f, y_true_f))    


'''
def dice_coef(y_true, y_pred, smooth=1, bRoundY=True):

    #Dice = (2*|X & Y|)/ (|X|+ |Y|)
    #      =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    #ref: https://arxiv.org/pdf/1606.04797v1.pdf

    if (bRoundY):
        y_pred = K.round(y_pred)  # Round to 0 or 1

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
    
def dice_coef_loss(y_true, y_pred, smooth=1, bRoundY=True):

    #Dice = (2*|X & Y|)/ (|X|+ |Y|)
    #     =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    #ref: https://arxiv.org/pdf/1606.04797v1.pdf

    if (bRoundY):
        y_pred = K.round(y_pred)  # Round to 0 or 1

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    dc = (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
    return (1-dc)    
    
'''

    

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_coef_loss(y_true, y_pred):
    print("dice loss")
    return 1.0 - dice_coef(y_true, y_pred)



#20190819 check if it works
def bce_dice_loss_for_UNet(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


''' Original method
def jaccard_index(y_true, y_pred, smooth=1, bRoundY=True):  # original: smooth=100
    # """
    # Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
    #        = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    #
    # The jaccard distance loss is usefull for unbalanced datasets. This has been
    # shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    # gradient.
    # 
    # Ref: https://en.wikipedia.org/wiki/Jaccard_index
    # 
    # @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    # @author: wassname
    # """
    if (bRoundY):
        y_pred = K.round(y_pred)  # Round to 0 or 1

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    
    sum_ = K.sum(K.abs(y_true_f) + K.abs(y_pred_f))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac
'''

''' backup method
def jaccard_index(y_true, y_pred, smooth=1, bRoundY=True):  # original: smooth=100
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    if (bRoundY):
        y_pred = K.round(y_pred)  # Round to 0 or 1
        
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac
'''


def jaccard_index(y_true, y_pred, smooth=1, bRoundY=True):  # original: smooth=100

    # Jaccard Index =TP/(TP + FP + FN)
    
    ddice_coef = dice_coef(y_true, y_pred)
    
    return ddice_coef/(2.0 - ddice_coef)


def jaccard_index_loss(y_true, y_pred, smooth=1, bRoundY=True):
    print("jaccard_index_loss")
    return 1-jaccard_index(y_true, y_pred)


def combined_binary_jaccard_loss(y_true, y_pred, dWeight = 0.5 ):
    #return dWeight * binary_crossentropy(y_true, y_pred) + (1.0-dWeight)*jaccard_index_loss(y_true, y_pred)
    return (binary_crossentropy(y_true, y_pred) + jaccard_index_loss(y_true, y_pred))/2.0


def combined_binary_jaccard(y_true, y_pred, dWeight = 0.5 ):
    return 1.0 - combined_binary_jaccard_loss(y_true, y_pred, dWeight)


'''
def combined_binary_dice(y_true, y_pred, dWeight= 0.5 ):
    return dWeight * binary_crossentropy(y_true, y_pred) + (1.0-dWeight)*dice_coef(y_true, y_pred)


def combined_binary_dice_loss(y_true, y_pred, dWeight = 0.5 ):
    return 1.0 - combined_binary_dice(y_true, y_pred, dWeight)
'''        


def combined_binary_dice_loss(y_true, y_pred, dWeight = 0.5):
    #return dWeight*binary_crossentropy(y_true, K.sigmoid(y_pred)) + (1.0-dWeight)*dice_coef_loss(y_true, K.sigmoid(y_pred))
    return (binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true,y_pred))/2.0


def combined_binary_dice(y_true, y_pred, dWeight = 0.5 ):
    return 1.0 - combined_binary_dice_loss(y_true, y_pred, dWeight)



def combined_binary_logdice_loss(y_true, y_pred, dWeight = 0.5):
    return dWeight*binary_crossentropy(y_true, y_pred) - (1.0-dWeight)*K.log(1. - dice_coef_loss(y_true, y_pred))

def combined_binary_logdice(y_true, y_pred, dWeight = 0.5):
    return 1.0 - combined_binary_logdice_loss(y_true, y_pred, dWeight)


'''
def jaccard_index_loss(y_true, y_pred, smooth=1, bRoundY=True):  # original: smooth=100
    # """
    # Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
    #        = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    #
    # The jaccard distance loss is usefull for unbalanced datasets. This has been
    # shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    # gradient.
    #
    # Ref: https://en.wikipedia.org/wiki/Jaccard_index
    #
    # @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    # @author: wassname
    # """
    if (bRoundY):
        y_pred = K.round(y_pred)  # Round to 0 or 1
        
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    
    #jw return (1 - jac) * smooth
    return (1 - jac)
'''

def combined_dice_ce_loss(y_true, y_pred, weight, smooth=1):
    """
    Combined Dice and Binary Cross Entropy Loss
    """
    return weight*dice_coef_loss(y_true, y_pred,smooth) + (1-weight)*K.losses.binary_crossentropy(y_true, y_pred)
            

def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)  #=> return a (y_true dimension,); if input=(10,5), output=(10,)
    #or return K.mean(K.equal(y_true, K.round(y_pred)))  => return a scalar


def categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())
    #or return K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))
    #K.argmax(y_true, axis=-1)  => if input (10,5), output = (10,), index of max value in the last axis
    #K.cast(input, K.floatx()) => change input to K.floatx(),, i.e., int to float


def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


def categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)


def sparse_categorical_crossentropy(y_true, y_pred):
    return K.sparse_categorical_crossentropy(y_true, y_pred)

'''
def class_weighted_pixelwise_crossentropy(y_true, y_pred):
    #output = tf.clip_by_value(output, 10e-8, 1.-10e-8)
    weights = [0.8, 0.2]
     #return -tf.reduce_sum(target * weights * tf.log(output))
    return ( 1 - K.sum(y_true * weights * y_pred)
'''    



#20190819 from https://forums.fast.ai/t/unbalanced-classes-in-image-segmentation/18289
import tensorflow as tf

def weighted_categorical_crossentropy(weights):
    """ weighted_categorical_crossentropy

        Args:
            * weights<ktensor|nparray|list>: crossentropy weights
        Returns:
            * weighted categorical crossentropy function
    """
    if isinstance(weights,list) or isinstance(np.ndarray):
        weights=K.variable(weights)

    def loss(y_true,y_pred,from_logits=False):
        if not from_logits:
            y_pred /= tf.reduce_sum(y_pred, len(y_pred.get_shape()) - 1, True)
            _epsilon = tf.convert_to_tensor(K.epsilon(), dtype=y_pred.dtype.base_dtype)
            y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
            weighted_losses = y_true * tf.log(y_pred) * weights
            return - tf.reduce_sum(weighted_losses,len(y_pred.get_shape()) - 1)
        else:
            raise ValueError('WeightedCategoricalCrossentropy: not valid with logits')
    return loss



def weighted_cross_entropy(beta):
    
    # BCE (G, O) = -((beta)*G*log(O)) + (1-G)*log(1.0 - O))
    def convert_to_logits(y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon() )
        
        return tf.log(y_pred/(1.0 - y_pred))
    
    def loss( y_true, y_pred ):
        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits( logits = y_pred, targets=y_true, pos_weight = beta)
        
        return tf.reduce_mean(loss*(1.0 - beta))
    
    return loss



def balanced_cross_entropy(beta):
    
    # BCE (G, O) = -((beta)*G*log(O)) + (1-beta)(1-G)*log(1.0 - O))
    def convert_to_logits(y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon() )
        
        return tf.log(y_pred/(1.0 - y_pred))
    
    def loss( y_true, y_pred ):
        y_pred = convert_to_logits(y_pred)
        pos_weight = beta/(1.0 - beta)
        loss = tf.nn.weighted_cross_entropy_with_logits( logits = y_pred, targets=y_true, pos_weight = pos_weight)
        
        return tf.reduce_mean(loss*(1.0 - beta))
    
    return loss


def Test():
    
    #y_true=[[1,0,0],[0,1,0], [0,0,1]]
    #y_pred=[[1,0,0],[0,1,0], [1,0,0]]

    y_true = np.zeros((4,3))
    y_true[0,0]=1.0
    y_true[1,1]=1.0
    y_true[2,2]=1.0
    y_true[3,2]=1.0

    y_pred = np.zeros((4,3))
    y_pred[0,0]=1.0
    y_pred[1,1]=1.0
    y_pred[2,2]=1.0
    y_pred[3,0]=1.0
    
    
    Out = weighted_categorical_crossentropy(y_true, y_pred)
    
    print(Out)
    
if __name__ == "__main__": 
            
    Test()