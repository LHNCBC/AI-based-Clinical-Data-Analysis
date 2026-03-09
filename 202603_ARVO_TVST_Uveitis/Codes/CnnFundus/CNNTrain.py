#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 12:13:49 2025

@author: jongkim
"""

############Load libraries#####################################################

import os
import sys
import numpy as np



from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as VGG16_preprocess_input

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as VGG19_preprocess_input 

from keras.applications.densenet import DenseNet121 
from keras.applications.densenet import preprocess_input as DenseNet121_preprocess_input

from keras.applications.densenet import DenseNet169 
from keras.applications.densenet import preprocess_input as DenseNet169_preprocess_input

from keras.applications.densenet import DenseNet201 
from keras.applications.densenet import preprocess_input as DenseNet201_preprocess_input

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as InceptionV3_preprocess_input 

from keras.applications.inception_resnet_v2 import InceptionResNetV2  #?
from keras.applications.inception_resnet_v2 import preprocess_input as InceptionResNetV2_preprocess_input 

from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input as MobileNet_preprocess_input

from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input as MobileNetV2_preprocess_input #?

from keras.applications.nasnet import NASNetMobile #224 by 224
from keras.applications.nasnet import preprocess_input as NASNetMobile_preprocess_input

from keras.applications.nasnet import NASNetLarge # 331 by 331
from keras.applications.nasnet import preprocess_input as NASNetLarge_preprocess_input

from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input as ResNet50_preprocess_input

from keras.applications.resnet import ResNet101
from keras.applications.resnet import preprocess_input as ResNet101_preprocess_input

from keras.applications.resnet import ResNet152
from keras.applications.resnet import preprocess_input as ResNet152_preprocess_input

from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.resnet_v2  import preprocess_input as ResNet50V2_preprocess_input

from keras.applications.resnet_v2 import ResNet101V2

from keras.applications.resnet_v2 import preprocess_input as ResNet101V2_preprocess_input

from keras.applications.resnet_v2 import ResNet152V2

from keras.applications.resnet_v2 import preprocess_input as ResNet152V2_preprocess_input

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as Xception_preprocess_input

#from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model, load_model
#from keras.models import Sequential

from keras.layers import Input, Dense, Dropout
#from keras.layers import Lambda, Flatten, Activation, BatchNormalization
#from keras.layers import Conv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import SGD, Adam, Nadam, RMSprop
#from keras import backend as K

#from keras import applications
#from keras import optimizers
from keras.callbacks import ModelCheckpoint

import cv2  #opencv_python-3.4.1-cp35-cp35m-win_amd64.whl
#import numpy as np

#from datetime import datetime 
import datetime
#from keras.utils import plot_model
import matplotlib.pyplot as plt
#import json
import shutil
import random

from keras.utils import np_utils

from keras.utils import multi_gpu_model

from optparse import OptionParser

import config

from ImageAugmentationRandom import  GetAugmentedImagesRegular
#from ImageAugmentationRandom import  GetAugmentedImagesRandom
#from ImageAugmentationRandom import  GetAugmentedImagesRegularOneInputRandom, GetAugmentedImagesRegularOneInput
from GenerateClahe import ClaheOperatorColor3D
from LossFunctions import Custom_weighted_categorical_crossentropy
#from LossFunctions import Combined_categorical_crossentropy
#20240320 to add "pyramid_pooling_module" after backbone model to make MAPNetClassifier
#from ModelJW_MAPNet import pyramid_pooling_module

# to generate heatmaps for each output
#from Visualization_Library_V3_1 import Load_Pretrained_Model, Heatmap_Function_Call, Generate_Heatmap_Image          
#from Visualization_Library_V3_1 import Generate_Heatmap_Image          


def SetDirectories( cfg ):
    
    parser = OptionParser()

    parser.add_option("--sl", dest="use_slurm",             type=str,    help="use slurm to train (True/False)", default='0')
    parser.add_option("--gn", dest="gpu_to_use",            type=int,    help="choose a gpu to use", default=0)
    parser.add_option("--op", dest="operation",             type=str,    help="traintest, test", default='traintest')
    parser.add_option("--ct", dest="CNN_Type",              type=str,    help="vgg16,vgg19, resnet50, cvption, inceptionV3, inceptionResNetV2, MobileNet", default='VGG16')
    parser.add_option("--pp", dest="PyramidPoolModule",     type=int,    help="Add pyramid pool module after backbone module", default=0)
    parser.add_option("--ep", dest="num_epochs",            type=int,    help="number of epochs to train", default =  100 )
    parser.add_option("--nc", dest="num_classes",           type=int,    help="number of classes", default = 2 )
    parser.add_option("--ca", dest="class_assignment",      type=str,    help="class assignment", default = '')
    
    parser.add_option("--co", dest="cnn_optimizer",         type=str,    help="Unet Optimizer (Adam, Nadam, RMSprop)",  default= 'SGD') 
    
    parser.add_option("--wd",  dest="weights_directory",      type=str,    help="weights for CNN directory", default= '')
    parser.add_option("--wd2", dest="weights_directory2",     type=str,    help="weights for CNN2 directory", default= '')
    parser.add_option("--wd3", dest="weights_directory3",     type=str,    help="weights for CNN3 directory", default= '')
    parser.add_option("--vm",  dest="voting_method",          type=str,    help="voting methods (hard, soft, or hybrid ", default= 'hybrid')   

    parser.add_option("--id", dest="train_image_directory", type=str,    help="train image directory", default= '/retina/ARED/Uveitis/Fundus/Images/All')
    parser.add_option("--if", dest="train_image_file",      type=str,    help="train image file", default= '/retina/ARED/Uveitis/Fundus/TrainTestFile/TrainSets5/Set501Train.txt')
    parser.add_option("--td", dest="test_image_directory",  type=str,    help="test image directory", default= '/retina/ARED/Uveitis/Fundus/Images/All')
    parser.add_option("--tf", dest="test_image_file",       type=str,    help="test image file", default= '/retina/ARED/Uveitis/Fundus/TrainTestFile/TrainSets5/Set501Test.txt')
    parser.add_option("--od", dest="output_directory",      type=str,    help="output directory to save", default= '/retina/ARED/Jongwoo/UveitisFundus/TrainResults')
    parser.add_option("--vr", dest="validation_rate",       type=float,  help="validation split rate", default= 0.2 )
    parser.add_option("--tvg", dest="train_val_data_generation", type=int,  help="generate train validation dataset based on vr", default = 0 )
    parser.add_option("--dr", dest="dropout_rate",          type=float,  help="dropout rate", default= 0.5 )
    parser.add_option("--bs", dest="batch_size",            type=int,    help="batch size", default= 1 )
    parser.add_option("--au", dest="augmentation_number",   type=int,    help="augmentation number", default = 0)
    parser.add_option("--pr", dest="preprocess_input",      type=int,    help="preprocess input", default = 0 )
    parser.add_option("--tl", dest="transfer_learning",     type=int,    help="transfer learning", default = 0)
    parser.add_option("--ex", dest="filename_extension",    type=str,    help="filename extension", default = '')
    
    parser.add_option("--cw",  dest="class_weight",                  type=int,    help="weight  for balancing data in each class ", default= 1)        

    parser.add_option("--cl",  dest="Clahe_Operation",               type=int,    help="CLAHE operation", default= 0)
    parser.add_option("--it",  dest="Image_Channel_Type",            type=str,    help="Image Channel Type", default= 'BGR')    
    parser.add_option("--in",  dest="Image_intensity_normalization", type=int,    help="normalization of image intensity ", default= 0)
    parser.add_option("--in1", dest="Image_intensity_normalization_Max1", type=int,    help="normalization of image intensity having max = 1", default= 0)
    parser.add_option("--inc3",  dest="Image_normalization_UseRGB",    type=int,    help="Use RGB or Global for image normalization", default= 1)

    parser.add_option("--bn",  dest="cnn_batchNormalization",        type=int,    help="use batch normalization (1=true/0=false)", default= 0)    

    parser.add_option("--ir", dest="img_row",               type=int,   help="Image Row", default=0 )
    parser.add_option("--ic", dest="img_column",            type=int,   help="Image Column", default=0 )
    parser.add_option("--ich",dest="img_channel",           type=int,   help="Image Channel", default=0 )
    parser.add_option("--cn", dest="sClassLabelNames",      type=str,   help="Class label names", default='' ) #default = 'NLK, LK'    
    parser.add_option("--pa", dest="nPrintAllTestOutputs",   type=int,   help="Print test all outputs", default=0 ) #default = (0: not print all test outputs)(1: print all test outputs)
          
    parser.add_option("--hm", dest="sHeatMapMethod",        type=str,   help="HeatMap method", default='' )    
    parser.add_option("--ht", dest="fHeatMapThreshold",    type=float,   help="HeatMap Threshold", default=0 ) #defalut=0.25
    parser.add_option("--hc", dest="nHeatMapConvLayerLocation",   type=int,   help="Location of last Convolutional layer", default=0 ) #default = -7 (ouput-dropout-dense-droupout-dense-globalaverage-pooling)
    parser.add_option("--hd", dest="nHeatMapNbrOfDenseLayers",   type=int,   help="Number of Dense layers", default=2 ) #default = -7 (ouput-dropout-dense-droupout-dense-globalaverage-pooling)

        
    (options, args) = parser.parse_args()
    
    
           
    if(options.use_slurm == '1'):
        cfg.bUseSlurm = '1'
    elif( options.use_slurm == 'GPUs' or options.use_slurm == 'gpus'):
        cfg.bUseSlurm = str(options.use_slurm)
    elif( options.use_slurm == 'GPUs2' or options.use_slurm == 'gpus2'):
        cfg.bUseSlurm = str(options.use_slurm)
        
    if(options.use_slurm != '1'):
        if( int(options.gpu_to_use) == 0 ):
            os.environ["CUDA_VISIBLE_DEVICES"]="0"
        elif( int(options.gpu_to_use) == 1 ):
            os.environ["CUDA_VISIBLE_DEVICES"]="1"
        elif( int(options.gpu_to_use) == 2 ):
            os.environ["CUDA_VISIBLE_DEVICES"]="2"
        elif( int(options.gpu_to_use) == 3 ):
            os.environ["CUDA_VISIBLE_DEVICES"]="3"
        elif( int(options.gpu_to_use) == 4 ):
            os.environ["CUDA_VISIBLE_DEVICES"]="4"
        elif( int(options.gpu_to_use) == 5 ):
            os.environ["CUDA_VISIBLE_DEVICES"]="5"
        elif( int(options.gpu_to_use) == 6 ):
            os.environ["CUDA_VISIBLE_DEVICES"]="6"
        elif( int(options.gpu_to_use) == 7 ):
            os.environ["CUDA_VISIBLE_DEVICES"]="7"
        elif( int(options.gpu_to_use) == 10 ):
            os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
        elif( int(options.gpu_to_use) == 23 ):
            os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
        elif( int(options.gpu_to_use) == 45 ):
            os.environ["CUDA_VISIBLE_DEVICES"]="4,5"
        elif( int(options.gpu_to_use) == 67 ):
            os.environ["CUDA_VISIBLE_DEVICES"]="6,7"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"]="0"
 


    if( len(str(options.operation).strip()) > 0):
        cfg.sOperation = str(options.operation)

    cfg.sCNNType = str(options.CNN_Type)

    #20240321 add pyramid pooling module
    cfg.nPyramidPoolModule = int(options.PyramidPoolModule)

    if( int(options.transfer_learning) >= 0):
        cfg.nTransferLearning = int(options.transfer_learning)


    if( int(options.num_epochs) > 0):
        cfg.nEpochs = int(options.num_epochs)
        
    if( int(options.num_classes) > 0):
        cfg.nNbrOfClasses = int(options.num_classes)
        

    if( len(options.sClassLabelNames) > 0 ):
        cfg.sClassLabelNames = str(options.sClassLabelNames).split(",")        
        
    if( len(str(options.class_assignment).strip()) > 0):
        cfg.sClassAssignment = str(options.class_assignment)

        
 
    if( int(options.preprocess_input) >= 0):
        cfg.nPreprocessInput = int(options.preprocess_input)


       
    if( int(options.augmentation_number) > 0 ):
        cfg.nAugmentationNumber = int(options.augmentation_number)
        
    #20201025    
    if( len(str(options.Clahe_Operation).strip()) > 0 ):
        cfg.nUseClaheForInputImage = int(options.Clahe_Operation)
        
    #20201025    
    if( len(str(options.Image_Channel_Type).strip()) > 0 ):
        cfg.sImageChannelType = str(options.Image_Channel_Type)
        
        

    if( int(options.Image_intensity_normalization_Max1)  > 0 ): #normalize image intensity by mean and standard deviation
        cfg.nImageIntensityNormalizationToMax1 = int(options.Image_intensity_normalization_Max1)
             
    if( int(options.Image_intensity_normalization) > 0 ): #normalize image intensity by mean and standard deviation
        cfg.nImageIntensityNormalization = int(options.Image_intensity_normalization)
          
    if( int(options.Image_normalization_UseRGB) > 0 ): #normalize image intensity for each RGB channel or globally (R+G+B)/3
        cfg.nImageIntensityNormalizationUseRGBChannel = int(options.Image_normalization_UseRGB)

      
    
    if( options.weights_directory != ''):
        cfg.sDirectoryForWeights = options.weights_directory
  
    if( options.weights_directory != ''):
        cfg.sDirectoryForWeights2 = options.weights_directory2

    if( options.weights_directory != ''):
        cfg.sDirectoryForWeights3 = options.weights_directory3
      
    if( options.output_directory != ''):
        cfg.sDirectoryForOutputRoot = options.output_directory
    
    if( options.voting_method != ''):
        cfg.sVotingMethod = options.voting_method
    
    
    if( options.train_image_directory != ''):
        cfg.sDirectoryForTrainImages = options.train_image_directory
        
    cfg.sFilePathForTrainImages = options.train_image_file
    
    
    cfg.sFilePathForTrainImagesForTest = options.train_image_file
    cfg.sFilePathForTrainImagesForTest = str(cfg.sFilePathForTrainImagesForTest).replace('Train','Test')
    
    
    if( options.test_image_directory != ''):
        cfg.sDirectoryForTestImages = options.test_image_directory
        
    cfg.sFilePathForTestImages = options.test_image_file

    cfg.sDirectoryForTrainingResultsRoot = str(options.output_directory)
       
    if( cfg.bUseSlurm == True or sys.platform == 'linux' ):
        cfg.nNbrOfGPUs = 1
   
    if( int(options.batch_size) > 0):
        cfg.nBatchSize = int(options.batch_size)
    
    if( float(options.dropout_rate) > 0.0 ):
        cfg.dDropoutRatio = float(options.dropout_rate)

    if( float(options.validation_rate) > 0.0 ):
        cfg.dValidationSplit = float(options.validation_rate)
    
    if( float(options.train_val_data_generation) > 0):
        cfg.dTrainValidDataGeneration = float(options.train_val_data_generation)

    if( options.filename_extension != ''):
        cfg.sFilenameExtension = options.filename_extension

    #20230522 updated
    if( int(options.img_row) > 0 ):
        cfg.nImageRows = int(options.img_row) 
    else:
        cfg.nImageRows = 512
        
    if( int(options.img_column) > 0 ):  
        cfg.nImageColumns = int(options.img_column)
    else:        
        cfg.nImageColumns = 512
        
    if( int(options.img_channel) > 0 ):         
        cfg.nImageChannels = int(options.img_channel)
    else:
        cfg.nImageChannels = 3
      
    #Heatmap parameters
    if( len(str(options.sHeatMapMethod).strip()) > 0):
        cfg.sHeatMapMethod = str(options.sHeatMapMethod)

    if( int(options.fHeatMapThreshold) >= 0):
        cfg.fHeatMapThreshold = float(options.fHeatMapThreshold)
     
    if( int(options.nHeatMapConvLayerLocation) > 0 ):
        cfg.nHeatMapConvLayerLocation = int(options.nHeatMapConvLayerLocation)    
        
    if( int(options.nHeatMapNbrOfDenseLayers) > 0 ):
        cfg.nHeatMapNbrOfDenseLayers = int(options.nHeatMapNbrOfDenseLayers)    
   
    #20230447 to print all test outputs
    if( int(options.nPrintAllTestOutputs) > 0 ):
        cfg.nPrintAllTestOutputs = int(options.nPrintAllTestOutputs)
        

    print('== options -> cfg === }')
    print('options.use_slurm:                {}  ->  cfg.bUseSlurm:                         {}'.format(options.use_slurm, cfg.bUseSlurm))
    print('options.operation:                {}  ->  cfg.sOperation:                        {}'.format(options.operation, cfg.sOperation))
    print('options.CNN_Type:                 {}  ->  cfg.sCNNType:                          {}'.format(options.CNN_Type, cfg.sCNNType))
    #print('options.PyramidPoolModule:        {}  ->  cfg.nPyramidPoolModule:                {}'.format(options.PyramidPoolModule, cfg.nPyramidPoolModule))
    print('options.transfer_learning:        {}  ->  cfg.nTransferLearning:                 {}'.format(options.transfer_learning, cfg.nTransferLearning ))
    print('options.train_image_directory:    {}  ->  cfg.sDirectoryForTrainImages:          {}'.format(options.train_image_directory, cfg.sDirectoryForTrainImages))
    print('options.train_image_file:         {}  ->  cfg.sFilePathForTrainImages:           {}'.format(options.train_image_file, cfg.sFilePathForTrainImages))
    print('options.test_image_directory:     {}  ->  cfg.sDirectoryForTestImages:           {}'.format(options.test_image_directory, cfg.sDirectoryForTestImages))
    print('options.test_image_file:          {}  ->  cfg.sFilePathForTestImages:            {}'.format(options.test_image_file, cfg.sFilePathForTestImages))
    print('options.train_image_file_for_test:{}  ->  cfg.sFilePathForTrainImagesForTest:    {}'.format('', cfg.sFilePathForTrainImagesForTest))
    print('options.train_image_file:         {}  ->  cfg.sFilePathForTrainImagesAll:        {}'.format('', cfg.sFilePathForTrainImagesAll))
    print('options.num_classes:              {}  ->  cfg.nNbrOfClasses:                     {}'.format(options.num_classes, cfg.nNbrOfClasses))
    print('options.cnn_optimizer:            {}  ->  cfg.sOptimizer:                        {}'.format(options.cnn_optimizer, cfg.sOptimizer))
    print('options.weights_directory:        {}  ->  cfg.sDirectoryForWeights:              {}'.format(options.weights_directory, cfg.sDirectoryForWeights))
    print('options.weights_directory2:       {}  ->  cfg.sDirectoryForWeights2:             {}'.format(options.weights_directory2, cfg.sDirectoryForWeights3))
    print('options.weights_directory3:       {}  ->  cfg.sDirectoryForWeights3:             {}'.format(options.weights_directory2, cfg.sDirectoryForWeights3))
    print('options.voting_method:            {}  ->  cfg.voting_method:                     {}'.format(options.voting_method, cfg.sVotingMethod))
    print('options.output_directory:         {}  ->  cfg.sDirectoryForOutputRoot:           {}'.format(options.output_directory, cfg.sDirectoryForOutputRoot))
    print('options.output_directory:         {}  ->  cfg.sDirectoryForTrainingResultsRoot:  {}'.format(options.output_directory, cfg.sDirectoryForTrainingResultsRoot))
    print('options.num_epochs:               {}  ->  cfg.nEpochs:                           {}'.format(options.num_epochs, cfg.nEpochs))
    print('options.batch_size:               {}  ->  cfg.nBatchSize:                        {}'.format(options.batch_size, cfg.nBatchSize))
    print('options.dropout_rate:             {}  ->  cfg.dDropoutRatio:                     {}'.format(options.dropout_rate, cfg.dDropoutRatio))
    print('options.validation_rate:          {}  ->  cfg.dValidationSplit:                  {}'.format(options.validation_rate, cfg.dValidationSplit))
    print('options.train_val_data_generation:{}  ->  cfg.dTrainValidDataGeneration:         {}'.format(options.train_val_data_generation, cfg.dTrainValidDataGeneration))
    print('options.augmentation_number:      {}  ->  cfg.nAugmentationNumber:               {}'.format(options.augmentation_number, cfg.nAugmentationNumber))
    print('options.filename_extension:       {}  ->  cfg.sFilenameExtension:                {}'.format(options.filename_extension, cfg.sFilenameExtension))
    print('options.Clahe_Operation:          {}  ->  cfg.Clahe_Operation:                   {}'.format(options.Clahe_Operation, cfg.nUseClaheForInputImage))
    print('options.Image_Channel_Type:       {}  ->  cfg.sImageChannelType:                 {}'.format(options.Image_Channel_Type, cfg.sImageChannelType))
    print('options.preprocess_input:         {}  ->  cfg.nPreprocessInput:                  {}'.format(options.preprocess_input, cfg.nPreprocessInput))
    print('options.Image_intensity_normalization:         {}  ->  cfg.nImageIntensityNormalization:                 {}'.format(options.Image_intensity_normalization, cfg.nImageIntensityNormalization))
    print('options.Image_intensity_normalization_Max1:    {}  ->  cfg.nImageIntensityNormalizationToMax1:           {}'.format(options.Image_intensity_normalization_Max1, cfg.nImageIntensityNormalizationToMax1))
    print('options.Image_normalization_UseRGB:            {}  ->  cfg.nImageIntensityNormalizationUseRGBChannel:    {}'.format(options.Image_normalization_UseRGB, cfg.nImageIntensityNormalizationUseRGBChannel))
    print('options.sClassLabelNames:        {}  ->  cfg.sClassLabelNames:                   {}'.format(options.sClassLabelNames, cfg.sClassLabelNames))
    print('options.sHeatMapMethod:              {}  ->  cfg.sHeatMapMethod:                 {}'.format(options.sHeatMapMethod, cfg.sHeatMapMethod))
    print('options.fHeatMapThreshold:           {}  ->  cfg.fHeatMapThreshold:              {}'.format(options.fHeatMapThreshold, cfg.fHeatMapThreshold))
    print('options.nHeatMapConvLayerLocation:   {}  ->  cfg.nHeatMapConvLayerLocation:      {}'.format(options.nHeatMapConvLayerLocation, cfg.nHeatMapConvLayerLocation))
    print('options.nHeatMapNbrOfDenseLayers:    {}  ->  cfg.nHeatMapNbrOfDenseLayers:       {}'.format(options.nHeatMapNbrOfDenseLayers, cfg.nHeatMapNbrOfDenseLayers))

    print('options.nPrintAllTestOutputs:        {}  ->  cfg.nPrintAllTestOutputs:           {}'.format(options.nPrintAllTestOutputs, cfg.nPrintAllTestOutputs))




def UpdateInputImageSize(cfg):
    """
    Define base model used in transfer learning.
    """

    if cfg.sCNNType == 'VGG16' or str(cfg.sCNNType).lower() == 'vgg16':     
        cfg.nImageRows = 224
        cfg.nImageColumns = 224

    elif cfg.sCNNType ==' VGG19' or str(cfg.sCNNType).lower() =='vgg19':
        cfg.nImageRows = 224
        cfg.nImageColumns = 224

    elif cfg.sCNNType == 'DenseNet' or str(cfg.sCNNType).lower() ==' densenet':
        cfg.nImageRows = 224
        cfg.nImageColumns = 224        

    elif cfg.sCNNType == 'DenseNet121' or str(cfg.sCNNType).lower() ==' densenet121':
        cfg.nImageRows = 224
        cfg.nImageColumns = 224        
        
    elif cfg.sCNNType == 'DenseNet169' or str(cfg.sCNNType).lower() ==' densenet169':
        cfg.nImageRows = 224
        cfg.nImageColumns = 224        
        
    elif cfg.sCNNType == 'DenseNet201' or str(cfg.sCNNType).lower() ==' densenet201':
        cfg.nImageRows = 224
        cfg.nImageColumns = 224        

    elif cfg.sCNNType == 'InceptionV3' or str(cfg.sCNNType).lower() =='inceptionv3':
        cfg.nImageRows = 299
        cfg.nImageColumns = 299
        #keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

    elif cfg.sCNNType == 'InceptionResNetV2' or str(cfg.sCNNType).lower() =='InceptionResnetv2':
        cfg.nImageRows = 299
        cfg.nImageColumns = 299
        #BaseModel = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        
    elif cfg.sCNNType == 'MobileNet' or str(cfg.sCNNType).lower() =='mobilenet':
        cfg.nImageRows = 224
        cfg.nImageColumns = 224
        #keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=False, weights='imagenet', input_shape= oInput_Shape, alpha=1.0, depth_multiplier=1, dropout=1e-3,  input_tensor=None, pooling=None, classes=1000)

    elif cfg.sCNNType == 'MobileNetV2' or str(cfg.sCNNType).lower() =='mobilenetv2':
        cfg.nImageRows = 224
        cfg.nImageColumns = 224
        #keras.applications.mobilenet_v2.MobileNetV2(input_shape= oInput_Shape, alpha=1.0, include_top=False, weights='imagenet', input_tensor=None, pooling=None, classes=1000)

    elif cfg.sCNNType == 'NASNet' or str(cfg.sCNNType).lower() =='nasnet':
        cfg.nImageRows = 224
        cfg.nImageColumns = 224
        
    elif cfg.sCNNType == 'NASNetMobile' or str(cfg.sCNNType).lower() =='nasnetmobile':
        cfg.nImageRows = 224
        cfg.nImageColumns = 224

    elif cfg.sCNNType == 'NASNetLarge' or str(cfg.sCNNType).lower() =='nasnetlarge':
        cfg.nImageRows = 331
        cfg.nImageColumns = 331
         
    elif cfg.sCNNType == 'ResNet' or str(cfg.sCNNType).lower() =='resnet':
        cfg.nImageRows = 224
        cfg.nImageColumns = 224

    elif cfg.sCNNType == 'ResNet50' or str(cfg.sCNNType).lower() =='resnet50':
        cfg.nImageRows = 224
        cfg.nImageColumns = 224
        
    elif cfg.sCNNType == 'ResNet101' or str(cfg.sCNNType).lower() =='resnet101':
        cfg.nImageRows = 224
        cfg.nImageColumns = 224

    elif cfg.sCNNType == 'ResNet152' or str(cfg.sCNNType).lower() =='resnet152':        
        cfg.nImageRows = 224
        cfg.nImageColumns = 224
        
    elif cfg.sCNNType == 'ResNet50V2' or str(cfg.sCNNType).lower() =='resnet50v2':
        cfg.nImageRows = 224
        cfg.nImageColumns = 224
        
    elif cfg.sCNNType == 'ResNet101V2' or str(cfg.sCNNType).lower() =='resnet101v2':
        cfg.nImageRows = 224
        cfg.nImageColumns = 224

    elif cfg.sCNNType == 'ResNet152V2' or str(cfg.sCNNType).lower() =='resnet152v2':        
        cfg.nImageRows = 224
        cfg.nImageColumns = 224
        
    elif cfg.sCNNType == 'Xception' or str(cfg.sCNNType).lower() =='xception':
        cfg.nImageRows = 299
        cfg.nImageColumns = 299   
        
    elif cfg.sCNNType == 'EfficientNetB0' or str(cfg.sCNNType).lower() =='efficientnetb0':
        cfg.nImageRows = 224
        cfg.nImageColumns = 224        

    elif cfg.sCNNType == 'EfficientNetB7' or str(cfg.sCNNType).lower() =='efficientnetb7':
        cfg.nImageRows = 600
        cfg.nImageColumns = 600        

    else:
        raise ValueError('Valid base model [{}] did not select preprocess_input function.'.format(cfg.sCNNType))
   
   


def LoadDataFromFileWithCategorizeLabel( cfg,  sFilePathForTrainImages ):
    
   
    
    print('sImageFile:              {}'.format(sFilePathForTrainImages))
    print('cfg.nImageRows:          {}'.format(cfg.nImageRows))
    print('cfg.nImageColumns:       {}'.format(cfg.nImageColumns))
    print('cfg.nImageChannels:      {}'.format(cfg.nImageChannels))
    print('cfg.sImageChannelType:   {}'.format(cfg.sImageChannelType))
    print('cfg.sClassAssignment:    {}'.format(cfg.sClassAssignment))
    

    
    sListForTrainDataFileNames=[]
    oFileOpen  = open(sFilePathForTrainImages, 'r') 
    for aline in oFileOpen:
        try:
            aLabel, aFilePath = aline.split(';')
            aImagePath, aImageName = os.path.split(aFilePath)
            sImageName = str(aImageName)            
            if sImageName.find('.tif') > 0 or sImageName.find('.jpeg') > 0 or sImageName.find('.png') > 0  or sImageName.find('.bmp') > 0:                
                sListForTrainDataFileNames.append(aline)
                              
        except:
            print('Image Error: %s' % aline)
            continue
            
    oFileOpen.close()
    
 
    
    print('NbrOfData before Class Label Reassignment:               {}'.format( len(sListForTrainDataFileNames)))
    print('NbrOfClasses before Class Label Reassignment:            {}'.format(cfg.nNbrOfClasses))

    
    sListForTrainDataFileNames, nNbrOfClasses = ReassignClassLabelsInList( cfg, sListForTrainDataFileNames )

    
    if(int(nNbrOfClasses) > 0 ):
        cfg.nNbrOfClasses = int(nNbrOfClasses)

    print('NbrOfData After Class Label Reassignment:               {}'.format(len(sListForTrainDataFileNames)))
    print('NbrOfClasses After Class Label Reassignment:            {}'.format(cfg.nNbrOfClasses))



  
    random.shuffle(sListForTrainDataFileNames)
    random.shuffle(sListForTrainDataFileNames)
    

    nNbrOfDataTemp= len(sListForTrainDataFileNames);
 
    if( cfg.nImageChannels == 3):

        DataX = np.ndarray((nNbrOfDataTemp, cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels), dtype=np.float32)
    else:

        DataX = np.ndarray((nNbrOfDataTemp, cfg.nImageRows, cfg.nImageColumns), dtype=np.float32)

    DataY = np.zeros((nNbrOfDataTemp,), dtype='uint8')


    lClass =[0]*100   # 100 is the max number of classes     
    
    nNbrOfData=0
    for aFileName in sListForTrainDataFileNames[:]:        
        aClassIndex, aFileNamePath  = aFileName.split(";")  
        sClassIndex = str(aClassIndex)
        sClassIndex = sClassIndex.lstrip()
        sClassIndex = sClassIndex.rstrip()       
        
        sImageName = str(aFileNamePath)
        sImageName = sImageName.strip('\r\n')
        sImageName = sImageName.lstrip()
        sImageName = sImageName.rstrip() 

        if sImageName.find('.tif') > 0 or sImageName.find('.jpeg') > 0 or sImageName.find('.png') > 0  or sImageName.find('.bmp') > 0:
                                
            if( os.path.isfile( os.path.join(cfg.sDirectoryForTrainImages, sImageName))  == False ):
                 sImageName =sImageName.replace('.tif','.jpg')
                 
                 if( os.path.isfile( os.path.join(cfg.sDirectoryForTrainImages, sImageName))  == False ):
                     sImageName =sImageName.replace('.jpg','.jpeg')
            
            
            if( cfg.nImageChannels == 3):
                img = cv2.imread(os.path.join(cfg.sDirectoryForTrainImages, sImageName), cv2.IMREAD_COLOR) 
            else:
                img = cv2.imread(os.path.join(cfg.sDirectoryForTrainImages, sImageName), cv2.IMREAD_GRAYSCALE)  
                
            imgGrey = cv2.imread(os.path.join(cfg.sDirectoryForTrainImages, sImageName), cv2.IMREAD_GRAYSCALE)  
                
            #20201025
            if( int(cfg.nUseClaheForInputImage) == 1 ):
                img = ClaheOperatorColor3D( img.copy(), cfg.nClaheTileSize )                            
                
            
            if( cfg.nImageChannels == 3):
                if( cfg.sImageChannelType == 'YCbCr' or cfg.sImageChannelType == 'ycbcr' ):

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)   #(if input image is RGB) 
                            
                elif( cfg.sImageChannelType == 'HSV' or cfg.sImageChannelType == 'hsv' ):
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  #(HSV (Hue, Saturation, Value) 
                            
                elif( cfg.sImageChannelType == 'HLS' or cfg.sImageChannelType == 'hls' ):
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)  #(HSL (Hue, Saturation, Lightness)
                            
                elif( cfg.sImageChannelType == 'LAB' or cfg.sImageChannelType == 'lab' ):                            

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)  #(HSL (Hue, Saturation, Lightness)    
                    
                elif( cfg.sImageChannelType == '3Grey' or cfg.sImageChannelType == '3grey' ):     #20240319                       
                    img[:,:,0] = imgGrey
                    img[:,:,1] = imgGrey
                    img[:,:,2] = imgGrey
                    
                            
            try:
                
                 
                if( img.shape[1] > cfg.nImageRows ):
                    img = cv2.resize(img, (cfg.nImageRows,cfg.nImageColumns), interpolation = cv2.INTER_AREA)  
                else:
                    img = cv2.resize(img, (cfg.nImageRows,cfg.nImageColumns), interpolation = cv2.INTER_CUBIC)
                
                DataX[nNbrOfData] = np.array([img])
                
                
                DataY[nNbrOfData] = int(sClassIndex)
                                   
                lClass[(int)(sClassIndex)] +=1
                
                nNbrOfData += 1
                
                
            except Exception as e:
                print('Image Error: [{}]  {}'.format(sImageName,  str(e)) )#sImageName
                continue

                             


            if nNbrOfData % 100 == 0:
                print('Done: {0}/{1} {2} images'.format(nNbrOfData, nNbrOfDataTemp, sImageName))
                 
        elif sImageName.find("thumbs.db") >= 0 :
            print('Removed wrong file {0}: {1}'.format(nNbrOfData, sImageName))               
        else :
            print('Removed wrong file {0}: {1}'.format(nNbrOfData, sImageName))  
            
    print('Loading Data done: {0} Data'.format(nNbrOfData))    


    nNbrOfClasses=0           
    for i in range(0, len(lClass), 1):
        if( lClass[i] != 0):
            nNbrOfClasses +=1            



    # to collect only available imagess  among images in the training/test list
    DataX = DataX[:nNbrOfData]


    
    if( nNbrOfClasses > 1 ):                              
        DataY = np_utils.to_categorical(DataY[:nNbrOfData], nNbrOfClasses)


    return DataX, DataY, nNbrOfData, nNbrOfClasses, lClass



def ReassignClassLabelsInList( cfg, sListForTrainDataFileNames ):
    

    sListForTrainDataFileNamesNew=[]    
    if( len(cfg.sClassAssignment) > 0 ):
        nNbrOfClasses, nClassNotUse, lClassAssigment = GetClassReassignmentInformation( cfg.sClassAssignment  )    
        print('nNbrOfClasses:       {}'.format(nNbrOfClasses))
        print('nClassNotUse:        {}'.format(nClassNotUse))
        print('lClassAssigment:     ')
        print(' {}'.format(lClassAssigment))


        for aFileName in sListForTrainDataFileNames[:]:  
            
            aFileNameNew = ReassignClassLabelInListOfAData( cfg, lClassAssigment, aFileName )
            
            if( len(aFileNameNew) > 0 ):              
                sListForTrainDataFileNamesNew.append(aFileNameNew)
                        
        return  sListForTrainDataFileNamesNew, nNbrOfClasses

    elif( cfg.nNbrOfClasses > 0 ):
        return  sListForTrainDataFileNames,  cfg.nNbrOfClasses   
    
    else:
        return  sListForTrainDataFileNames, 0
       
  

# 20200723 reassign classes
def GetClassReassignmentInformation( sClassAssignment  ):
    
    #sClassAssignment = '0,0;1,0;2,1;3,1'
    sClassIndex = sClassAssignment.split(";")  
    
    nNbrOfClasses = int(len(sClassIndex))
    lClassAssigment = np.zeros((nNbrOfClasses,2), dtype = np.int)
    nClassNotUse=0
    
    lClassCount = np.zeros(100).astype(int)
    for i in range(0, nNbrOfClasses ):
        aClassIndex = sClassIndex[i]
        sIndexOrg, sIndexNew = aClassIndex.split(",")  
        
        if( sIndexNew != 'X' ):
            lClassAssigment[i,0] = int(sIndexOrg)
            lClassAssigment[i,1] = int(sIndexNew)
            
            lClassCount[int(sIndexNew)] +=1          
        else:
            lClassAssigment[i,0] = int(sIndexOrg)
            lClassAssigment[i,1] = -1
            
            nClassNotUse=1
        
    for i in range(0, nNbrOfClasses ):
        print( '[{}] Org:{}  -> New:{}'.format(i,lClassAssigment[i,0],lClassAssigment[i,1] ))
        
        
        
    nNbrOfClassesNew = 0    
    for i in range(len(lClassCount)):
        if(lClassCount[i] > 0):
            nNbrOfClassesNew += 1
            
    
    return nNbrOfClassesNew, nClassNotUse, lClassAssigment

    
    
    

def ReassignClassLabelInListOfAData( cfg, lClassAssigment, aListForTrainDataFileName ):
    
    if( len(cfg.sClassAssignment) > 0 ):
        aClassIndex, aFileNamePath  = aListForTrainDataFileName.split(";")  
        sClassIndex = str(aClassIndex)
        sClassIndex = sClassIndex.lstrip()
        sClassIndex = sClassIndex.rstrip()  
        nClassIndex = int(sClassIndex)  
                 
        aListForTrainDataFileNameNew = ''     
        if (lClassAssigment[nClassIndex,1] != -1 ):
            aListForTrainDataFileNameNew = aListForTrainDataFileName.replace('{};'.format(aClassIndex), '{};'.format(lClassAssigment[nClassIndex,1]) )    
        else:
            aListForTrainDataFileNameNew = ''
            
        return aListForTrainDataFileNameNew
    
    else:
        return aListForTrainDataFileName
        




def GetPreprocessInput( cfg, Data ):
 
    if( int(cfg.nPreprocessInput) == 1 ):
        if cfg.sCNNType == 'VGG16' or str(cfg.sCNNType).lower() == 'vgg16':    
            Data = VGG16_preprocess_input(Data)
        elif cfg.sCNNType ==' VGG19' or str(cfg.sCNNType).lower() =='vgg19':
            Data = VGG19_preprocess_input(Data)

        elif cfg.sCNNType == 'DenseNet' or str(cfg.sCNNType).lower() ==' densenet':
            Data = DenseNet121_preprocess_input(Data)
        elif cfg.sCNNType == 'DenseNet121' or str(cfg.sCNNType).lower() ==' densenet121':
            Data = DenseNet121_preprocess_input(Data)
            
        elif cfg.sCNNType == 'DenseNet169' or str(cfg.sCNNType).lower() ==' densenet169':
            Data = DenseNet169_preprocess_input(Data)
            
        elif cfg.sCNNType == 'DenseNet201' or str(cfg.sCNNType).lower() ==' densenet201':
            Data = DenseNet201_preprocess_input(Data)
            
        elif cfg.sCNNType == 'InceptionV3' or str(cfg.sCNNType).lower() =='inceptionv3':
            Data = InceptionV3_preprocess_input(Data)

        elif cfg.sCNNType == 'InceptionResNetV2' or str(cfg.sCNNType).lower() =='Inceptionresnetv2':
            Data = InceptionResNetV2_preprocess_input(Data)
            
        elif cfg.sCNNType == 'MobileNet' or str(cfg.sCNNType).lower() =='mobilenet':
            Data = MobileNet_preprocess_input(Data)
 
        elif cfg.sCNNType == 'MobileNetV2' or str(cfg.sCNNType).lower() =='mobilenetv2':
            Data = MobileNetV2_preprocess_input(Data)

        elif cfg.sCNNType == 'NASNet' or str(cfg.sCNNType).lower() =='nasnet':
            Data = NASNetMobile_preprocess_input(Data)
        elif cfg.sCNNType == 'NASNetMobile' or str(cfg.sCNNType).lower() =='nasnetmobile':
            Data = NASNetMobile_preprocess_input(Data)

        elif cfg.sCNNType == 'NASNetLarge' or str(cfg.sCNNType).lower() =='nasnetlarge':
            Data = NASNetLarge_preprocess_input(Data)
           
        elif cfg.sCNNType == 'ResNet' or str(cfg.sCNNType).lower() =='resnet':
            Data = ResNet50_preprocess_input(Data)
        elif cfg.sCNNType == 'ResNet50' or str(cfg.sCNNType).lower() =='resnet50':  
            Data = ResNet50_preprocess_input(Data)
            
        elif cfg.sCNNType == 'ResNet101' or str(cfg.sCNNType).lower() =='resnet101':
            Data = ResNet101_preprocess_input(Data)
            
        elif cfg.sCNNType == 'ResNet152' or str(cfg.sCNNType).lower() =='resnet152':
            Data = ResNet152_preprocess_input(Data)
            
        elif cfg.sCNNType == 'ResNet50V2' or str(cfg.sCNNType).lower() =='resnet50v2':
            Data = ResNet50V2_preprocess_input(Data)
            
        elif cfg.sCNNType == 'ResNet101V2' or str(cfg.sCNNType).lower() =='resnet101v2':
            Data = ResNet101V2_preprocess_input(Data)
            
        elif cfg.sCNNType == 'ResNet152V2' or str(cfg.sCNNType).lower() =='resnet152v2':
            Data = ResNet152V2_preprocess_input(Data)

        elif cfg.sCNNType == 'Xception' or str(cfg.sCNNType).lower() ==' xception':
            Data =Xception_preprocess_input(Data)

        else:
            raise ValueError('Valid base model [{}] did not select preprocess_input function.'.format(cfg.sCNNType))

    return Data    
    
 



# 20210927 Normalize images using mean and std
def NormalizeImagesIntensityUsingMeanAndStd ( oImages, nUseRGB = 1, bUseMean = True, bUseStdVariation = True ) :
    
    print('oImages.shape[0]: ', oImages.shape[0] )
    oImages = oImages.astype('float32')
        
    if ( nUseRGB > 0 ): # estimate mean and std of ecah RGB channels separately for each image
        
        if (len(oImages.shape)  > 3 ): # for color images
            for i in range(oImages.shape[0]):
                aImage = oImages[i]
                for j in range( aImage.shape[2]):
                    dMean = np.mean(aImage[:,:,j])  # mean for data centering
                    dStd  = np.std(aImage[:,:,j])  # std for data normalization
                    aImage[:,:,j] -= dMean
                    if( bUseStdVariation ):
                        aImage[:,:,j] /= dStd
                        
                oImages[i] = aImage                    
                        
        else: # for grey images
            for i in range(oImages.shape[0]):
                dMean = np.mean(oImages[i])  # mean for data centering
                dStd  = np.std(oImages[i])  # std for data normalization
                oImages[i] -= dMean
                if( bUseStdVariation ):
                    oImages[i] /= dStd      
                    
    else: #combine  RGB channels to estimate a common mean and std for each image
        for i in range(oImages.shape[0]):
            dMean = np.mean(oImages[i])  # mean for data centering
            dStd = np.std(oImages[i])  # std for data normalization
            oImages[i] -= dMean
            oImages[i] /= dStd
        
        
    
    return oImages




def GetBaseCnnModel(cfg):
    """
    Define base model used in transfer learning.
    """

    sModelSummary = ''

    if cfg.sCNNType == 'VGG16' or str(cfg.sCNNType).lower() == 'vgg16':   
        
        # cfg.nImageRows = 224
        # cfg.nImageColumns = 224

        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = VGG16(include_top=False, weights='imagenet', input_shape= oInput_Shape)   
        else:
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))
        

    elif cfg.sCNNType ==' VGG19' or str(cfg.sCNNType).lower() =='vgg19':
        # cfg.nImageRows = 224
        # cfg.nImageColumns = 224

        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = VGG19(include_top=False, weights='imagenet', input_shape= oInput_Shape)   
        else:
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = VGG19(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))


    elif cfg.sCNNType == 'DenseNet' or str(cfg.sCNNType).lower() ==' densenet':
        # cfg.nImageRows = 224
        # cfg.nImageColumns = 224        
               
        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = DenseNet121(include_top=False, weights='imagenet', input_shape= oInput_Shape)   
        else:
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = DenseNet121(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))
        
        
    elif cfg.sCNNType == 'DenseNet121' or str(cfg.sCNNType).lower() ==' densenet121':
        # cfg.nImageRows = 224
        # cfg.nImageColumns = 224        
        
        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = DenseNet121(include_top=False, weights='imagenet', input_shape= oInput_Shape)   
        else:
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = DenseNet121(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))
        
            
    elif cfg.sCNNType == 'DenseNet121n' or str(cfg.sCNNType).lower() ==' densenet121n':
        # cfg.nImageRows = 224
        # cfg.nImageColumns = 224        
        
        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = DenseNet121(include_top=False, weights='imagenet', input_shape= oInput_Shape)   
        else:
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = DenseNet121(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))
        
        
    elif cfg.sCNNType == 'DenseNet169' or str(cfg.sCNNType).lower() ==' densenet169':
        # cfg.nImageRows = 224
        # cfg.nImageColumns = 224        
        
        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = DenseNet169(include_top=False, weights='imagenet', input_shape= oInput_Shape)   
        else:
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = DenseNet169(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))
            
        
    elif cfg.sCNNType == 'DenseNet201' or str(cfg.sCNNType).lower() ==' densenet201':
        # cfg.nImageRows = 224
        # cfg.nImageColumns = 224        
        
        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = DenseNet201(include_top=False, weights='imagenet', input_shape= oInput_Shape)   
        else:
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = DenseNet201(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))
        

    elif cfg.sCNNType == 'InceptionV3' or str(cfg.sCNNType).lower() =='inceptionv3':
        # cfg.nImageRows = 299
        # cfg.nImageColumns = 299
        
        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):
            cfg.nImageRows = 229
            cfg.nImageColumns = 229
        
        
        if( cfg.nImageRows == 299 and cfg.nImageColumns == 299 ):
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = InceptionV3(include_top=False, weights='imagenet', input_shape= oInput_Shape)   
        else:
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))
        
            
    elif cfg.sCNNType == 'InceptionResNetV2' or str(cfg.sCNNType).lower() =='InceptionResnetv2':
        # cfg.nImageRows = 299
        # cfg.nImageColumns = 299
        
        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):
            cfg.nImageRows = 229
            cfg.nImageColumns = 229        
        
        if( cfg.nImageRows == 299 and cfg.nImageColumns == 299 ):
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = InceptionResNetV2(include_top=False, weights='imagenet', input_shape= oInput_Shape)   
        else:
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = InceptionResNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))
        
        
    elif cfg.sCNNType == 'MobileNet' or str(cfg.sCNNType).lower() =='mobilenet':
        # cfg.nImageRows = 224
        # cfg.nImageColumns = 224
        
        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = MobileNet(include_top=False, weights='imagenet', input_shape= oInput_Shape)   
        else:
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = MobileNet(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))
        
            
    elif cfg.sCNNType == 'MobileNetV2' or str(cfg.sCNNType).lower() =='mobilenetv2':
        # cfg.nImageRows = 224
        # cfg.nImageColumns = 224
        
        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = MobileNetV2(include_top=False, weights='imagenet', input_shape= oInput_Shape)   
        else:
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))
        

    elif cfg.sCNNType == 'NASNet' or str(cfg.sCNNType).lower() =='nasnet':
        # cfg.nImageRows = 224
        # cfg.nImageColumns = 224
        
        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = NASNetMobile(include_top=False, weights='imagenet', input_shape= oInput_Shape)   
        else:
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = NASNetMobile(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))
        
    #20210823  need to update     
    elif cfg.sCNNType == 'NASNetMobile' or str(cfg.sCNNType).lower() =='nasnetmobile':
        # cfg.nImageRows = 224
        # cfg.nImageColumns = 224
        
        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = NASNetMobile(include_top=False, weights='imagenet', input_shape= oInput_Shape)   
        else:
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = NASNetMobile(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))
        

    elif cfg.sCNNType == 'NASNetLarge' or str(cfg.sCNNType).lower() =='nasnetlarge':
        # cfg.nImageRows = 331
        # cfg.nImageColumns = 331
        
        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):
            cfg.nImageRows = 331
            cfg.nImageColumns = 331
            
        if( cfg.nImageRows == 331 and cfg.nImageColumns == 331 ):
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = NASNetLarge(include_top=False, weights='imagenet', input_shape= oInput_Shape)   
        else:
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = NASNetLarge(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))
        
         
    elif cfg.sCNNType == 'ResNet' or str(cfg.sCNNType).lower() =='resnet':
        # cfg.nImageRows = 224
        # cfg.nImageColumns = 224
        
        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = ResNet50(include_top=False, weights='imagenet', input_shape= oInput_Shape)   
        else:
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))
                
                
    elif cfg.sCNNType == 'ResNet50' or str(cfg.sCNNType).lower() =='resnet50':
        # cfg.nImageRows = 224
        # cfg.nImageColumns = 224
        
        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):
            BaseModel = ResNet50(include_top=False, weights='imagenet')   
        else:
            
            BaseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))
                           
                       
    elif cfg.sCNNType == 'ResNet101' or str(cfg.sCNNType).lower() =='resnet101':
        # cfg.nImageRows = 224
        # cfg.nImageColumns = 224

        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):            
            BaseModel = ResNet101(include_top=False, weights='imagenet')   
        else:
            BaseModel = ResNet101(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))
        
                        


    elif cfg.sCNNType == 'ResNet152' or str(cfg.sCNNType).lower() =='resnet152':
        
        # cfg.nImageRows = 224
        # cfg.nImageColumns = 224
        
        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):
            BaseModel = ResNet152(include_top=False, weights='imagenet')   
        else:
            BaseModel = ResNet152(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))
                        
        
    elif cfg.sCNNType == 'ResNet50V2' or str(cfg.sCNNType).lower() =='resnet50v2':
        # cfg.nImageRows = 224
        # cfg.nImageColumns = 224
        
        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = ResNet50V2(include_top=False, weights='imagenet', input_shape= oInput_Shape)   
        else:
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = ResNet50V2(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))                        
        
    elif cfg.sCNNType == 'ResNet101V2' or str(cfg.sCNNType).lower() =='resnet101v2':
        # cfg.nImageRows = 224
        # cfg.nImageColumns = 224
        
        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = ResNet101V2(include_top=False, weights='imagenet', input_shape= oInput_Shape)   
        else:
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = ResNet101V2(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))
        
    elif cfg.sCNNType == 'ResNet152V2' or str(cfg.sCNNType).lower() =='resnet152v2':        
        # cfg.nImageRows = 224
        # cfg.nImageColumns = 224
        
        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = ResNet152V2(include_top=False, weights='imagenet', input_shape= oInput_Shape)   
        else:
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = ResNet152V2(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))
            
        
    elif cfg.sCNNType == 'Xception' or str(cfg.sCNNType).lower() =='xception':
        # cfg.nImageRows = 299
        # cfg.nImageColumns = 299        \        
        
        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):
            cfg.nImageRows = 299
            cfg.nImageColumns = 299
                
        
        if( cfg.nImageRows == 299 and cfg.nImageColumns == 299 ):
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = Xception(include_top=False, weights='imagenet', input_shape= oInput_Shape)   
        else:
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = Xception(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))
        
    elif cfg.sCNNType == 'EfficientNetB0' or str(cfg.sCNNType).lower() =='efficientnetb0':
                        
        
        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = EfficientNetB0(include_top=False, weights='imagenet', input_shape= oInput_Shape)   
        else:
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = EfficientNetB0(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))
        
                        
        
        if( cfg.nImageRows == 600 and cfg.nImageColumns == 600 ):
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = EfficientNetB0(include_top=False, weights='imagenet', input_shape= oInput_Shape)   
        else:
            oInput_Shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels )        
            BaseModel = EfficientNetB0(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))

    else:
        raise ValueError('Valid base model [{}] did not select preprocess_input function.'.format(cfg.sCNNType))
        print(' ResNet50 is used as a default base CNN')
        
        if( cfg.nImageRows == 224 and cfg.nImageColumns == 224 ):
            BaseModel = ResNet50(include_top=False, weights='imagenet')   
        else:            
            BaseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels)))

    
    
    #freeze pretrained model weights          
    if( int(cfg.nTransferLearning) == 1 ):
        for layer in BaseModel.layers:            
            layer.trainable = False
     
        
    sModelSummary += '=== Get Base CNN Model [{}] ===\n'.format(cfg.sCNNType)                   
    sModelSummary += str(BaseModel.summary())
       
    sModelSummary += '===Trainable Layer Status of {}===\n'.format(cfg.sCNNType)
    for layer in BaseModel.layers:
        sModelSummary += '{} {}\n'.format(layer, layer.trainable)           
                 
                
        
    #Create your own input format (here 3x200x200)
    #oInput = Input(shape=oInput_Shap, name = 'image_input')
    

    x = BaseModel.output
    

        
    #Add the fully-connected layers 
    
    
    if cfg.sCNNType == 'VGG16' or str(cfg.sCNNType).lower() == 'vgg16':            
        # 20201015 x = Flatten(name='flatten')(x)
        x = GlobalAveragePooling2D()(x)
        
    elif cfg.sCNNType ==' VGG19' or str(cfg.sCNNType).lower() == 'vgg19':
        # 20201015 x = Flatten(name='flatten')(x)
        x = GlobalAveragePooling2D()(x)
        #x = Flatten(name='flatten')(x)
        
    elif cfg.sCNNType == 'DenseNet' or str(cfg.sCNNType).lower() =='densenet': 
        x = GlobalAveragePooling2D()(x)
        
    elif cfg.sCNNType == 'DenseNet121' or str(cfg.sCNNType).lower() =='densenet121':
        x = GlobalAveragePooling2D()(x)
        
    elif cfg.sCNNType == 'DenseNet169' or str(cfg.sCNNType).lower() =='densenet169':
        x = GlobalAveragePooling2D()(x)
        
    elif cfg.sCNNType == 'DenseNet201' or str(cfg.sCNNType).lower() =='densenet201':
        x = GlobalAveragePooling2D()(x)
                
    elif cfg.sCNNType == 'InceptionV3' or str(cfg.sCNNType).lower() =='inceptionv3': 
        x = GlobalAveragePooling2D()(x)
        
    elif cfg.sCNNType == 'InceptionResNetV2' or str(cfg.sCNNType).lower() =='inceptionresnetv2': #?
        # 20201015 x = Flatten(name='flatten')(x)
        x = GlobalAveragePooling2D()(x)
        
    elif cfg.sCNNType == 'MobileNet' or str(cfg.sCNNType).lower() == 'mobilenet':
        x = GlobalAveragePooling2D()(x)
        
    elif cfg.sCNNType == 'MobileNetV2' or str(cfg.sCNNType).lower() == 'mobilenetv2':
        x = GlobalAveragePooling2D()(x)
        
    elif cfg.sCNNType == 'NasNet' or str(cfg.sCNNType).lower() == 'nasnet':
        x = GlobalAveragePooling2D()(x)

    elif cfg.sCNNType == 'NASNetMobile' or str(cfg.sCNNType).lower() == 'nasnetmobile':
        x = GlobalAveragePooling2D()(x)
        
    elif cfg.sCNNType == 'NASNetLarge' or str(cfg.sCNNType).lower() == 'nasnetlarge':
        x = GlobalAveragePooling2D()(x)
        
    elif cfg.sCNNType == 'ResNet50' or str(cfg.sCNNType).lower() == 'resnet50':
        x = GlobalAveragePooling2D()(x)
        
    elif cfg.sCNNType == 'ResNet101' or str(cfg.sCNNType).lower() == 'resnet101':
        x = GlobalAveragePooling2D()(x)
        
    elif cfg.sCNNType == 'ResNet152' or str(cfg.sCNNType).lower() == 'resnet152':
        x = GlobalAveragePooling2D()(x)
        
    elif cfg.sCNNType == 'ResNet50V2' or str(cfg.sCNNType).lower() == 'resnet50v2':
        x = GlobalAveragePooling2D()(x)
        
    elif cfg.sCNNType == 'ResNet101V2' or str(cfg.sCNNType).lower() == 'resnet101v2':
        x = GlobalAveragePooling2D()(x)
        
    elif cfg.sCNNType == 'ResNet152V2' or str(cfg.sCNNType).lower() == 'resnet152v2':
        x = GlobalAveragePooling2D()(x)
        
    elif cfg.sCNNType == 'Xception' or str(cfg.sCNNType).lower() == 'xception':
        x = GlobalAveragePooling2D()(x)
        
    elif cfg.sCNNType == 'EfficientNetB0' or str(cfg.sCNNType).lower() == 'efficientnetb0':
        x = GlobalAveragePooling2D()(x)

    elif cfg.sCNNType == 'EfficientNetB7' or str(cfg.sCNNType).lower() == 'efficientnetb7':
        x = GlobalAveragePooling2D()(x)
  
    else:
        raise ValueError('Valid base model [{}] did not select either Flatten or GlobalAveragePooling2D layer.'.format(cfg.sCNNType))
    
    
    x = Dense(1024, activation='relu', name='fcDense1')(x)
    
    if( float(cfg.dDropoutRatio) > 0.0 ):
        x = Dropout(cfg.dDropoutRatio)(x)       
        
    x = Dense(1024, activation='relu', name='fcDense2')(x)
    
    if( float(cfg.dDropoutRatio) > 0.0 ):
        x = Dropout(cfg.dDropoutRatio)(x)       
    
    
    if( cfg.nNbrOfClasses >= 2 ):
        Prediction = Dense(cfg.nNbrOfClasses, activation='softmax', name='fcOutput')(x)   #to have 0 or 1 result for each class
    else:
        Prediction = Dense(cfg.nNbrOfClasses-1, activation='sigmoid', name='fcOutput')(x)   #to have 0 or 1 result for each class
        
    
    model = Model(inputs = BaseModel.input, outputs = Prediction)
    
      

    if( cfg.sOptimizer == 'SGD' or str(cfg.sOptimizer).lower() == 'sgd'):
        dLearningRate = 1e-4        
        if( str(cfg.sOperation).lower().find('retrain') >= 0 ):
            dLearningRate = dLearningRate*0.1
    
        oOptimizer = SGD(lr=dLearningRate, decay=dLearningRate*0.01, momentum=0.9, nesterov=True)
        
    elif( cfg.sOptimizer == 'Adam' or str(cfg.sOptimizer).lower() == 'adam'):
        dLearningRate = 1e-5        
        if( str(cfg.sOperation).lower().find('retrain') >= 0 ):
            dLearningRate = dLearningRate*0.1
            
        oOptimizer = Adam(lr = dLearningRate)
    elif( cfg.sOptimizer == 'Nadam' or str(cfg.sOptimizer).lower() == 'nadam'):
        dLearningRate = 1e-5        
        if( str(cfg.sOperation).lower().find('retrain') >= 0 ):
            dLearningRate = dLearningRate*0.1
            
        oOptimizer = Nadam(lr = dLearningRate)
    elif( cfg.sOptimizer == 'RMSprop' or str(cfg.sOptimizer).lower() == 'rmsprop'):
        dLearningRate = 1e-5        
        if( str(cfg.sOperation).lower().find('retrain') >= 0 ):
            dLearningRate = dLearningRate*0.1
            
        oOptimizer = RMSprop(lr = dLearningRate)
    else:
        dLearningRate = 1e-4        
        if( str(cfg.sOperation).lower().find('retrain') >= 0 ):
            dLearningRate = dLearningRate*0.1

        oOptimizer = SGD(lr=dLearningRate, decay=dLearningRate*0.01, momentum=0.9, nesterov=True)

   
    if int(cfg.nNbrOfGPUs) <= 1:            
                
        if( int(cfg.nUseWeightedCategorialCrossEntropy) == 0 ):   
            if( int(cfg.nNbrOfClasses) <= 2 ):
                model.compile( optimizer=oOptimizer, loss='binary_crossentropy', metrics=['accuracy'])
            else:
                model.compile( optimizer=oOptimizer, loss='categorical_crossentropy', metrics=['accuracy'])            
        else:
           model.compile( optimizer=oOptimizer, loss= Custom_weighted_categorical_crossentropy, metrics=['accuracy'])          
           
    else:
        # Replicates `model` on 8 GPUs.
        # This assumes that your machine has 8 available GPUs.
        model = multi_gpu_model(model, gpus = int( cfg.nNbrOfGPUs))
                
        if( int(cfg.nNbrOfClasses) <= 2 ):
            model.compile(loss='binary_crossentropy', optimizer=oOptimizer, metrics=['accuracy'])
        else:
            model.compile(loss='categorical_crossentropy', optimizer=oOptimizer, metrics=['accuracy'])
            
            
            
    sModelSummary += ' '
    sModelSummary += '=== Final model summary ===\n'   
    sModelSummary += str(model.summary())       


    sModelSummary += '\n\n  Optimizer: {}  \n  Learning Rate: {}  \n'.format(cfg.sOptimizer, dLearningRate)
                
    
    return model, sModelSummary





def TrainAndTestCNNUsingInputFile( cfg ):
        

    
    if (not os.path.exists(cfg.sDirectoryForTrainingResultsRoot) ):
        os.mkdir(cfg.sDirectoryForTrainingResultsRoot)
        
    print('------------------------------------------------------------------------------------------------------')
    print('Start: Get training image set')  
    

    # update input image size
    if( cfg.nImageRows == 0 and cfg.nImageColumns == 0 ):
        UpdateInputImageSize(cfg)
           
    nDropOutRatio = int(float(cfg.dDropoutRatio)*100.0)
    nValidationSplit = int(float(cfg.dValidationSplit)*100.0)
    
    sClassAssignment= cfg.sClassAssignment
    
    if( len(cfg.sClassAssignment) > 0 ):
        for i in range (100):
            sClassAssignment =sClassAssignment.replace('{},'.format(i),'')
        sClassAssignment = sClassAssignment.replace(';','')
        

    dt=datetime.datetime.today()
    stime='{}{}{}{}{}{}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    


    sFileName = '{}_Ca{}_Cl{}_{}_Ro{}_Co{}_Ch{}_Vs{}_TVG{}_Do{}_Au{}_Pr{}_InM1{}_InC3{}_Vt{}_Tl{}_Bs{}_{}_Ep{}_{}_{}'.format(cfg.sCNNType,sClassAssignment,cfg.nUseClaheForInputImage, cfg.sImageChannelType, cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels, nValidationSplit, cfg.dTrainValidDataGeneration, nDropOutRatio, cfg.nAugmentationNumber, cfg.nPreprocessInput, cfg.nImageIntensityNormalizationToMax1, cfg.nImageIntensityNormalizationUseRGBChannel, cfg.sVotingMethod, cfg.nTransferLearning, cfg.nBatchSize, cfg.sOptimizer, cfg.nEpochs, cfg.sFilenameExtension, stime )

    cfg.sDirectoryForTrainingResultsCurrent = os.path.join(cfg.sDirectoryForTrainingResultsRoot, sFileName )
    
    if (os.path.exists(cfg.sDirectoryForTrainingResultsCurrent) ):
        shutil.rmtree(cfg.sDirectoryForTrainingResultsCurrent)

    os.mkdir(cfg.sDirectoryForTrainingResultsCurrent)
    
    
    cfg.sDirectoryForWeights = cfg.sDirectoryForTrainingResultsCurrent

              
    sFileName = '{}-{}-Ep{}-TrainProcess.txt'.format(cfg.sCNNType, cfg.sOptimizer, cfg.nEpochs)
    sResultFilePath = os.path.join(cfg.sDirectoryForTrainingResultsCurrent,sFileName )
    print('TrainProcess: ', sResultFilePath)
    fileWrite = open(sResultFilePath,"w") 
    

    print( 'Satrt reading Input Data ')
    TrainX, TrainY, nNbrOfDataTrain, nNbrOfClasses, nClassNumbers = LoadDataFromFileWithCategorizeLabel( cfg, cfg.sFilePathForTrainImages )
       
    print('Original   TrainX: max: {},  min: {}'.format( np.max(TrainX), np.min(TrainX)))
    
    
    #20230421 to save weights based on validity score
    sWeightsH_FilePath = os.path.join(cfg.sDirectoryForWeights, 'weights.h5' )


    print('-'*30)    
    print('=== Before Augmentation===      ' )  
    print('nNbrOfClasses:       {}'.format(nNbrOfClasses) )  
    print('TrainX.shape:        {}'.format(TrainX.shape))
    print('TrainY.shape:        {}'.format(TrainY.shape))
    print('-'*30)
    
    print('=== Number of Data in Each Class ===')
    for i in range (nNbrOfClasses):
        print('Class[{}] = {}'.format(i, nClassNumbers[i]))


    fileWrite.write('-'*30)    
    fileWrite.write('\n')    
    fileWrite.write('=== Before Augmentation===      \n' )  
    fileWrite.write('nNbrOfClasses:       {} \n'.format(nNbrOfClasses) )  
    fileWrite.write('TrainX.shape:        {} \n'.format(TrainX.shape))
    fileWrite.write('TrainY.shape:        {} \n'.format(TrainY.shape))
    fileWrite.write('-'*30)
    fileWrite.write('\n')    
 
    fileWrite.write('=== Number of Data in Each Class === \n')
    for i in range (nNbrOfClasses):
        fileWrite.write('Class[{}] = {} \n'.format(i, nClassNumbers[i]))
 
    TrainX, TrainY, nNbrOfDataTrain, nClassNumbers = GetAugmentedImagesRegular( cfg.nAugmentationNumber, TrainX, TrainY, nNbrOfDataTrain, nNbrOfClasses, nClassNumbers, 2 )    

    print('Before Preporcess TrainX: max: {},  min: {}'.format( np.max(TrainX), np.min(TrainX)))
    fileWrite.write('Before Preporcess TrainX: max: {},  min: {} \n'.format( np.max(TrainX), np.min(TrainX)))
    #convert to  preprocess_input  
    
    if( int(cfg.nPreprocessInput) == 1 ):    
        TrainX = GetPreprocessInput(cfg, TrainX)   
                
        print('Preporcess [GetPreprocessInput] TrainX: max: {},  min: {}'.format( np.max(TrainX), np.min(TrainX)))
        fileWrite.write('Preporcess [GetPreprocessInput] TrainX: max: {},  min: {} \n'.format( np.max(TrainX), np.min(TrainX)))
  
    
    if( int(cfg.nImageIntensityNormalizationToMax1) == 1 ):    
        TrainX = TrainX/255.0  
    
        print('Preporcess [nImageIntensityNormalizationToMax1] TrainX: max: {},  min: {}'.format( np.max(TrainX), np.min(TrainX)))
        fileWrite.write('Preporcess[nImageIntensityNormalizationToMax1] Preporcess TrainX: max: {},  min: {} \n'.format( np.max(TrainX), np.min(TrainX)))
    
    
    if( int(cfg.nImageIntensityNormalizationUseRGBChannel) == 1 ):    
        TrainX = NormalizeImagesIntensityUsingMeanAndStd( TrainX, cfg.nImageIntensityNormalizationUseRGBChannel, True, True )    
    
        print('Preporcess [nImageIntensityNormalizationUseRGBChannel] TrainX: max: {},  min: {}'.format( np.max(TrainX), np.min(TrainX)))
        fileWrite.write('Preporcess[nImageIntensityNormalizationUseRGBChannel] Preporcess TrainX: max: {},  min: {} \n'.format( np.max(TrainX), np.min(TrainX)))
    
 
    
    print('-'*30)    
    print('=== After Augmentation===      ' )  
    print('nNbrOfClasses:       {}'.format(nNbrOfClasses) )  
    print('TrainX.shape:        {}'.format(TrainX.shape))
    print('TrainY.shape:        {}'.format(TrainY.shape))
    print('-'*30)
    
    print('=== Number of Data in Each Class ===')
    for i in range (nNbrOfClasses):
        print('Class[{}] = {}'.format(i, nClassNumbers[i]))


    fileWrite.write('-'*30)    
    fileWrite.write('\n')    
    fileWrite.write('=== After Augmentation===      \n' )  
    fileWrite.write('nNbrOfClasses:       {} \n'.format(nNbrOfClasses) )  
    fileWrite.write('TrainX.shape:        {} \n'.format(TrainX.shape))
    fileWrite.write('TrainY.shape:        {} \n'.format(TrainY.shape))
    fileWrite.write('-'*30)
    fileWrite.write('\n')    
 
    fileWrite.write('=== Number of Data in Each Class === \n')
    for i in range (nNbrOfClasses):
        fileWrite.write('Class[{}] = {} \n'.format(i, nClassNumbers[i]))


    
    tTrainingTimeStart=datetime.datetime.now() 
    
    print('-'*30)
    print('Get {} model'.format(cfg.sCNNType))
    print('-'*30)    
    print('cfg.bUseSlurm:                           {}'.format(cfg.bUseSlurm))
    print('cfg.sOperation:                          {}'.format(cfg.sOperation))
    print('cfg.sCNNType:                            {}'.format(cfg.sCNNType))
    print('cfg.nTransferLearning:                   {}'.format(cfg.nTransferLearning))
    print('cfg.sDirectoryForTrainImages:            {}'.format(cfg.sDirectoryForTrainImages))
    print('cfg.sFilePathForTrainImages:             {}'.format(cfg.sFilePathForTrainImages))
    print('cfg.sDirectoryForTestImages:             {}'.format(cfg.sDirectoryForTestImages))
    print('cfg.sFilePathForTestImages:              {}'.format(cfg.sFilePathForTestImages))
    print('cfg.nNbrOfClasses:                       {}'.format(cfg.nNbrOfClasses))
    print('cfg.sFilenameExtension:                  {}'.format(cfg.sFilenameExtension))
    print('cfg.sOptimizer:                          {}'.format(cfg.sOptimizer))
    print('cfg.sDirectoryForWeights:                {}'.format(cfg.sDirectoryForWeights))
    print('cfg.sDirectoryForTrainingResultsRoot:    {}'.format(cfg.sDirectoryForTrainingResultsRoot))
    print('cfg.sDirectoryForTrainingResultsCurrent: {}'.format(cfg.sDirectoryForTrainingResultsCurrent))
    print('cfg.nEpochs:                             {}'.format(cfg.nEpochs))
    print('cfg.nBatchSize:                          {}'.format(cfg.nBatchSize))
    print('cfg.dDropoutRatio:                       {}'.format(cfg.dDropoutRatio))
    print('cfg.dValidationSplit:                    {}'.format(cfg.dValidationSplit))
    print('cfg.nImageRows                           {}'.format(cfg.nImageRows))
    print('cfg.nImageColumns                        {}'.format(cfg.nImageColumns))
    print('cfg.nImageChannels                       {}'.format(cfg.nImageChannels))
    print('cfg.nNbrOfClasses                        {}'.format(cfg.nNbrOfClasses))
    print('cfg.nEpochs                              {}'.format(cfg.nEpochs))
    print('cfg.nBatchSize                           {}'.format(cfg.nBatchSize))
    print('cfg.dDropoutRatio                        {}'.format(cfg.dDropoutRatio))
    print('cfg.dValidationSplit                     {}'.format(cfg.dValidationSplit))
    print('cfg.nNbrOfGPUs                           {}'.format(cfg.nNbrOfGPUs))
    print('cfg.sImageChannelType                    {}'.format(cfg.sImageChannelType))
    print('cfg.nUseClaheForInputImage               {}'.format(cfg.nUseClaheForInputImage))
    print('cfg.nPreprocessInput:                    {}'.format(cfg.nPreprocessInput ))
    print('cfg.nImageIntensityNormalizationToMax1   {}'.format(cfg.nImageIntensityNormalizationToMax1))
    print('cfg.nImageIntensityNormalization         {}'.format(cfg.nImageIntensityNormalization))
    print('cfg.nImageIntensityNormalizationUseRGBChannel {}'.format(cfg.nImageIntensityNormalizationUseRGBChannel))
    print('TrainX.shape                             {}'.format(TrainX.shape))
    print('TrainY.shape                             {}'.format(TrainY.shape))
    print('nNbrOfDataTrain                          {}'.format(nNbrOfDataTrain))
    print('sWeightsH_FilePath                       {}'.format(sWeightsH_FilePath))
   
    
    
    fileWrite.write('-'*30)
    fileWrite.write('\n')
    fileWrite.write('Get {} model  \n'.format(cfg.sCNNType))
    fileWrite.write('-'*30)    
    fileWrite.write('\n')
    fileWrite.write('cfg.bUseSlurm:                           {} \n'.format(cfg.bUseSlurm))
    fileWrite.write('cfg.sOperation:                          {} \n'.format(cfg.sOperation))
    fileWrite.write('cfg.sCNNType:                            {} \n'.format(cfg.sCNNType))
    fileWrite.write('cfg.nTransferLearning:                   {} \n'.format(cfg.nTransferLearning))
    fileWrite.write('cfg.sDirectoryForTrainImages:            {} \n'.format(cfg.sDirectoryForTrainImages))
    fileWrite.write('cfg.sFilePathForTrainImages:             {} \n'.format(cfg.sFilePathForTrainImages))
    fileWrite.write('cfg.sDirectoryForTestImages:             {} \n'.format(cfg.sDirectoryForTestImages))
    fileWrite.write('cfg.sFilePathForTestImages:              {} \n'.format(cfg.sFilePathForTestImages))
    fileWrite.write('cfg.nNbrOfClasses:                       {} \n'.format(cfg.nNbrOfClasses))
    fileWrite.write('cfg.sFilenameExtension:                  {} \n'.format(cfg.sFilenameExtension))
    fileWrite.write('cfg.sOptimizer:                          {} \n'.format(cfg.sOptimizer))
    fileWrite.write('cfg.sDirectoryForWeights:                {} \n'.format(cfg.sDirectoryForWeights))
    fileWrite.write('cfg.sDirectoryForTrainingResultsRoot:    {} \n'.format(cfg.sDirectoryForTrainingResultsRoot))
    fileWrite.write('cfg.sDirectoryForTrainingResultsCurrent: {} \n'.format(cfg.sDirectoryForTrainingResultsCurrent))
    fileWrite.write('cfg.nEpochs:                             {} \n'.format(cfg.nEpochs))
    fileWrite.write('cfg.nBatchSize:                          {} \n'.format(cfg.nBatchSize))
    fileWrite.write('cfg.dDropoutRatio:                       {} \n'.format(cfg.dDropoutRatio))
    fileWrite.write('cfg.dValidationSplit:                    {} \n'.format(cfg.dValidationSplit))
    fileWrite.write('cfg.nImageRows                           {} \n'.format(cfg.nImageRows))
    fileWrite.write('cfg.nImageColumns                        {} \n'.format(cfg.nImageColumns))
    fileWrite.write('cfg.nImageChannels                       {} \n'.format(cfg.nImageChannels))
    fileWrite.write('cfg.nNbrOfClasses                        {} \n'.format(cfg.nNbrOfClasses))
    fileWrite.write('cfg.nEpochs                              {} \n'.format(cfg.nEpochs))
    fileWrite.write('cfg.nBatchSize                           {} \n'.format(cfg.nBatchSize))
    fileWrite.write('cfg.dDropoutRatio                        {} \n'.format(cfg.dDropoutRatio))
    fileWrite.write('cfg.dValidationSplit                     {} \n'.format(cfg.dValidationSplit))
    fileWrite.write('cfg.nNbrOfGPUs                           {} \n'.format(cfg.nNbrOfGPUs))
    fileWrite.write('cfg.sImageChannelType                    {} \n'.format(cfg.sImageChannelType))
    fileWrite.write('cfg.nUseClaheForInputImage               {} \n'.format(cfg.nUseClaheForInputImage))
    fileWrite.write('cfg.nPreprocessInput:                    {} \n'.format(cfg.nPreprocessInput ))
    fileWrite.write('cfg.nImageIntensityNormalizationToMax1   {} \n'.format(cfg.nImageIntensityNormalizationToMax1))
    fileWrite.write('cfg.nImageIntensityNormalization         {} \n'.format(cfg.nImageIntensityNormalization))
    fileWrite.write('cfg.nImageIntensityNormalizationUseRGBChannel {} \n'.format(cfg.nImageIntensityNormalizationUseRGBChannel))    
    fileWrite.write('TrainX.shape                             {} \n'.format(TrainX.shape))
    fileWrite.write('TrainY.shape                             {} \n'.format(TrainY.shape))
    fileWrite.write('nNbrOfDataTrain                          {} \n'.format(nNbrOfDataTrain))
    fileWrite.write('sWeightsH_FilePath                       {} \n'.format(sWeightsH_FilePath))
    
    #compute the training time   
    tTrainingTimeStart=datetime.datetime.now() 
    print('-'*60)
    print('Start Training the Custom Model...')
    print('-'*60)    
    
     

    model, sModelSummary = GetBaseCnnModel(cfg) 

    print('{}'.format(sModelSummary))
    fileWrite.write('{}'.format(sModelSummary))
    
    
    
    model_checkpoint = ModelCheckpoint( sWeightsH_FilePath, monitor='val_loss', save_best_only=True)


    history = model.fit(TrainX, TrainY, epochs=cfg.nEpochs, batch_size=cfg.nBatchSize, validation_split=cfg.dValidationSplit, verbose=2, shuffle=True, callbacks=[model_checkpoint])



   #compute the training time
    tTrainingTimeEnd=datetime.datetime.now() 
    tTrainingTimeAll = tTrainingTimeEnd - tTrainingTimeStart 
    
    print('Time Start   (hh:mm:ss.ms): {} '.format(tTrainingTimeStart))
    print('Time End     (hh:mm:ss.ms): {} '.format(tTrainingTimeEnd))
    print('Time Elapsed (hh:mm:ss.ms): {} '.format(tTrainingTimeAll))
    
    fileWrite.write('Time Start   (hh:mm:ss.ms): {} \n'.format(tTrainingTimeStart))
    fileWrite.write('Time End     (hh:mm:ss.ms): {} \n'.format(tTrainingTimeEnd))
    fileWrite.write('Time Elapsed (hh:mm:ss.ms): {} \n'.format(tTrainingTimeAll))
            
    print('===Save Trained model===')
    sFileName = '{}-cl{}-Ro{}-Co{}-Ch{}-Vs{}-Do{}-Au{}-Pr{}-Tr{}-{}-Ep{}.h5'.format(cfg.sCNNType,cfg.nUseClaheForInputImage, cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels, nValidationSplit, nDropOutRatio, cfg.nAugmentationNumber, cfg.nPreprocessInput, cfg.nTransferLearning, cfg.sOptimizer, cfg.nEpochs)

    sFileNameModel  = os.path.join(cfg.sDirectoryForTrainingResultsCurrent,sFileName )
    sFileNameModel1 = os.path.join(cfg.sDirectoryForTrainingResultsCurrent,'weights.h5' )

    model.save(sFileNameModel)
    model.save(sFileNameModel1)
    print('Save the Model to :', sFileNameModel)
    

   
    print('-'*60)
    print('===Training History===')   
    
    fileWrite.write('-'*30)
    fileWrite.write('\n')
    fileWrite.write('===Training History===')
    fileWrite.write('-'*30)    
    fileWrite.write('\n')
      
    sHistoryOut = (str)(history.history) 
    fileWrite.write(sHistoryOut)
    fileWrite.write('\n')

    fileWrite.write('-'*30)    
    fileWrite.write('\n \n')
    
    
    if( cfg.bUseSlurm == False or str(cfg.bUseSlurm) == 'False'  or str(cfg.bUseSlurm) == 'GPUs2' ):   
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()   
           
       

  
    fileWrite.close()       
   


# read a input file  (sFileForTestImages) that has the full path of testing images
#20240319 add an option to changge input color immage into grey image
#20200723 add a function to reassign class labels
def TestCNNUsingImageInputFile( cfg ):       

    dt=datetime.datetime.today()
    stime='{}{}{}{}{}{}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    
    # update input image size
    if( cfg.nImageRows == 0 and cfg.nImageColumns == 0 ):    
        UpdateInputImageSize(cfg)

           
    nDropOutRatio = int(float(cfg.dDropoutRatio)*100.0)
    nValidationSplit = int(float(cfg.dValidationSplit)*100.0)

    sFileName = '{}-Vs{}-Do{}-Au{}-Pr{}-Tl{}-{}-Ep{}-TestResults{}.txt'.format(cfg.sCNNType,nValidationSplit, nDropOutRatio, cfg.nAugmentationNumber, cfg.nPreprocessInput, cfg.nTransferLearning, cfg.sOptimizer, cfg.nEpochs, stime)
    

    
    print(' ')
    print('===TestNetUsingImageInputFile===')     
    
            
    if( cfg.sDirectoryForWeights == '' ):
         cfg.sDirectoryForWeights = cfg.sDirectoryForTrainingResultsCurrent

    print('cfg.sDirectoryForWeights       : {0} '.format(cfg.sDirectoryForWeights) )  


    sResultFilePath = os.path.join(cfg.sDirectoryForWeights, sFileName )
    print('sResultFilePath: ', sResultFilePath)
    
    
    if( cfg.sHeatMapMethod != '' ):
         cfg.sDirectoryForSavingHeatMapImage = os.path.join(cfg.sDirectoryForWeights, 'HeatMap' )
          
         if (not os.path.exists(cfg.sDirectoryForSavingHeatMapImage) ):
             os.mkdir(cfg.sDirectoryForSavingHeatMapImage)        
    
    
    fileWrite = open(sResultFilePath,"w") 
   
    sWeight = os.path.join(cfg.sDirectoryForWeights, 'weights.h5' )
 
    print('-'*30)
    print('===TestNetUsingImageInput=== \n') 
    print('cfg.sDirectoryForWeights                         : {} '.format(cfg.sDirectoryForWeights) )  
    print('cfg.sFilePathForTestImages                       : {} '.format(cfg.sFilePathForTestImages) )   
    print('cfg.sDirectoryForTestImages                      : {} '.format(cfg.sDirectoryForTestImages) )   
    print('sWeight                                          : {} '.format(sWeight) )  
    print('cfg.nImageRows                                   : {} '.format(cfg.nImageRows))
    print('cfg.nImageColumns                                : {} '.format(cfg.nImageColumns) )   
    print('cfg.nImageChannels                               : {} '.format(cfg.nImageChannels) )  
    print('cfg.nNbrOfClasses                                : {} '.format(cfg.nNbrOfClasses) )  
    print('cfg.sClassLabelNames                             : {} '.format(cfg.sClassLabelNames) )  
    print('cfg.nUseClaheForInputImage                       : {} '.format(cfg.nUseClaheForInputImage) )  
    print('cfg.nPreprocessInput                             : {} '.format(cfg.nPreprocessInput) )  
    print('cfg.nImageIntensityNormalizationToMax1           : {} '.format(cfg.nImageIntensityNormalizationToMax1) )  
    print('cfg.nImageIntensityNormalization                 : {} '.format(cfg.nImageIntensityNormalization) )  
    print('cfg.nImageIntensityNormalizationUseRGBChannel    : {} '.format(cfg.nImageIntensityNormalizationUseRGBChannel) ) 
    
    print('cfg.sHeatMapMethod                               : {} '.format(cfg.sHeatMapMethod) )  
    print('cfg.nPositiveThreshold                           : {} '.format(cfg.nPositiveThreshold) )  
    print('cfg.nConvolutionalLayer                          : {} '.format(cfg.nConvolutionalLayer) )  
    print('cfg.sDirectoryForSavingHeatMapImage              : {} '.format(cfg.sDirectoryForSavingHeatMapImage) )  
    print('-'*30)

   
    fileWrite.write('-'*30)
    fileWrite.write('\n')
    fileWrite.write('===TestNetUsingImageInput=== \n') 
    fileWrite.write('cfg.sDirectoryForWeights                       : {} \n'.format(cfg.sDirectoryForWeights) )  
    fileWrite.write('cfg.sFilePathForTestImages                     : {} \n'.format(cfg.sFilePathForTestImages) )   
    fileWrite.write('cfg.sDirectoryForTestImages                    : {} \n'.format(cfg.sDirectoryForTestImages) )   
    fileWrite.write('sWeight                                        : {} \n'.format(sWeight) )  
    fileWrite.write('cfg.nImageRows                                 : {} \n'.format(cfg.nImageRows))
    fileWrite.write('cfg.nImageColumns                              : {} \n'.format(cfg.nImageColumns) )   
    fileWrite.write('cfg.nImageChannels                             : {} \n'.format(cfg.nImageChannels) )  
    fileWrite.write('cfg.nNbrOfClasses                              : {} \n'.format(cfg.nNbrOfClasses) )  
    fileWrite.write('cfg.sClassLabelNames                           : {} \n'.format(cfg.sClassLabelNames) )  
    fileWrite.write('cfg.nUseClaheForInputImage                     : {} \n'.format(cfg.nUseClaheForInputImage) )  
    fileWrite.write('cfg.nPreprocessInput                           : {} \n'.format(cfg.nPreprocessInput) )  
    fileWrite.write('cfg.nImageIntensityNormalizationToMax1         : {} \n'.format(cfg.nImageIntensityNormalizationToMax1) )  
    fileWrite.write('cfg.nImageIntensityNormalization               : {} \n'.format(cfg.nImageIntensityNormalization) )  
    fileWrite.write('cfg.nImageIntensityNormalizationUseRGBChannel  : {} \n'.format(cfg.nImageIntensityNormalizationUseRGBChannel) )  
    
    fileWrite.write('cfg.sHeatMapMethod                             : {} \n'.format(cfg.sHeatMapMethod) )  
    fileWrite.write('cfg.nPositiveThreshold                         : {} \n'.format(cfg.nPositiveThreshold) )  
    fileWrite.write('cfg.nConvolutionalLayer                        : {} \n'.format(cfg.nConvolutionalLayer) )  
    fileWrite.write('cfg.sDirectoryForSavingHeatMapImage            : {} \n'.format(cfg.sDirectoryForSavingHeatMapImage) )  
    
    fileWrite.write('-'*30)
    fileWrite.write('\n')
    
       
            
    model = load_model(sWeight)


    sListForTestDataFileNames=[]
    oFileOpen  = open(cfg.sFilePathForTestImages, 'r') 
    for aline in oFileOpen:

        sImageName = str(aline)            
        if sImageName.find('.tif') > 0 or sImageName.find('.jpeg') > 0 or sImageName.find('.png') > 0  or sImageName.find('.bmp') > 0: 
            sListForTestDataFileNames.append(aline)

    oFileOpen.close()
    
    
    
    print('nNbrOfTestData                                   : {} '.format(len(sListForTestDataFileNames)) )  
    fileWrite.write('nNbrOfTestData                         : {} \n'.format(len(sListForTestDataFileNames)) )  





    #20200723 remove files not using or remplace existing labels to new labels

    print('NbrOfData before Class Label Reassignment:               {}'.format( len(sListForTestDataFileNames)))
    print('NbrOfClasses before Class Label Reassignment:            {}'.format(cfg.nNbrOfClasses))
    fileWrite.write('NbrOfData before Class Label Reassignment:            {} \n'.format( len(sListForTestDataFileNames)))
    fileWrite.write('NbrOfClasses before Class Label Reassignment:         {} \n'.format(cfg.nNbrOfClasses))
    
    sListForTestDataFileNames, nNbrOfClasses = ReassignClassLabelsInList( cfg, sListForTestDataFileNames )
    
    if(int(nNbrOfClasses) > 0 ):
        cfg.nNbrOfClasses = int(nNbrOfClasses)

    print('NbrOfData After Class Label Reassignment:               {}'.format(len(sListForTestDataFileNames)))
    print('NbrOfClasses After Class Label Reassignment:            {}'.format(cfg.nNbrOfClasses))
    fileWrite.write('NbrOfData After Class Label Reassignment:            {} \n'.format( len(sListForTestDataFileNames) ))
    fileWrite.write('NbrOfClasses After Class Label Reassignment:         {} \n'.format(cfg.nNbrOfClasses))


    print(' \n\n')
    fileWrite.write('\n \n')
    
    print ('=== Error Images === ')



    nStatistics = [[0] * cfg.nNbrOfClasses for i in range(cfg.nNbrOfClasses)] 
    

    nNbrOfData=0
    for aFileName in sListForTestDataFileNames[:]:

        aClassIndex, aFileNamePath  = aFileName.split(";")  
        sClassIndex = (str)(aClassIndex)
        sClassIndex = sClassIndex.lstrip()
        sClassIndex = sClassIndex.rstrip()       
        
        sImageName = (str)(aFileNamePath)
        sImageName = sImageName.strip('\n')
        sImageName = sImageName.lstrip()
        sImageName = sImageName.rstrip() 
                    
        if sImageName.find('.tif') > 0 or sImageName.find('.jpeg') > 0 or sImageName.find('.png') > 0  or sImageName.find('.bmp') > 0:
            
            
            if( cfg.nImageChannels == 3):
                DataX = np.ndarray((1, cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels), dtype=np.uint8)
            else:
                DataX = np.ndarray((1, cfg.nImageRows, cfg.nImageColumns), dtype=np.uint8)
                

            if( os.path.isfile( os.path.join(cfg.sDirectoryForTestImages, sImageName))  == False ):
                 sImageName =sImageName.replace('.tif','.jpg')
                 
                 if( os.path.isfile( os.path.join(cfg.sDirectoryForTestImages, sImageName))  == False ):
                     sImageName =sImageName.replace('.jpg','.jpeg')
            
                                      
            if( cfg.nImageChannels == 3):
                img = cv2.imread(os.path.join(cfg.sDirectoryForTestImages, sImageName), cv2.IMREAD_COLOR)  
            else:
                img = cv2.imread(os.path.join(cfg.sDirectoryForTestImages, sImageName), cv2.IMREAD_GRAYSCALE)   
              
            imgGrey = cv2.imread(os.path.join(cfg.sDirectoryForTestImages, sImageName), cv2.IMREAD_GRAYSCALE)   
                
            imgOriginal = img.copy()
            #20200524    
            if( int(cfg.nUseClaheForInputImage)  == 1 ):
                img = ClaheOperatorColor3D( img.copy(), cfg.nClaheTileSize)
                

            
            
            
            # 20230424 to change image channel format
            if( cfg.nImageChannels == 3):
                if( cfg.sImageChannelType == 'YCbCr' or cfg.sImageChannelType == 'ycbcr' ):

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)   #(if input image is RGB) 
                            
                elif( cfg.sImageChannelType == 'HSV' or cfg.sImageChannelType == 'hsv' ):
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  #(HSV (Hue, Saturation, Value) 
                            
                elif( cfg.sImageChannelType == 'HLS' or cfg.sImageChannelType == 'hls' ):
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)  #(HSL (Hue, Saturation, Lightness)
                            
                elif( cfg.sImageChannelType == 'LAB' or cfg.sImageChannelType == 'lab' ):                            

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)  #(HSL (Hue, Saturation, Lightness)                
            
                elif( cfg.sImageChannelType == '3Grey' or cfg.sImageChannelType == '3grey' ):     #20240319                       
                    img[:,:,0] = imgGrey
                    img[:,:,1] = imgGrey
                    img[:,:,2] = imgGrey            
            
            
            
            try:
                
                if( img.shape[1] > cfg.nImageRows ):
 
                    img = cv2.resize(img, (cfg.nImageRows,cfg.nImageColumns), interpolation = cv2.INTER_AREA)  
                else:
 
                    img = cv2.resize(img, (cfg.nImageRows,cfg.nImageColumns), interpolation = cv2.INTER_CUBIC)
                
            except:

                print('Image Error:[{}] {}'.format(sImageName, sImageName.find ))
                continue
            
            #Original 
            DataX[0] = np.array([img])

            
  
    
            DataX1 = DataX.copy() 

            if( int(cfg.nPreprocessInput) == 1 ):
 
                DataX1 = GetPreprocessInput(cfg, DataX.copy())                  
           
            if( int(cfg.nImageIntensityNormalizationToMax1) == 1 ):    
                DataX1 = DataX.copy()/255.0  
            
            if( int(cfg.nImageIntensityNormalizationUseRGBChannel) == 1 ):    
                DataX1 = NormalizeImagesIntensityUsingMeanAndStd( DataX.copy() , cfg.nImageIntensityNormalizationUseRGBChannel, True, True )    

    
           
            dPrediction = model.predict(DataX1, verbose=2)
                            
            #20200728 add to handle two class problem
            if( cfg.nNbrOfClasses >= 2 ):
                dMax=0.0
                nPredictionIndex = 0
                
                for i in range(len(dPrediction[0])):
                    if( dPrediction[0][i] > dMax):
                        dMax = dPrediction[0][i]
                        nPredictionIndex = i
            else:
                if( dPrediction[0,0] >= 0.5 ):
                    nPredictionIndex = 1
                else:
                    nPredictionIndex = 0
            

            nClassIndex = int(sClassIndex)

            nStatistics[nClassIndex][nPredictionIndex] = int(nStatistics[nClassIndex][nPredictionIndex]) + 1

            if( nClassIndex != nPredictionIndex ):
                if( cfg.nNbrOfClasses == 2 ):
 
                    fileWrite.write('Error [{0:5}]: {1}->{2} [{3:.5f},{4:.5f}] : {5} \n'.format(nNbrOfData, nClassIndex, nPredictionIndex, (1.0-dPrediction[0,0]), dPrediction[0,0], sImageName ))                       
                    print('Error [{0:5}]: {1}->{2} [{3:.5f},{4:.5f}] : {5} '.format(nNbrOfData, nClassIndex, nPredictionIndex, (1.0-dPrediction[0,0]), dPrediction[0,0], sImageName ))                       

                else:

                    fileWrite.write('Error [{0:5}]: {1}({2:.5f})->{3} ({4:.5f}) / [0({5:.5f}), 1({6:.5f}), 2({7:.5f}), 3({8:.5f})]; {9} \n'.format(nNbrOfData, nClassIndex, dPrediction[0][nClassIndex], nPredictionIndex, dPrediction[0][nPredictionIndex], dPrediction[0][0], dPrediction[0][1], dPrediction[0][2], dPrediction[0][3], sImageName ))
                    print('Error [{0:5}]: {1}({2:.5f})->{3} ({4:.5f}) / [0({5:.5f}), 1({6:.5f}), 2({7:.5f}), 3({8:.5f})]; {9} '.format(nNbrOfData, nClassIndex, dPrediction[0][nClassIndex], nPredictionIndex, dPrediction[0][nPredictionIndex], dPrediction[0][0], dPrediction[0][1], dPrediction[0][2], dPrediction[0][3], sImageName ))
             
            #Print out classes changed              
            if( nNbrOfData%1000 == 0 and False):  
                if( cfg.nNbrOfClasses == 2 ):
                    print('      [{0:5}]: {1}->{2} [{3:.5f}, {4:.5f}) ; {5}'.format(nNbrOfData, nClassIndex, nPredictionIndex,  (1.0-dPrediction[0,0]), dPrediction[0,0], sImageName ))
                else:    

                    print( '      [{0:5}]: {1}({2:.5f})->{3} ({4:.5f}) / [0({5:.5f}), 1({6:.5f}), 2({7:.5f}), 3({8:.5f})]; {9}'.format(nNbrOfData, nClassIndex, dPrediction[0][nClassIndex], nPredictionIndex, dPrediction[0][nPredictionIndex], dPrediction[0][0], dPrediction[0][1], dPrediction[0][2], dPrediction[0][3], sImageName ))
            
            
 
            
            nNbrOfData+=1
           
    
    if( cfg.nNbrOfClasses == 2 ):
        Accuracy        =   (float)( (float)(nStatistics[0][0]+nStatistics[1][1])/(float)(nNbrOfData))
        
        if((nStatistics[0][0]+nStatistics[1][0]) > 0 ):
            Precision       =   (float)( (float)(nStatistics[0][0])/(float)(nStatistics[0][0]+nStatistics[1][0])) 
        else:
            Precision       = 0

        if((nStatistics[0][0]+nStatistics[0][1]) > 0 ):
            Recall          =   (float)( (float)(nStatistics[0][0])/(float)(nStatistics[0][0]+nStatistics[0][1]))    
        else:
            Recall       = 0

        if((nStatistics[1][0]+nStatistics[1][1]) > 0 ):
            TrueNegative    =   (float)( (float)(nStatistics[1][1])/(float)(nStatistics[1][0]+nStatistics[1][1]))    
        else:
            TrueNegative       = 0

        if((Precision+ Recall) > 0 ):
            FMeasure        =   (float)( (2.0*Precision*Recall)/(Precision+ Recall))    
        else:
            FMeasure       = 0

        if((nStatistics[0][0]+nStatistics[0][1]) > 0 ):
            Sensitivity     =   (float)( (float)(nStatistics[0][0])/(float)(nStatistics[0][0]+nStatistics[0][1]))    
        else:
            Sensitivity       = 0

        if((nStatistics[1][0]+nStatistics[1][1]) > 0 ):
            Specificity     =   (float)( (float)(nStatistics[1][1])/(float)(nStatistics[1][0]+nStatistics[1][1]))   
        else:
            Specificity       = 0
            
    else:
        Accuracy        =   0    
        Precision       =   0    
        Recall          =   0    
        TrueNegative    =   0    
        FMeasure        =   0    
        Sensitivity     =   0    
        Specificity     =   0  
        
        nCount=0
        for i in range(0,cfg.nNbrOfClasses,1):
            nCount += nStatistics[i][i]
            
        Accuracy        =   (float)(nCount)/(float)(nNbrOfData)
        
        
             
    print('-'*30)    
    print( '== Result Statistics ==' )
    print('-'*30)    
        
    fileWrite.write('-'*30)
    fileWrite.write('\n')
    fileWrite.write( '== Result Statistics== \n' )
    fileWrite.write('-'*30)
    fileWrite.write('\n')
    
    if( cfg.nNbrOfClasses == 2 ):
        for i in range(0,cfg.nNbrOfClasses,1):
            print( 'C[{0}] : {1:5}, {2:5} '.format(i,nStatistics[i][0], nStatistics[i][1]) )
            fileWrite.write( 'C[{0}] : {1:5}, {2:5} \n'.format(i,nStatistics[i][0], nStatistics[i][1]) )
    else:
        print( '{0:7}   {1:5}, {2:5}, {3:5}, {4:5}, {5:5}, {6:5}, {7}'.format('','[CNV]', '[DME]', '[DUR]', '[NOR]', 'Total', '#Acuracy', '%Acuracy') )

        for i in range(0,cfg.nNbrOfClasses,1):

            #12 classes
            cCount = 0
            for j in range(0,cfg.nNbrOfClasses):
                cCount += nStatistics[i][j]
            
            cAcuracy = (float)(nStatistics[i][i])/(float)(cCount)
                
            #4 CLASSES
            print( 'C[{0:2}] : {1:5}, {2:5}, {3:5}, {4:5}, {5:6}, {6:6}, {7:9}'.format(cfg.sLabelNames[i],nStatistics[i][0], nStatistics[i][1], nStatistics[i][2], nStatistics[i][3], cCount, nStatistics[i][i], cAcuracy ))
            fileWrite.write( 'C[{0:2}] : {1:5}, {2:5}, {3:5}, {4:5}, {5:6}, {6:6}, {7:9} \n'.format(cfg.sLabelNames[i],nStatistics[i][0], nStatistics[i][1], nStatistics[i][2], nStatistics[i][3], cCount, nStatistics[i][i], cAcuracy ))
 
        
    print('-'*30)    
    fileWrite.write('-'*30)
    fileWrite.write('\n')

    print( 'Accuracy:       {0} '.format(Accuracy) )
    print( 'Sensitivity:    {0} '.format(Sensitivity) )
    print( 'Specificity:    {0} '.format(Specificity) )
    print( 'Precision:      {0} '.format(Precision) )
    print( 'Recall:         {0} '.format(Recall) )
    print( 'TrueNegative:   {0} '.format(TrueNegative) )
    print( 'FMeasure:       {0} '.format(FMeasure) )
    print('-'*30)    

    fileWrite.write( 'Accuracy:       {0} \n'.format(Accuracy) )
    fileWrite.write( 'Sensitivity:    {0} \n'.format(Sensitivity) )
    fileWrite.write( 'Specificity:    {0} \n'.format(Specificity) )
    fileWrite.write( 'Precision:      {0} \n'.format(Precision) )
    fileWrite.write( 'Recall:         {0} \n'.format(Recall) )
    fileWrite.write( 'TrueNegative:   {0} \n'.format(TrueNegative) )
    fileWrite.write( 'FMeasure:       {0} \n'.format(FMeasure) )
    fileWrite.write('-'*30)
    fileWrite.write('\n')
    
    

    
    sTotalAccuracy = '[ Acc:{}, Sen:{}, Spe:{}, Pre:{}, Rec:{}, Tn{}, Fme:{} ]'.format(Accuracy, Sensitivity, Specificity, Precision, Recall, TrueNegative, FMeasure  )
    
    print(sTotalAccuracy)
    print('-'*30)    
    
    fileWrite.write(sTotalAccuracy)    
    fileWrite.write('-'*30)
    fileWrite.write('\n')
    
    
    
    
    fileWrite.write('-'*30)  
    fileWrite.write('\n')    
    fileWrite.write( '===Error List===\n' )
    fileWrite.write('-'*30)    
    fileWrite.write('\n')
    
  
        
    print( ' ')
    print( 'cfg.sDirectoryForOutput                        : {} '.format(cfg.sDirectoryForOutput))
        
    fileWrite.write(' \n' )       
    fileWrite.write('cfg.sDirectoryForOutput                        : {} \n'.format(cfg.sDirectoryForOutput) )       
        
        
    fileWrite.close() 
    
    
  



# read a input file  (sFileForTestImages) that has the full path of testing images
#20230319 add a code to change color input image into grey image      
#20200723 add a function to reassign class labels
def TestCNNUsingImageInputFileEnsemble3( cfg ):       
#def TestNetUsingImageInputFile(sDirectoryForSavingTestResults, sImageFileForTest, sWeight, nImageRows, nImageColumns, nImageChannels, nNbrOfClasses ):       

    dt=datetime.datetime.today()
    stime='{}{}{}{}{}{}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    
    # update input image size
    if( cfg.nImageRows == 0 and cfg.nImageColumns == 0 ):    
        UpdateInputImageSize(cfg)

           
    nValidationSplit = int(float(cfg.dValidationSplit)*100.0)
    
    sCNNModelNames =['ConvNeXtXLarge','ConvNeXtLarge','ConvNeXtBase','ConvNeXtSmall','ConvNeXtTiny',
                     'EfficientNetV2L','EfficientNetV2M','EfficientNetV2S', 'EfficientNetV2B3','EfficientNetV2B2','EfficientNetV2B1','EfficientNetV2B0',
                     'EfficientNetB7','EfficientNetB6','EfficientNetB5','EfficientNetB4','EfficientNetB3','EfficientNetB2','EfficientNetB1','EfficientNetB0',
                     'NASNetLarge','NASNetMobile','DenseNet201','DenseNet169','DenseNet121','MobileNetV2','MobileNet','InceptionResNetV2','InceptionV3',
                     'ResNet152V2','ResNet152','ResNet101V2','ResNet101','ResNet50V2','ResNet50','VGG19','VGG16','Xception']

    sCNNModels =''
    sCNNModelList = []
    

    
    for i in range(len(sCNNModelNames)):
        if( cfg.sDirectoryForWeights.find(sCNNModelNames[i]) > 0 ):
            if( sCNNModels != '' ):
                sCNNModels += '_'
            sCNNModels += '{}'.format(sCNNModelNames[i])
            sCNNModelList.append(sCNNModelNames[i])

    for i in range(len(sCNNModelNames)):
        if( cfg.sDirectoryForWeights2.find(sCNNModelNames[i]) > 0 ):
            if( sCNNModels != '' ):
                sCNNModels += '_'
            sCNNModels += '{}'.format(sCNNModelNames[i])
            sCNNModelList.append(sCNNModelNames[i])

    for i in range(len(sCNNModelNames)):
        if(  cfg.sDirectoryForWeights3.find(sCNNModelNames[i]) > 0 ):
            if( sCNNModels != '' ):
                sCNNModels += '_'
            sCNNModels += '{}'.format(sCNNModelNames[i])
            sCNNModelList.append(sCNNModelNames[i])
            
    sFileName = '{}-{}-Vs{}-Au{}-Pr{}-Tl{}-{}-TestResults{}.txt'.format(sCNNModels,cfg.sVotingMethod,nValidationSplit, cfg.nAugmentationNumber, cfg.nPreprocessInput, cfg.nTransferLearning, cfg.sOptimizer, stime)
    
    
    print(' ')
    print('===TestNetUsingImageInputFile===')     
    
        
    
    if( cfg.sDirectoryForOutputRoot != '' and cfg.sDirectoryForWeights != '' and cfg.sDirectoryForWeights2 != '' and cfg.sDirectoryForWeights != ''):        
            
        sClassAssignment =''
        if( len(cfg.sClassAssignment) > 0 ):
            for i in range (100):
                sClassAssignment =sClassAssignment.replace('{},'.format(i),'')
            sClassAssignment = sClassAssignment.replace(';','')
        
        sClassAssignment= cfg.sClassAssignment
    
        sDirectoryName = '{}-{}-Ca{}-Cl{}-Ro{}-Co{}-Ch{}-Vs{}-TVG{}-Au{}-Pr{}-Tl{}-Bs{}-{}-{}-{}'.format( sCNNModels, cfg.sVotingMethod, sClassAssignment,cfg.nUseClaheForInputImage, cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels, nValidationSplit, cfg.dTrainValidDataGeneration, cfg.nAugmentationNumber, cfg.nPreprocessInput, cfg.nTransferLearning, cfg.nBatchSize, cfg.sOptimizer, cfg.sFilenameExtension, stime )        
       
        cfg.sDirectoryForOutput = os.path.join( cfg.sDirectoryForOutputRoot, sDirectoryName )
        
        
        #Create root directory for all outputs
        if (not os.path.exists(cfg.sDirectoryForOutputRoot) ):
            os.mkdir(cfg.sDirectoryForOutputRoot)
            
        #Create a directory for ensemble learning    
        if (os.path.exists(cfg.sDirectoryForOutput) ):
            shutil.rmtree(cfg.sDirectoryForOutput)
    
        os.mkdir(cfg.sDirectoryForOutput)
        
        
    else:
        
        cfg.sDirectoryForOutput = cfg.sDirectoryForWeights
        cfg.sDirectoryForOutputRoot = cfg.sDirectoryForWeights
        
        
        
    sResultFilePath = os.path.join(cfg.sDirectoryForOutput, sFileName )       


    print('sResultFilePath: ', sResultFilePath)
    
    
    if( cfg.sHeatMapMethod != '' ):
         cfg.sDirectoryForSavingHeatMapImage = os.path.join(cfg.sDirectoryForOutput, 'HeatMap' )
          
         if (not os.path.exists(cfg.sDirectoryForSavingHeatMapImage) ):
             os.mkdir(cfg.sDirectoryForSavingHeatMapImage)        
    
    
    fileWrite = open(sResultFilePath,"w") 
   

 
    print('-'*30)
    print('===TestNetUsingImageInput=== \n') 
    print('cfg.sDirectoryForWeights                         : {} '.format(cfg.sDirectoryForWeights) )  
    print('cfg.sDirectoryForWeights2                        : {} '.format(cfg.sDirectoryForWeights2) )  
    print('cfg.sDirectoryForWeights3                        : {} '.format(cfg.sDirectoryForWeights3) )  
    print('cfg.sVotingMethod                                : {} '.format(cfg.sVotingMethod) ) 
    
    print('cfg.sDirectoryForOutputRoot                      : {} '.format(cfg.sDirectoryForOutputRoot) ) 
    print('cfg.sDirectoryForOutput                          : {} '.format(cfg.sDirectoryForOutput) ) 
    
    
    print('cfg.sFilePathForTestImages                       : {} '.format(cfg.sFilePathForTestImages) )   
    print('cfg.sDirectoryForTestImages                      : {} '.format(cfg.sDirectoryForTestImages) )   

    print('cfg.sWeight                                      : {} '.format(cfg.sWeight) )  
    print('cfg.nImageRows                                   : {} '.format(cfg.nImageRows))
    print('cfg.nImageColumns                                : {} '.format(cfg.nImageColumns) )   
    print('cfg.nImageChannels                               : {} '.format(cfg.nImageChannels) )  
    print('cfg.nNbrOfClasses                                : {} '.format(cfg.nNbrOfClasses) )  
    print('cfg.sClassLabelNames                             : {} '.format(cfg.sClassLabelNames) )  
    print('cfg.nUseClaheForInputImage                       : {} '.format(cfg.nUseClaheForInputImage) )  
    print('cfg.nImageIntensityNormalization                 : {} '.format(cfg.nImageIntensityNormalization) )  
    print('cfg.nImageIntensityNormalizationUseRGBChannel    : {} '.format(cfg.nImageIntensityNormalizationUseRGBChannel) ) 
    
    print('cfg.sHeatMapMethod                               : {} '.format(cfg.sHeatMapMethod) )  
    print('cfg.nPositiveThreshold                           : {} '.format(cfg.nPositiveThreshold) )  
    print('cfg.nConvolutionalLayer                          : {} '.format(cfg.nConvolutionalLayer) )  
    print('cfg.sDirectoryForSavingHeatMapImage              : {} '.format(cfg.sDirectoryForSavingHeatMapImage) )  
    print('-'*30)

   
    fileWrite.write('-'*30)
    fileWrite.write('\n')
    fileWrite.write('===TestNetUsingImageInput=== \n') 
    fileWrite.write('cfg.sDirectoryForWeights                       : {} \n'.format(cfg.sDirectoryForWeights) )  
    fileWrite.write('cfg.sDirectoryForWeights2                      : {} \n'.format(cfg.sDirectoryForWeights2) )  
    fileWrite.write('cfg.sDirectoryForWeights3                      : {} \n'.format(cfg.sDirectoryForWeights3) ) 
    fileWrite.write('cfg.sVotingMethod                              : {} \n'.format(cfg.sVotingMethod) ) 
    
    fileWrite.write('cfg.sDirectoryForOutputRoot                    : {} \n'.format(cfg.sDirectoryForOutputRoot) )   
    fileWrite.write('cfg.sDirectoryForOutput                        : {} \n'.format(cfg.sDirectoryForOutput) )       
    
    fileWrite.write('cfg.sFilePathForTestImages                     : {} \n'.format(cfg.sFilePathForTestImages) )   
    fileWrite.write('cfg.sDirectoryForTestImages                    : {} \n'.format(cfg.sDirectoryForTestImages) )   

    
    fileWrite.write('cfg.sWeight                                    : {} \n'.format(cfg.sWeight) )      
    fileWrite.write('cfg.nImageRows                                 : {} \n'.format(cfg.nImageRows))
    fileWrite.write('cfg.nImageColumns                              : {} \n'.format(cfg.nImageColumns) )   
    fileWrite.write('cfg.nImageChannels                             : {} \n'.format(cfg.nImageChannels) )  
    fileWrite.write('cfg.nNbrOfClasses                              : {} \n'.format(cfg.nNbrOfClasses) )  
    fileWrite.write('cfg.sClassLabelNames                           : {} \n'.format(cfg.sClassLabelNames) )  
    fileWrite.write('cfg.nUseClaheForInputImage                     : {} \n'.format(cfg.nUseClaheForInputImage) )  
    fileWrite.write('cfg.nImageIntensityNormalization               : {} \n'.format(cfg.nImageIntensityNormalization) )  
    fileWrite.write('cfg.nImageIntensityNormalizationUseRGBChannel  : {} \n'.format(cfg.nImageIntensityNormalizationUseRGBChannel) )  
    
    fileWrite.write('cfg.sHeatMapMethod                             : {} \n'.format(cfg.sHeatMapMethod) )  
    fileWrite.write('cfg.nPositiveThreshold                         : {} \n'.format(cfg.nPositiveThreshold) )  
    fileWrite.write('cfg.nConvolutionalLayer                        : {} \n'.format(cfg.nConvolutionalLayer) )  
    fileWrite.write('cfg.sDirectoryForSavingHeatMapImage            : {} \n'.format(cfg.sDirectoryForSavingHeatMapImage) )  
    
    fileWrite.write('-'*30)
    fileWrite.write('\n')
    
       
            
    model  = load_model(cfg.sDirectoryForWeights)
    model2 = load_model(cfg.sDirectoryForWeights2)
    model3 = load_model(cfg.sDirectoryForWeights3)

    
    print('==============CNN Weights ======================')
    print('{}: {}'.format(sCNNModelList[0], cfg.sDirectoryForWeights))
    print('{}: {}'.format(sCNNModelList[1], cfg.sDirectoryForWeights2))
    print('{}: {}'.format(sCNNModelList[2], cfg.sDirectoryForWeights3))
    print('  ')
    
    fileWrite.write('==============CNN Weights ======================')
    fileWrite.write('{}: {} \n'.format(sCNNModelList[0], cfg.sDirectoryForWeights))
    fileWrite.write('{}: {} \n'.format(sCNNModelList[1], cfg.sDirectoryForWeights2))
    fileWrite.write('{}: {} \n'.format(sCNNModelList[2], cfg.sDirectoryForWeights3))
    fileWrite.write(' \n ')
    
    

    sListForTestDataFileNames=[]
    oFileOpen  = open(cfg.sFilePathForTestImages, 'r') 
    for aline in oFileOpen:

        sImageName = str(aline)            
        if sImageName.find('.tif') > 0 or sImageName.find('.jpeg') > 0 or sImageName.find('.png') > 0  or sImageName.find('.bmp') > 0: 
            sListForTestDataFileNames.append(aline)
            '''
            if match_the_error_image(sImageName) == False:
                #sListForTempTestData.append(aline)
                sListForTestDataFileNames.append(aline)
            '''
    oFileOpen.close()
    
    


    
    print('nNbrOfTestData                                   : {} '.format(len(sListForTestDataFileNames)) )  
    fileWrite.write('nNbrOfTestData                         : {} \n'.format(len(sListForTestDataFileNames)) )  





    #20200723 remove files not using or remplace existing labels to new labels

    print('NbrOfData before Class Label Reassignment:               {}'.format( len(sListForTestDataFileNames)))
    print('NbrOfClasses before Class Label Reassignment:            {}'.format(cfg.nNbrOfClasses))
    fileWrite.write('NbrOfData before Class Label Reassignment:            {} \n'.format( len(sListForTestDataFileNames)))
    fileWrite.write('NbrOfClasses before Class Label Reassignment:         {} \n'.format(cfg.nNbrOfClasses))
    
    sListForTestDataFileNames, nNbrOfClasses = ReassignClassLabelsInList( cfg, sListForTestDataFileNames )
    
    if(int(nNbrOfClasses) > 0 ):
        cfg.nNbrOfClasses = int(nNbrOfClasses)

    print('NbrOfData After Class Label Reassignment:               {}'.format(len(sListForTestDataFileNames)))
    print('NbrOfClasses After Class Label Reassignment:            {}'.format(cfg.nNbrOfClasses))
    fileWrite.write('NbrOfData After Class Label Reassignment:            {} \n'.format( len(sListForTestDataFileNames) ))
    fileWrite.write('NbrOfClasses After Class Label Reassignment:         {} \n'.format(cfg.nNbrOfClasses))


    print(' \n\n')
    fileWrite.write('\n \n')
    
    print ('=== Error Images === ')

    '''
    sGoodListPath = []
    sBadListPath = []
    sErrorListPath = []
    '''


    # 20240506 save results for ROC and AUC graphs for binary class
    sOutputForROC_AUC_Voting              = ''
    sOutputForROC_AUC_PredictionWeight0   = ''
    sOutputForROC_AUC_PredictionWeight1   = ''
    sOutputForROC_AUC_PredictionWeight2   = ''
    sOutputForROC_AUC_PredictionWeight    = []


    nStatistics = [[0] * cfg.nNbrOfClasses for i in range(cfg.nNbrOfClasses)] 
    
    nNbrOfData=0
    for aFileName in sListForTestDataFileNames[:]:

        aClassIndex, aFileNamePath  = aFileName.split(";")  
        sClassIndex = (str)(aClassIndex)
        sClassIndex = sClassIndex.lstrip()
        sClassIndex = sClassIndex.rstrip()       
        
        sImageName = (str)(aFileNamePath)
        sImageName = sImageName.strip('\n')
        sImageName = sImageName.lstrip()
        sImageName = sImageName.rstrip() 
                    
        if sImageName.find('.tif') > 0 or sImageName.find('.jpeg') > 0 or sImageName.find('.png') > 0  or sImageName.find('.bmp') > 0:
            
            
            if( cfg.nImageChannels == 3):
                DataX = np.ndarray((1, cfg.nImageRows, cfg.nImageColumns, cfg.nImageChannels), dtype=np.uint8)
            else:
                DataX = np.ndarray((1, cfg.nImageRows, cfg.nImageColumns), dtype=np.uint8)
                

            if( os.path.isfile( os.path.join(cfg.sDirectoryForTestImages, sImageName))  == False ):
                 sImageName =sImageName.replace('.tif','.jpg')
                 
                 if( os.path.isfile( os.path.join(cfg.sDirectoryForTestImages, sImageName))  == False ):
                     sImageName =sImageName.replace('.jpg','.jpeg')
            
                                      
            if( cfg.nImageChannels == 3):
                img = cv2.imread(os.path.join(cfg.sDirectoryForTestImages, sImageName), cv2.IMREAD_COLOR)  
            else:
                img = cv2.imread(os.path.join(cfg.sDirectoryForTestImages, sImageName), cv2.IMREAD_GRAYSCALE)   
              
            imgGrey = cv2.imread(os.path.join(cfg.sDirectoryForTestImages, sImageName), cv2.IMREAD_GRAYSCALE)   
                
            imgOriginal = img.copy()
            #20200524    
            if( int(cfg.nUseClaheForInputImage)  == 1 ):
                img = ClaheOperatorColor3D( img.copy(), cfg.nClaheTileSize)
                



            # 20230424 to change image channel format
            if( cfg.nImageChannels == 3):
                if( cfg.sImageChannelType == 'YCbCr' or cfg.sImageChannelType == 'ycbcr' ):

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)   #(if input image is RGB) 
                            
                elif( cfg.sImageChannelType == 'HSV' or cfg.sImageChannelType == 'hsv' ):
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  #(HSV (Hue, Saturation, Value) 
                            
                elif( cfg.sImageChannelType == 'HLS' or cfg.sImageChannelType == 'hls' ):
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)  #(HSL (Hue, Saturation, Lightness)
                            
                elif( cfg.sImageChannelType == 'LAB' or cfg.sImageChannelType == 'lab' ):                            

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)  #(HSL (Hue, Saturation, Lightness)                
            
                elif( cfg.sImageChannelType == '3Grey' or cfg.sImageChannelType == '3grey' ):     #20240319                       
                    img[:,:,0] = imgGrey
                    img[:,:,1] = imgGrey
                    img[:,:,2] = imgGrey



            
            try:
                
                if( img.shape[1] > cfg.nImageRows ):

                    img = cv2.resize(img, (cfg.nImageRows,cfg.nImageColumns), interpolation = cv2.INTER_AREA)  
                else:

                    img = cv2.resize(img, (cfg.nImageRows,cfg.nImageColumns), interpolation = cv2.INTER_CUBIC)
                
            except:

                print('Image Error:[{}] {}'.format(sImageName, sImageName.find ))
                continue
            
            #Original 
            DataX[0] = np.array([img])

            
    
            dPrediction = np.zeros((3, cfg.nNbrOfClasses), dtype=np.float) 

            if( int(cfg.nPreprocessInput) == 1 ):
                
                DataX0 = DataX.copy()
                
                cfg.sCNNType = sCNNModelList[0]                
                DataX1 = GetPreprocessInput(cfg, DataX0)  
                                
                cfg.sCNNType = sCNNModelList[1]                
                DataX2 = GetPreprocessInput(cfg, DataX0)  
                
                cfg.sCNNType = sCNNModelList[2]                
                DataX3 = GetPreprocessInput(cfg, DataX0)  
                
                cfg.sCNNType = sCNNModelList
            else:
                
                DataX1 = DataX.copy() 
                DataX2 = DataX.copy() 
                DataX3 = DataX.copy() 
               
            
            dPredictionResult = model.predict(DataX1, verbose=2)
            dPrediction[0,:]  = dPredictionResult[0]
            
            dPredictionResult = model2.predict(DataX2, verbose=2)
            dPrediction[1,:]  = dPredictionResult[0]

            dPredictionResult = model3.predict(DataX3, verbose=2)
            dPrediction[2,:]  = dPredictionResult[0]
            
            # 20230119 save heatmap images
            if( len(cfg.sHeatMapMethod) > 0 and False ):
                HeatMapImages = Generate_Heatmap_Image( img.copy(), model, cfg.nNbrOfClasses, cfg.sHeatMapMethod, cfg.nPositiveThreshold, cfg.nConvolutionalLayer )
                        
                for i in range( cfg.nNbrOfClasses ):
                                        
                    sImageName2 = sImageName
                                      
                    AHeatMapOverlapImage3 = HeatMapImages[i].copy().astype(np.uint8) 
                    AHeatMapOverlapImage3 = cv2.resize(AHeatMapOverlapImage3, (imgOriginal.shape[1],imgOriginal.shape[0]))


                    sImageName2 = sImageName2.replace('.tif', '_{}.tif'.format(cfg.sClassLabelNames[i]))
                    sImageName2 = sImageName2.replace('.jpg', '_{}.jpg'.format(cfg.sClassLabelNames[i]))
                    sImageName2 = sImageName2.replace('.jpeg','_{}.jpeg'.format(cfg.sClassLabelNames[i]))
                    sImageName2 = sImageName2.replace('.png', '_{}.png'.format(cfg.sClassLabelNames[i]))
                    sImageName2 = sImageName2.replace('.bmp', '_{}.bmp'.format(cfg.sClassLabelNames[i]))
                    
                    cv2.imwrite(os.path.join(cfg.sDirectoryForSavingHeatMapImage,sImageName2), AHeatMapOverlapImage3 )      
                
                                
            #20200728 add to handle two class problem
            dVotingSoft = np.zeros((cfg.nNbrOfClasses), dtype=np.float) 
            dVotingHard = np.zeros((cfg.nNbrOfClasses), dtype=np.float) 
            
            for m in range(3):
                if( cfg.nNbrOfClasses >= 2 ):
                    
                    dMax=0.0
                    nPredictionIndex = 0
                    
                    for i in range(len(dPrediction[m,:])):
                        if( dPrediction[m,i] > dMax):
                            dMax = dPrediction[m,i]
                            nPredictionIndex = i
                            
                    dVotingHard[nPredictionIndex] += 1
                    dVotingSoft += dPrediction[m,:]
                            
                else:
                    if( dPrediction[m,0] >= 0.5 ):
                        nPredictionIndex = 1
                        dVotingHard[nPredictionIndex] += 1                        
                    else:
                        nPredictionIndex = 0
                        dVotingHard[m,nPredictionIndex] += 1
                        
                    dVotingSoft[0] += (1.0 - dPrediction[m,0])                        
                    dVotingSoft[1] += dPrediction[0]
            
            dVotingHard /= 3.0
            dVotingSoft /= 3.0
            dVotingBoth = (dVotingHard + dVotingSoft)/2.0
            
           
            if( cfg.sVotingMethod == 'hard' ):
                dPredictionVoting = dVotingHard
            elif( cfg.sVotingMethod == 'soft' ):
                dPredictionVoting = dVotingSoft
            elif( cfg.sVotingMethod == 'hybrid' ):
                dPredictionVoting = dVotingBoth
            else:
                dPredictionVoting = dVotingSoft
                
            nPredictionIndex = 0
            dPredictionMax = 0.0
            for i in range(len(dPredictionVoting)):
                if( dPredictionVoting[i] > dPredictionMax ):
                    dPredictionMax = dPredictionVoting[i]
                    nPredictionIndex = i
            
            
 
            nClassIndex = int(sClassIndex)

            nStatistics[nClassIndex][nPredictionIndex] = int(nStatistics[nClassIndex][nPredictionIndex]) + 1
            

            sOutputError = 'Error [{:5}]: {}->{} ['.format(nNbrOfData, nClassIndex, nPredictionIndex)                       

            for m in range(cfg.nNbrOfClasses):
                sOutputError += '{:.5f},'.format( dPredictionVoting[m])                       

            sOutputError += '] : {}'.format( sImageName )                       


            if( nClassIndex != nPredictionIndex ):
                fileWrite.write( '{} \n'.format( sOutputError) )                       
                print( '{} '.format( sOutputError ))                       
                
            if( nNbrOfData%1000 == 0 and False):  
                fileWrite.write( '{} \n'.format( sOutputError) )                       
                print( '{} '.format( sOutputError ))                       


            # 20240506 save results for ROC and AUC graphs for binary class
            sOutputForROC_AUC_Voting            += '{}, {:.5f} \n'.format( nClassIndex, dPredictionVoting[int(cfg.nNbrOfClasses) - 1] )
            sOutputForROC_AUC_PredictionWeight0 += '{}, {:.5f} \n'.format( nClassIndex, dPrediction[0, int(cfg.nNbrOfClasses) - 1] )
            sOutputForROC_AUC_PredictionWeight1 += '{}, {:.5f} \n'.format( nClassIndex, dPrediction[1, int(cfg.nNbrOfClasses) - 1] )
            sOutputForROC_AUC_PredictionWeight2 += '{}, {:.5f} \n'.format( nClassIndex, dPrediction[2, int(cfg.nNbrOfClasses) - 1] )


            
            
            nNbrOfData+=1
           
    
    if( cfg.nNbrOfClasses == 2 ):
        Accuracy        =   (float)( (float)(nStatistics[0][0]+nStatistics[1][1])/(float)(nNbrOfData))
        
        if((nStatistics[0][0]+nStatistics[1][0]) > 0 ):
            Precision       =   (float)( (float)(nStatistics[0][0])/(float)(nStatistics[0][0]+nStatistics[1][0])) 
        else:
            Precision       = 0

        if((nStatistics[0][0]+nStatistics[0][1]) > 0 ):
            Recall          =   (float)( (float)(nStatistics[0][0])/(float)(nStatistics[0][0]+nStatistics[0][1]))    
        else:
            Recall       = 0

        if((nStatistics[1][0]+nStatistics[1][1]) > 0 ):
            TrueNegative    =   (float)( (float)(nStatistics[1][1])/(float)(nStatistics[1][0]+nStatistics[1][1]))    
        else:
            TrueNegative       = 0

        if((Precision+ Recall) > 0 ):
            FMeasure        =   (float)( (2.0*Precision*Recall)/(Precision+ Recall))    
        else:
            FMeasure       = 0

        if((nStatistics[0][0]+nStatistics[0][1]) > 0 ):
            Sensitivity     =   (float)( (float)(nStatistics[0][0])/(float)(nStatistics[0][0]+nStatistics[0][1]))    
        else:
            Sensitivity       = 0

        if((nStatistics[1][0]+nStatistics[1][1]) > 0 ):
            Specificity     =   (float)( (float)(nStatistics[1][1])/(float)(nStatistics[1][0]+nStatistics[1][1]))   
        else:
            Specificity       = 0
            
    else:
        Accuracy        =   0    
        Precision       =   0    
        Recall          =   0    
        TrueNegative    =   0    
        FMeasure        =   0    
        Sensitivity     =   0    
        Specificity     =   0  
        
        nCount=0
        for i in range(0,cfg.nNbrOfClasses,1):
            nCount += nStatistics[i][i]
            
        Accuracy        =   (float)(nCount)/(float)(nNbrOfData)
                
             
    print('-'*30)    
    print( '== Result Statistics ==' )
    print('-'*30)    
        
    fileWrite.write('-'*30)
    fileWrite.write('\n')
    fileWrite.write( '== Result Statistics== \n' )
    fileWrite.write('-'*30)
    fileWrite.write('\n')
    
    if( cfg.nNbrOfClasses == 2 ):
        for i in range(0,cfg.nNbrOfClasses,1):
            print( 'C[{0}] : {1:5}, {2:5} '.format(i,nStatistics[i][0], nStatistics[i][1]) )
            fileWrite.write( 'C[{0}] : {1:5}, {2:5} \n'.format(i,nStatistics[i][0], nStatistics[i][1]) )
    else:
        print( '{0:7}   {1:5}, {2:5}, {3:5}, {4:5}, {5:5}, {6:5}, {7}'.format('','[CNV]', '[DME]', '[DUR]', '[NOR]', 'Total', '#Acuracy', '%Acuracy') )
 
        for i in range(0,cfg.nNbrOfClasses,1):

            cCount = 0
            for j in range(0,cfg.nNbrOfClasses):
                cCount += nStatistics[i][j]
            
            cAcuracy = (float)(nStatistics[i][i])/(float)(cCount)
                
            #4 CLASSES
            print( 'C[{0:2}] : {1:5}, {2:5}, {3:5}, {4:5}, {5:6}, {6:6}, {7:9}'.format(cfg.sLabelNames[i],nStatistics[i][0], nStatistics[i][1], nStatistics[i][2], nStatistics[i][3], cCount, nStatistics[i][i], cAcuracy ))
            fileWrite.write( 'C[{0:2}] : {1:5}, {2:5}, {3:5}, {4:5}, {5:6}, {6:6}, {7:9} \n'.format(cfg.sLabelNames[i],nStatistics[i][0], nStatistics[i][1], nStatistics[i][2], nStatistics[i][3], cCount, nStatistics[i][i], cAcuracy ))
               
    print('-'*30)    
    fileWrite.write('-'*30)
    fileWrite.write('\n')

    print( 'Accuracy:       {0} '.format(Accuracy) )
    print( 'Sensitivity:    {0} '.format(Sensitivity) )
    print( 'Specificity:    {0} '.format(Specificity) )
    print( 'Precision:      {0} '.format(Precision) )
    print( 'Recall:         {0} '.format(Recall) )
    print( 'TrueNegative:   {0} '.format(TrueNegative) )
    print( 'FMeasure:       {0} '.format(FMeasure) )
    print('-'*30)    

    fileWrite.write( 'Accuracy:       {0} \n'.format(Accuracy) )
    fileWrite.write( 'Sensitivity:    {0} \n'.format(Sensitivity) )
    fileWrite.write( 'Specificity:    {0} \n'.format(Specificity) )
    fileWrite.write( 'Precision:      {0} \n'.format(Precision) )
    fileWrite.write( 'Recall:         {0} \n'.format(Recall) )
    fileWrite.write( 'TrueNegative:   {0} \n'.format(TrueNegative) )
    fileWrite.write( 'FMeasure:       {0} \n'.format(FMeasure) )
    fileWrite.write('-'*30)
    fileWrite.write('\n')
        
    sTotalAccuracy = '[ Acc:{}, Sen:{}, Spe:{}, Pre:{}, Rec:{}, Tn{}, Fme:{} ]'.format(Accuracy, Sensitivity, Specificity, Precision, Recall, TrueNegative, FMeasure  )
    
    print(sTotalAccuracy)
    print('-'*30)    
    
    fileWrite.write(sTotalAccuracy)    
    fileWrite.write('-'*30)
    fileWrite.write('\n')   
    fileWrite.write('-'*30)  
    fileWrite.write('\n')    
    fileWrite.write( '===Error List===\n' )
    fileWrite.write('-'*30)    
    fileWrite.write('\n')
    
    print( ' ')
    print( 'cfg.sDirectoryForOutput                        : {} '.format(cfg.sDirectoryForOutput))
        
    fileWrite.write(' \n' )       
    fileWrite.write('cfg.sDirectoryForOutput                        : {} \n'.format(cfg.sDirectoryForOutput) )       
              
    fileWrite.close() 


    if( int(cfg.nNbrOfClasses) == 2 ):
       
        sOutputForROC_AUC_PredictionWeight.append(sOutputForROC_AUC_PredictionWeight0)
        sOutputForROC_AUC_PredictionWeight.append(sOutputForROC_AUC_PredictionWeight1)
        sOutputForROC_AUC_PredictionWeight.append(sOutputForROC_AUC_PredictionWeight2)        
        
        nNbrOfCNNModels = len(sCNNModelList)
        
        for i in range( nNbrOfCNNModels + 1 ):
            
            if( i < nNbrOfCNNModels ):
                
                sFileNameForRocAuc = 'RocAuc_{}.txt'.format( sCNNModelList[i] )                    
                sResultFilePathForRocAuc = os.path.join(cfg.sDirectoryForOutput, sFileNameForRocAuc )       
                
                fileWrite = open(sResultFilePathForRocAuc,"w") 
                fileWrite.write( sOutputForROC_AUC_PredictionWeight[i] )       
                fileWrite.close()                 
                
            else:
                
                sFileNameForRocAuc = 'RocAuc_{}.txt'.format('Voting' )     
                sResultFilePathForRocAuc = os.path.join(cfg.sDirectoryForOutput, sFileNameForRocAuc )                       
                
                fileWrite = open(sResultFilePathForRocAuc,"w") 
                fileWrite.write( sOutputForROC_AUC_Voting )       
                fileWrite.close()                 
                
           


    
def main():
    
    cfg = config.Config()
    SetDirectories( cfg )
    
    if( str(cfg.sOperation).lower() ==  "traintest" ):
        TrainAndTestCNNUsingInputFile( cfg )
        TestCNNUsingImageInputFile( cfg )         
        
    elif( str(cfg.sOperation).lower() == "testensemble3" ):
        TestCNNUsingImageInputFileEnsemble3( cfg )   
 
   
    
if __name__ == "__main__":

    main()


