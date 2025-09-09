# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 15:04:21 2018

@author: jongkim
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
#import imutils
import numpy as np
import os
from PIL import Image
import random
#import math

'''
def translationx(min,max,rn):
    return np.array([[1,0,(max-min)*rn+min],[0,1,0]],dtype=np.float32)

def translationy(min,max,rn):
    return np.array([[1,0,0],[0,1,(max-min)*rn+min]],dtype=np.float32)
    
def scaling(min,max,rn,width,height):
    return np.array([[(max-min)*rn+min,0,-width*((max-min)*rn+min-1)/2],[0,(max-min)*rn+min,-height*((max-min)*rn+min-1)/2]],dtype=np.float32)
    
def flipx(chance,rn,width):
    if (rn<chance):
        return np.array([[-1,0,width],[0,1,0]],dtype=np.float32)
    else:
        return np.array([[1,0,0],[0,1,0]],dtype=np.float32)

def flipy(chance,rn,height):
    if (rn<chance):
        return np.array([[1,0,0],[0,-1,height]],dtype=np.float32)
    else:
        return np.array([[1,0,0],[0,1,0]],dtype=np.float32)
        
def rotation(min,max,rn,width,height):
    angle = (max-min)*rn+min
    sine = np.sin(angle)
    cosine = np.cos(angle)
    return np.array([[cosine,-sine,width*(1-cosine)/2+sine*height/2],[sine,cosine,height*(1-cosine)/2-sine*width/2]],dtype=np.float32)
'''        
'''
m_sFileForInputImagesRandomNumber   = 'X:\Jongwoo\RIGA-Train\MESSIDOR\TrainAndTestSets\TrainSets5\Set5-1-Train-AugRandomNumber.txt'
m_sFileForInputImagesAugment        = 'X:\Jongwoo\RIGA-Train\MESSIDOR\TrainAndTestSets\TrainSets5\Set5-1-TrainAug.txt'
m_sDirectoryForInputImagesAugment   = 'X:\\Jongwoo\\RIGA-Train\MESSIDOR\\Augmentation\\'
'''

'''
m_sFileForInputImages               = 'X:\Jongwoo\RIGA-Train\MESSIDOR\TrainAndTestSets\TrainSets5\Set5-1-Train.txt'
m_sDirectoryForInputImage           = 'X:\\Jongwoo\\RIGA-Train\\MESSIDOR\\'
m_sDirectoryForInputImagesROI       = m_sDirectoryForInputImage + "ROI\\"
m_sDirectoryInputImagesOpticDisc    = m_sDirectoryForInputImage + "OpticDisc\\"
m_sDirectoryForInputImagesCup       = m_sDirectoryForInputImage + "Cup\\"
m_sDirectoryForInputImagesOpticDiscAndCup = m_sDirectoryForInputImage + "OpticDiscAndCup255\\"
'''

m_sFileForInputImages               = 'R:\MyIVFA_FIle-GT-BloodVessels\TrainAndTestSetsLeakage\TrainSets5\Set5-1-TrainSample.txt'
m_sDirectoryForInputImage           = 'R:\MyIVFA_File\IVFA-Jpg\Ground truth Images\\'
m_sDirectoryInputImagesLabelImages  = m_sDirectoryForInputImage + 'ImagesBlackWhite\All\\'
#m_sDirectoryForInputImagesROI       = 'R:\MyIVFA_FIle-GT-BloodVessels\All-Combined-Clahe\\'

m_sFileForInputImages               = 'X:\Jongwoo\RIGA-Train\MESSIDOR\TrainAndTestSets\TrainSets5\Set5-1-Train.txt'
m_sDirectoryForInputImage           = 'X:\\Jongwoo\\RIGA-Train\\MESSIDOR\\'
m_sDirectoryForInputImagesAugment   = 'X:\\Jongwoo\\RIGA-Train\MESSIDOR\\Augmentation\\'

#For using ROI images 
m_sFileForInputImages               = 'X:\Jongwoo\RigaRefuge-Train-UNet\TrainTestSets\TrainSets10\Set10-1-TestSample.txt'
m_sDirectoryForInputImage           = 'X:\Jongwoo\RigaRefuge-Train-UNet\AllSets\RoiSize15\ROI\\'
m_sDirectoryInputImagesLabelImages  = 'X:\Jongwoo\RigaRefuge-Train-UNet\AllSets\RoiSize15\OpticDisc\\'
m_sDirectoryForInputImagesAugment   = 'X:\Jongwoo\RigaRefuge-Train-UNet\AllSets\RoiSize15\AugFromTrain\\'


m_nNbrOfAugmentation = 1 # must be even number if m_bFlipUsage    = True

m_bScaleUsage   = False
m_nIndexScale   = 0
m_nScaleStart   = 95    
m_nScaleEnd     = 105
m_nScaleStep    = 5
m_nScaleNumber  = 3  #(0.85,1.00,1.15)  #(1.0,1.15, 1.30) 

m_bFlipUsage    = True
m_nIndexFlip    = 1
m_nFlipStart    = 0   #(0:original, 1:flip using X axis, 2:flip based on Y axis)
m_nFlipEnd      = 3
m_nFlipStep     = 2 
m_nFlipNumber   = 2 #(original, flipY)


m_bRotationUsage    = True;
m_nIndexRotation    = 1
m_nRotationStart    = 0
m_nRotationEnd      = 360
m_nRotationStep     = 10
m_nExcludeRange     = 25 #15 # 25
m_nRotationNumber   = 5 #5(0,360,5,11,5):(0,5,10,350,355)  #3(0,360,10,15,3):(0, 10, 350) #5(0,360,10,25,5):(0, 10, 20, 340, 350), #(0, 15, 30, 330, 345)


m_bTranslationUsage    = False;
m_nIndexTranslation    = 1
m_nTranslationStart    = -10
m_nTranslationEnd      =  10
m_nTranslationStep     = 10
m_nTranslationNumber   = 1 #(original and one more random)  9 #( 5*5)





'''
m_bRotationUsage    = True;
m_nRotationStart    = 0
m_nRotationEnd      = 360
m_nRotationStep     = 15
m_nRotationNumber   = 3 #(0, 15, 345)
m_nExcludeRange     = 25
'''
    
if(m_bScaleUsage == False):
    m_nIndexScale   = 0
    m_nScaleStart = 100
    m_nScaleEnd   = 120
    m_nScaleStep  = 1000
    m_nScaleNumber  = 1
    
if(m_bFlipUsage == False):
    m_nIndexFlip    = 0
    m_nFlipStart = 0
    m_nFlipEnd   = 2
    m_nFlipStep  = 1000
    m_nFlipNumber   = 1
        
if(m_bRotationUsage == False):
    m_nIndexRotation    = 0
    m_nRotationStart = 0
    m_nRotationEnd   = 360
    m_nRotationStep  = 1000
    m_nRotationNumber   = 1    

if(m_bTranslationUsage == False):
    m_nIndexTranslation    = 0
    m_nTranslationStart    = 0
    m_nTranslationEnd      = 1
    m_nTranslationStep     = 10
    m_nTranslationNumber   = 1 #( 5*5)
    
    

    
    
'''
m_AugScale          = np.array([[100,120,1000,1],[85,120,15,3]])
m_AugFlip           = np.array([[0,2,1000,1],[0,3,2,2]])
m_AugRotation       = np.array([[0,360,1000,1],[0,360,10,5,25]])
m_AugTranslation    = np.array([[0,1,10,1],[-10,11,10,9]])
'''

m_bPrintOutResult = False   


'''
translatex=False #True
translatey=False #True
scale=True
fx=True #flip over y-axis
fy=False  #flip over x-axis
rotate=True
'''

def SetAugmentationFlipVariable ( nFlipStart, nFlipEnd, nFlipStep ):
    m_nFlipStart = nFlipStart
    m_nFlipEnd   = nFlipEnd
    m_nFlipStep  = nFlipStep

def GetAugmentationFlipVariable ():
    return m_nFlipStart, m_nFlipEnd, m_nFlipStep


def GetAugmentationInformation():
    
    sOutput = '\n '
    sOutput += '=== Augmentation Information ===\n'
    sOutput += 'Scale:          Usage={}, Start={}, End={}, Step={}, Number={}\n'.format(m_bScaleUsage,m_nScaleStart,m_nScaleEnd,m_nScaleStep,m_nScaleNumber)
    sOutput += 'Flip:           Usage={}, Start={}, End={}, Step={}, Number={}\n'.format(m_bFlipUsage,m_nFlipStart,m_nFlipEnd,m_nFlipStep,m_nFlipNumber)
    sOutput += 'Rotation:       Usage={}, Start={}, End={}, Step={}, ExcludeRange={}, Number={}\n'.format(m_bRotationUsage,m_nRotationStart,m_nRotationEnd,m_nRotationStep,m_nExcludeRange,m_nRotationNumber)
    sOutput += 'Translation:    Usage={}, Start={}, End={}, Step={}, Number={}\n'.format(m_bTranslationUsage,m_nTranslationStart,m_nTranslationEnd,m_nTranslationStep,m_nTranslationNumber)
    sOutput += 'Total AugmentationNumber: {}\n'.format(m_nScaleNumber*m_nFlipNumber*m_nRotationNumber*m_nTranslationNumber)
    sOutput += '===============================\n'
    sOutput += ' \n'
    
    return sOutput


def SetAugmentation( bScale, bFlip, bRotation, bTranslation ):

    if( bScale ):
        nIndexScale = 1
    else:
        nIndexScale = 0

    if( bFlip ):
        nIndexFlip = 1
    else:
        nIndexFlip = 0

    if( bRotation ):
        nIndexRotation = 1
    else:
        nIndexRotation = 0

    if( bTranslation ):
        nIndexTranslation = 1
    else:
        nIndexTranslation = 0      
        
    return nIndexScale, nIndexFlip, nIndexRotation, nIndexTranslation
        

##JW 20190404 This is the final correct version
##JW 20190404 add a function to remove white boundary in the original images if exists
'''
def SetAugmentation( bScale, bFlip, bRotation, bTranslation ):

    m_bScaleUsage       = bScale
    m_bFlipUsage        = bFlip
    m_bRotationUsage    = bRotation
    m_bTranslationUsage = bTranslation
    
    
    if(m_bScaleUsage == False):
        m_nScaleStart = 100
        m_nScaleEnd   = 120
        m_nScaleStep  = 1000
        m_nScaleNumber  = 1
    else:
        m_nScaleStart   = 85    
        m_nScaleEnd     = 120
        m_nScaleStep    = 15
        m_nScaleNumber  = 3  #(0.85,1.00,1.15)  #(1.0,1.15, 1.30)         
        
        
    if(m_bFlipUsage == False):
        m_nFlipStart = 0
        m_nFlipEnd   = 2
        m_nFlipStep  = 1000
        m_nFlipNumber   = 1
    else:
        m_nFlipStart    = 0   #(0:original, 1:flip using X axis, 2:flip based on Y axis)
        m_nFlipEnd      = 3
        m_nFlipStep     = 2 
        m_nFlipNumber   = 2 #(original, flipY)

            
    if(m_bRotationUsage == False):
        m_nRotationStart = 0
        m_nRotationEnd   = 360
        m_nRotationStep  = 1000
        m_nRotationNumber   = 1 
    else:
        m_nRotationStart    = 0
        m_nRotationEnd      = 360
        m_nRotationStep     = 10
        m_nRotationNumber   = 5 #(0, 10, 20, 340, 350), #(0, 15, 30, 330, 345)
        m_nExcludeRange     = 25        
    
    
    if(m_bTranslationUsage == False):
        m_nTranslationStart    = 0
        m_nTranslationEnd      = 1
        m_nTranslationStep     = 10
        m_nTranslationNumber   = 1 #( 5*5)  
    else:
        m_nTranslationStart    = -10
        m_nTranslationEnd      =  11
        m_nTranslationStep     = 10
        m_nTranslationNumber   = 9 #( 5*5)    
'''    


def AugmentImagesRegularVariations():
    

    sAugmentationInformation = GetAugmentationInformation()
    print( sAugmentationInformation )

      
    sListForTestDataFileNames=[]
    oFileOpen  = open(m_sFileForInputImages, 'r') 
    for aline in oFileOpen:
        aImagePath, aImageName = os.path.split(aline)
        sImageName = str(aImageName)            
        if sImageName.find('.tif') > 0 or sImageName.find('jpg') > 0 or sImageName.find('.png') > 0  or sImageName.find('.bmp') > 0:                
            sLine = (str)(aline)
            sLine = sLine.strip('\r\n')
            sLine = sLine.lstrip()
            sLine = sLine.rstrip()
            sListForTestDataFileNames.append(sLine)       
    oFileOpen.close()
  
   
 
    nCount=0
    for aFileName in sListForTestDataFileNames[:]:
        aImagePath, aImageName = os.path.split(aFileName)
        

        sImagePath = str(aImagePath)
        sImagePath = sImagePath.strip('\r\n')
        sImagePath = sImagePath.lstrip()
        sImagePath = sImagePath.rstrip()
        
        if( aImagePath == '' and len(m_sDirectoryForInputImage) > 0 ):
            aImagePath = m_sDirectoryForInputImage

        sImageName = str(aImageName)
        sImageName = sImageName.strip('\r\n')
        sImageName = sImageName.lstrip()
        sImageName = sImageName.rstrip()    
    
            
        if sImageName.find('.tif') > 0 or sImageName.find('jpg') > 0 or sImageName.find('.png') > 0  or sImageName.find('.bmp') > 0:    
    
            #aImage      = cv2.imread(os.path.join(aImagePath, aImageName), cv2.IMREAD_COLOR) 
            aImage      = cv2.imread(os.path.join(aImagePath, aImageName)) 

            aImageAdjust = AdjustBoundaryPixelValuesForAugmentation( aImage, 20 )
            
            sDirectoryForOutputImages ='./temp'  
            sOutfileName=sImageName                      
            sOutfileName=sOutfileName.replace('.tif','-Adjust.tif')
            sOutfileName=sOutfileName.replace('.jpg','-Adjust.jpg')
            cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), aImageAdjust )     
            
            #nImageChannels = len(aImage.shape)
            print('=={}=='.format(aFileName))
            ImageAu, nNbrOfImages = GetAugmentedImagesRegularOneInput( aImage, m_bPrintOutResult )


            if (True):
                print("Image [{}] ImageAu.shape=[{}], NumberOfAuImages =[{}]".format(aImageName,ImageAu.shape, nNbrOfImages))
                for i in range(nNbrOfImages):
                    print("Augmented Image [{}]".format(i))
                    if( aImage.shape[2] == 3):
                        plt.imshow(cv2.cvtColor(ImageAu[i], cv2.COLOR_BGR2RGB))
                    else:
                        plt.imshow(ImageAu[i])
                        
                    plt.show()                 
 
                    sDirectoryForOutputImages ='./temp'                        
                    sOutfileName=sImageName
                    sOutfileName=sOutfileName.replace('.tif','{}.tif'.format(i))
                    sOutfileName=sOutfileName.replace('.jpg','{}.jpg'.format(i))
                    cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), ImageAu[i] )     
                    
           
            nCount+=1
            if( nCount == 1 ):
                break



def AugmentTwoImagesRegularVariations():
    
    sAugmentationInformation = GetAugmentationInformation()
    print( sAugmentationInformation )
      
      
    sListForTestDataFileNames=[]
    oFileOpen  = open(m_sFileForInputImages, 'r') 
    for aline in oFileOpen:
        aImagePath, aImageName = os.path.split(aline)
        sImageName = str(aImageName)            
        if sImageName.find('.tif') > 0 or sImageName.find('jpg') > 0 or sImageName.find('.png') > 0  or sImageName.find('.bmp') > 0:                
            sLine = (str)(aline)
            sLine = sLine.strip('\r\n')
            sLine = sLine.lstrip()
            sLine = sLine.rstrip()
            sListForTestDataFileNames.append(sLine)       
    oFileOpen.close()
  
   
 
    nCount=0
    for aFileName in sListForTestDataFileNames[:]:
        aImagePath, aImageName = os.path.split(aFileName)

        sImagePath = str(aImagePath)
        sImagePath = sImagePath.strip('\r\n')
        sImagePath = sImagePath.lstrip()
        sImagePath = sImagePath.rstrip()

        if( aImagePath == '' and len(m_sDirectoryForInputImage) > 0 ):
            aImagePath = m_sDirectoryForInputImage


        sImageName = str(aImageName)
        sImageName = sImageName.strip('\r\n')
        sImageName = sImageName.lstrip()
        sImageName = sImageName.rstrip()    
    
            
        if sImageName.find('.tif') > 0 or sImageName.find('jpg') > 0 or sImageName.find('.png') > 0  or sImageName.find('.bmp') > 0:    
    
            aImage      = cv2.imread(os.path.join(aImagePath, aImageName), cv2.IMREAD_COLOR) 
            
            #aImageLabel = cv2.imread(os.path.join(m_sDirectoryForInputImagesOpticDiscAndCup, aImageName), cv2.IMREAD_GRAYSCALE ) 
            aImageLabel = cv2.imread(os.path.join(m_sDirectoryInputImagesLabelImages, aImageName), cv2.IMREAD_GRAYSCALE ) 
            
                       
            nImageChannels = len(aImage.shape)
  
            if(False):
                ImageAu, nNbrOfImages = GetAugmentedImagesRegularOneInput( aImage, m_bPrintOutResult )
                ImageLabelAu, nNbrOfImagesLabel = GetAugmentedImagesRegularOneInput( aImageLabel, m_bPrintOutResult )
            else:
                ImageAu, ImageLabelAu, nNbrOfImages = GetAugmentedImagesRegularTwoInputs( aImage, aImageLabel )
            

            print("Image [{}] ImageAu.shape=[{}], ImageAuLabel.shape=[{}], NumberOfAuImages =[{}]".format(aImageName, ImageAu.shape, ImageLabelAu.shape, nNbrOfImages))
            for i in range(nNbrOfImages):
                print("Augmented Image [{}]".format(i))
                if( nImageChannels == 3):
                    plt.imshow(cv2.cvtColor(ImageAu[i], cv2.COLOR_BGR2RGB))
                else:
                    plt.imshow(ImageAu[i])
                    
                plt.show()                 

                plt.imshow(ImageLabelAu[i])                    
                plt.show()                 
                   
            nCount+=1
            if( nCount == 2 ):
                break


def GetNumberOfAugmentedImagesRegular_Old():
        
    nNbrOfScales = 3    #(1.0,1.15, 1.30)
    nNbrOfFlips  = 2    #(original, flipX)
    #nNbrOfRotations = 12 #( 360/30 =12)
    nNbrOfRotations = 5 #(0, 15, 30, 330, 345)

    nNbrOfImages = nNbrOfScales*nNbrOfFlips*nNbrOfRotations
    
    return nNbrOfImages


def GetNumberOfAugmentedImagesRegular_20220810():
    
    #nNbrOfImages = m_nScaleNumber*m_nFlipNumber*m_nRotationNumber *m_nTranslationNumber      
    nNbrOfImages = m_nNbrOfAugmentation
             
    return nNbrOfImages

# 20220810
def GetNumberOfAugmentedImagesRegular( nAugOption = 125 ):
    
    m_nRotationNumber = 5
    if( nAugOption == 123):
        m_nRotationNumber = 3

        
    nNbrOfImages = m_nScaleNumber*m_nFlipNumber*m_nRotationNumber        
             
    return nNbrOfImages


def GetAugmentedImagesFromOneArray ( oImages, nNbrOfAugmentations ) :
    
    print('oImages.shape: ', oImages.shape )
    nImageLength = len(oImages.shape)   
           
    if( nImageLength > 3 ):
        DataX = np.ndarray((oImages.shape[0]*nNbrOfAugmentations, oImages.shape[1], oImages.shape[2], oImages.shape[3]), dtype=np.uint8)
    else:
        DataX = np.ndarray((oImages.shape[0]*nNbrOfAugmentations, oImages.shape[1], oImages.shape[2]), dtype=np.uint8)

    nNbrOfData=0
    for i in range(oImages.shape[0]):
        
        ImageAu, nNbrOfImages = GetAugmentedImagesRegularOneInput(oImages[i], m_bPrintOutResult)
        
        for j in range (ImageAu.shape[0]):
            DataX[nNbrOfData] = np.array([ImageAu[j]])  
            nNbrOfData +=1
            
        if( i%100 == 0):
            print('  augmented images {} of {}'.format(i, oImages.shape[0]))
    return DataX


def GetAugmentedImagesFromOneImage ( oImages, nNbrOfAugmentations ) :
    
    nImageChannels = len(oImages.shape)   
          
    if( nImageChannels > 2 ):
        DataX = np.ndarray((nNbrOfAugmentations, oImages.shape[0], oImages.shape[1], oImages.shape[2]), dtype=np.uint8)
    else:
        DataX = np.ndarray((nNbrOfAugmentations, oImages.shape[0], oImages.shape[1]), dtype=np.uint8)
    
    ImageAu, nNbrOfImages = GetAugmentedImagesRegularOneInput(oImages, m_bPrintOutResult)
     
    
    nNbrOfData=0
    for j in range (0,len(ImageAu)):
        DataX[nNbrOfData] = np.array([ImageAu[j]])  
             

    return DataX



def GetAugmentedImagesFromTwoArrays ( oImages, oImageLabels, nNbrOfAugmentations ) :
    
    print('oImages.shape: ', oImages.shape )
    nImageChannels = len(oImages.shape)   
           
    if( nImageChannels == 4):
        DataX = np.ndarray((oImages.shape[0]*nNbrOfAugmentations, oImages.shape[1], oImages.shape[2], oImages.shape[3]), dtype=np.uint8)
    else:
        DataX = np.ndarray((oImages.shape[0]*nNbrOfAugmentations, oImages.shape[1], oImages.shape[2]), dtype=np.uint8)


    #nImageLabelChannels = len(oImages.shape)    changed 20181026
    nImageLabelChannels = len(oImageLabels.shape)   

    if(nImageLabelChannels > 3 ):
        DataXLabel = np.ndarray((oImageLabels.shape[0]*nNbrOfAugmentations, oImageLabels.shape[1], oImageLabels.shape[2], oImageLabels.shape[3]), dtype=np.uint8)
    else:
        DataXLabel = np.ndarray((oImageLabels.shape[0]*nNbrOfAugmentations, oImageLabels.shape[1], oImageLabels.shape[2]), dtype=np.uint8)

    nNbrOfData=0
    for i in range(oImages.shape[0]):
        
        ImageAu, ImageLabelAu, nNbrOfImages = GetAugmentedImagesRegularTwoInputs(oImages[i],oImageLabels[i])
        
        for j in range (ImageAu.shape[0]):
            DataX[nNbrOfData] = np.array([ImageAu[j]])  
            DataXLabel[nNbrOfData] = np.array([ImageLabelAu[j]])  
            nNbrOfData +=1
            
    
    return DataX, DataXLabel



def GetAugmentedImagesFromTwoImages ( oImages, oImageLabels, nNbrOfAugmentations ) :
    
    nImageChannels = len(oImages.shape)   
           
    if( nImageChannels == 3):
        DataX = np.ndarray((nNbrOfAugmentations, oImages.shape[1], oImages.shape[2], oImages.shape[3]), dtype=np.uint8)
    else:
        DataX = np.ndarray((nNbrOfAugmentations, oImages.shape[1], oImages.shape[2]), dtype=np.uint8)


    #nImageLabelChannels = len(oImages.shape)    changed 20181026
    nImageLabelChannels = len(oImageLabels.shape)   

    if(nImageLabelChannels > 2 ):
        DataXLabel = np.ndarray((nNbrOfAugmentations, oImageLabels.shape[1], oImageLabels.shape[2], oImageLabels.shape[3]), dtype=np.uint8)
    else:
        DataXLabel = np.ndarray((nNbrOfAugmentations, oImageLabels.shape[1], oImageLabels.shape[2]), dtype=np.uint8)

        
    ImageAu, ImageLabelAu, nNbrOfImages = GetAugmentedImagesRegularTwoInputs(oImages,oImageLabels)
    
    nNbrOfData=0        
    for j in range (ImageAu.shape[0]):
        DataX[nNbrOfData] = np.array([ImageAu[j]])  
        DataXLabel[nNbrOfData] = np.array([ImageLabelAu[j]])  
        nNbrOfData +=1
            
    
    return DataX, DataXLabel





# receive a Image and the corresponding label image, create augmented images and label images, and return the images in numpy arrays
def GetAugmentedImagesRegularOneInput_20220810( aImage, bPrintOutResult ):

    
    '''
    nNbrOfScales = 3    #(1.0,1.15, 1.30)
    nNbrOfFlips  = 2    #(original, flipX)
    nNbrOfRotations = 12 #( 360/30 =12)
    nNbrOfImages = nNbrOfScales*nNbrOfFlips*nNbrOfRotations
    '''
    
    #nNbrOfImages = GetNumberOfAugmentedImagesRegular()
    nNbrOfImages = m_nNbrOfAugmentation

    nRows = aImage.shape[0]
    nCols = aImage.shape[1] 
    nImageChannels = len(aImage.shape)   

    
    if( nImageChannels == 3):
        DataX = np.ndarray((nNbrOfImages, nRows, nCols, nImageChannels), dtype=np.uint8)
    else:
        DataX = np.ndarray((nNbrOfImages, nRows, nCols), dtype=np.uint8)
        

    if( nImageChannels == 3 ):
        nImageBlue  = int( (int(aImage[1,1,0]) + int(aImage[nRows-2,1,0]) + int(aImage[1,nCols-2,0]) + int(aImage[nRows-2,nCols-2,0]))/4)
        nImageGreen = int( (int(aImage[1,1,1]) + int(aImage[nRows-2,1,1]) + int(aImage[1,nCols-2,1]) + int(aImage[nRows-2,nCols-2,1]))/4)
        nImageRed   = int( (int(aImage[1,1,2]) + int(aImage[nRows-2,1,2]) + int(aImage[1,nCols-2,2]) + int(aImage[nRows-2,nCols-2,2]))/4)
                
        ImageM = np.zeros( (nRows*3, nCols*3,3), np.uint8)
        ImageM[:] = (nImageBlue,nImageGreen, nImageRed)
    else: 
        nImageGreen = int( (int(aImage[1,1]) + int(aImage[nRows-2,1]) + int(aImage[1,nCols-2]) + int(aImage[nRows-2,nCols-2]))/4)           
        ImageM = np.zeros( (nRows*3, nCols*3), np.uint8)
        ImageM[:] = nImageGreen

    
    ImageMCopy = ImageM.copy()
    
    
    
    for m in range (nRows):
        for n in range (nCols):
            ImageM[int(nRows)+m, int(nCols)+n] = aImage[m,n]
                
                        
    nNbrOfData=0
    
    
    '''    
    m_bScaleUsage   = True
    m_nScaleStart   = 95    
    c     = 105
    
    m_bFlipUsage    = True;
    m_nFlipStart    = 0   #(0:original, 1:flip using X axis, 2:flip based on Y axis)
    m_nFlipEnd      = 3
    
    
    m_bRotationUsage    = True;
    m_nRotationStart    = 0
    m_nRotationEnd      = 360
    
    
    m_bTranslationUsage    = True;
    m_nTranslationStart    = -10
    m_nTranslationEnd      =  11
    '''
    
    
    for i in range(m_nFlipStart, m_nFlipEnd, m_nFlipStep): #flip x, Y, or X & Y 
                               
        if( i == 0):                
            aImageF         = ImageM
            if(bPrintOutResult):
                print("===original===")
        elif( i == 1 ):                    # do not use it because same as Horizontal flip  with 180 rotation
            aImageF         = cv2.flip(ImageM, 0) #vertical flip
            if( bPrintOutResult ):
                print("===Flip Vertical===")
        elif( i == 2 ):
            aImageF         = cv2.flip(ImageM, 1) #Horizonal flip
            if(bPrintOutResult):
                print("===Flip Horizontal===")
        elif( i == 3 ):                     # do not use it because same as original image with 180 rotation
            aImageF         = cv2.flip(ImageM, -1) #Both flip
            if(bPrintOutResult):
                print("===Flip Horizontal and Vertical===")    
       
    
        if( m_bFlipUsage == True):
            nNbrOfAugmentation = int(m_nNbrOfAugmentation/2)
        else:
            nNbrOfAugmentation = int(m_nNbrOfAugmentation)
            
    
    
        for y in range( nNbrOfAugmentation):
            
            if( y == 0 ):
                                
                row = aImage.shape[0]
                col = aImage.shape[1]
            
                nTransRow = row 
                nTransCol = col 
                                                                        
                aImageT      = aImageF[nTransRow:nTransRow+row, nTransCol:nTransCol+col]
                                
                DataX[nNbrOfData] = np.array([aImageT]) 
                nNbrOfData+=1
                                                               
                if(bPrintOutResult):
                    nScale=100
                    nFlip=i
                    nRotation=0
                    nTransRow=0
                    nTransCol=0
                    
                    print("=Scale=[{}] Flip=[{}] Rotate [{}] Translation[{},{}] Imageshape [{}]   ".format(nScale,nFlip,nRotation,nTransRow, nTransCol, aImageT.shape))
                                
                    if( nImageChannels == 3):
                        plt.imshow(cv2.cvtColor(aImageT, cv2.COLOR_BGR2RGB))
                    else:
                        plt.imshow(aImageT)
                                        
                    plt.show()    
            
    
                    sDirectoryForOutputImages ='../temp'
                                    
                    sOutfileName='Sc{}-Fl{}-Ro{}-Trr{}-Trc{}-Image.jpg'.format(nScale,nFlip,nRotation,nTransRow, nTransCol)
                    cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), aImageT )
                                    
                    
                continue
                
                
            
            if(m_bScaleUsage == True):
                nScale = random.randint(m_nScaleStart,m_nScaleEnd)
            else:
                nScale = 100
                
            aImageS = cv2.resize(aImageF, (int(aImageF.shape[1]*nScale*0.01),int(aImageF.shape[0]*nScale*0.01)))
            
            #20191225 move rescaled images in the middle
            if( nScale < 100 ):
                nReSize = int(aImageF.shape[0]*nScale*0.01)
                nRowEnd   = nReSize               
                nRowDif  = int(abs((int(aImageF.shape[0]) - int(aImageF.shape[0]*nScale*0.01))/2))
                
                nReSize = int(aImageF.shape[1]*nScale*0.01)
                nColEnd   = nReSize                
                nColDif  = int(abs((int(aImageF.shape[1]) - int(aImageF.shape[1]*nScale*0.01))/2))
                          
                aImageSM =ImageMCopy.copy()                                             
                aImageSM[nRowDif:nRowEnd+nRowDif, nColDif:nColEnd+nColDif] = aImageS[0:nRowEnd, 0:nColEnd]
                aImageS = aImageSM.copy()                
                                               
            elif( nScale > 100 ):
                nReSize = int(aImageF.shape[0]*nScale*0.01)
                nRowDif  = int(abs((int(aImageF.shape[0]) - int(aImageF.shape[0]*nScale*0.01))/2))
                
                nReSize = int(aImageF.shape[1]*nScale*0.01)
                nColDif  = int(abs((int(aImageF.shape[1]) - int(aImageF.shape[1]*nScale*0.01))/2))
                          
                aImageSM =ImageMCopy.copy()                                             
                aImageSM[0:int(aImageF.shape[0])-nRowDif, 0:int(aImageF.shape[1])-nColDif] = aImageS[nRowDif:int(aImageF.shape[0]), nColDif:int(aImageF.shape[1])]
                aImageS = aImageSM.copy()                
                

            if(m_bRotationUsage == True):
                
                nRotation = 180
                while ( nRotation > m_nExcludeRange and nRotation < 360-m_nExcludeRange ):
                    nRotation = random.randint(m_nRotationStart,m_nRotationEnd)                    
                    
                M = cv2.getRotationMatrix2D((int(aImageS.shape[1]/2), int(aImageS.shape[0]/2)),nRotation,1)
                aImageR      = cv2.warpAffine(aImageS,M,(aImageS.shape[1],aImageS.shape[0]))
                    
            else:
                nRotation = 0
                aImageR      = aImageS
                
                            
            #ListRotation.insert(0,0) 
            row = aImage.shape[0]
            col = aImage.shape[1]
            
            if (m_bTranslationUsage == True ):
                                            
                nTransRow = row - random.randint(m_nTranslationStart,m_nTranslationEnd)
                nTransCol = col - random.randint(m_nTranslationStart,m_nTranslationEnd)
                                                                        
                aImageT      = aImageR[nTransRow:nTransRow+row, nTransCol:nTransCol+col]
                
            else:
                nTransRow = row 
                nTransCol = col 
                                                                        
                aImageT      = aImageR[nTransRow:nTransRow+row, nTransCol:nTransCol+col]
                
                    
            DataX[nNbrOfData] = np.array([aImageT]) 
            nNbrOfData+=1
                                                           
            if(bPrintOutResult):
                print("=Scale=[{}] Flip=[{}] Rotate [{}] Translation[{},{}] Imageshape [{}]  ".format(nScale,i,nRotation,nTransRow, nTransCol, aImageT.shape))
                            
                if( nImageChannels == 3):
                    plt.imshow(cv2.cvtColor(aImageT, cv2.COLOR_BGR2RGB))
                else:
                    plt.imshow(aImageT)
                                    
                plt.show()    
        

                sDirectoryForOutputImages ='../temp'
                                
                sOutfileName='Sc{}-Fl{}-Ro{}-Trr{}-Trc{}-Image.jpg'.format(nScale,i,nRotation,nTransRow, nTransCol)
                cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), aImageT )
                                
                    
                            
                    
                    
    #return DataX, DataLabelX, nNbrOfImages
    return DataX, nNbrOfData





#20220810
# receive a Image, create augmented images, and return the images in numpy array
def GetAugmentedImagesRegularOneInput_DoNotUse( aImage, bPrintOutResult, nNbrOfAugmentations = 10, nAugOption = 125 ):
    
    '''
    nNbrOfScales = 3    #(1.0,1.15, 1.30)
    nNbrOfFlips  = 2    #(original, flipX)
    #nNbrOfRotations = 12 #( 360/30 =12)
    nNbrOfRotations = 5 #( 0,15,30, 330, 345)

    nNbrOfImages = nNbrOfScales*nNbrOfFlips*nNbrOfRotations
    '''
   
    #20211227 
    #nNbrOfImages = GetNumberOfAugmentedImagesRegular()
    
    
    if( nAugOption == 125 ):
        nNbrOfImages = GetNumberOfAugmentedImagesRegular() 
    elif( nAugOption == 123 ):
        nNbrOfImages = nNbrOfAugmentations 
    else:
        nNbrOfImages = GetNumberOfAugmentedImagesRegular() 
    
    
    nRows           = aImage.shape[0]
    nCols           = aImage.shape[1] 
    nImageLength    = len(aImage.shape)   
    
    if( nImageLength == 3):
        nChannels       = aImage.shape[2] 
    else:
        nChannels       = 1 
        
    
    if( nImageLength == 3):
        DataX = np.ndarray((nNbrOfImages, nRows, nCols, nChannels), dtype=np.uint8)
    else:
        DataX = np.ndarray((nNbrOfImages, nRows, nCols), dtype=np.uint8)
        
    nImageBlue=nImageGreen=nImageRed=0
    nTRange=5

    if( nChannels == 3 ):
        #nImageBlue  = int( (int(aImage[1,1,0]) + int(aImage[nRows-2,1,0]) + int(aImage[1,nCols-2,0]) + int(aImage[nRows-2,nCols-2,0]))/4)
        #nImageGreen = int( (int(aImage[1,1,1]) + int(aImage[nRows-2,1,1]) + int(aImage[1,nCols-2,1]) + int(aImage[nRows-2,nCols-2,1]))/4)
        #nImageRed   = int( (int(aImage[1,1,2]) + int(aImage[nRows-2,1,2]) + int(aImage[1,nCols-2,2]) + int(aImage[nRows-2,nCols-2,2]))/4)
        nTRange=10
        nImageBlue  = int( (int(aImage[nTRange,nTRange,0]) + int(aImage[nRows-nTRange,nTRange,0]) + int(aImage[nTRange,nCols-nTRange,0]) + int(aImage[nRows-nTRange,nCols-nTRange,0]))/4)
        nImageGreen = int( (int(aImage[nTRange,nTRange,1]) + int(aImage[nRows-nTRange,nTRange,1]) + int(aImage[nTRange,nCols-nTRange,1]) + int(aImage[nRows-nTRange,nCols-nTRange,1]))/4)
        nImageRed   = int( (int(aImage[nTRange,nTRange,2]) + int(aImage[nRows-nTRange,nTRange,2]) + int(aImage[nTRange,nCols-nTRange,2]) + int(aImage[nRows-nTRange,nCols-nTRange,2]))/4)
                        
        ImageM = np.zeros( (nRows*3, nCols*3, nChannels), np.uint8)
        ImageM[:] = (nImageBlue,nImageGreen, nImageRed)
    elif( nChannels == 4):            
        ImageM = np.zeros( (nRows*3, nCols*3, nChannels), np.uint8)
        ImageM[:] = (0, 0, 0, 0)
    elif( nChannels == 5):            
        ImageM = np.zeros( (nRows*3, nCols*3, nChannels), np.uint8)
        ImageM[:] = (0, 0, 0, 0, 0)
    elif( nChannels == 1):  
        nTRange=10
        nImageGreen = int( (int(aImage[nTRange,nTRange]) + int(aImage[nRows-nTRange,nTRange]) + int(aImage[nTRange,nCols-nTRange]) + int(aImage[nRows-nTRange,nCols-nTRange]))/4)          
        ImageM = np.zeros( (nRows*3, nCols*3), np.uint8)
        ImageM[:] = nImageGreen

    
    for m in range (nRows):
        for n in range (nCols):
            ImageM[int(nRows)+m, int(nCols)+n] = aImage[m,n]
                
            
    #To remove white boundary (intensity= 25) if exist
    for i in range(0, ImageM.shape[0]): #flip x, Y, or X & Y 
        if( abs(i-aImage.shape[0]) <= nTRange or abs(i-2*aImage.shape[0]) <= nTRange):
            for j in range( 0, ImageM.shape[1]):
                if( nChannels == 3):                
                    ImageM[i,j]= (nImageBlue,nImageGreen, nImageRed)
                else:
                    ImageM[i,j]= nImageGreen
                    
        
    for j in range(0, ImageM.shape[1]): #flip x, Y, or X & Y 
        if( abs(j-aImage.shape[1]) <= nTRange or abs(j-2*aImage.shape[1]) <= nTRange):
            for i in range( 0, ImageM.shape[0]):
                if( nChannels == 3):                    
                    ImageM[i,j]= (nImageBlue,nImageGreen, nImageRed)
                else:
                    ImageM[i,j]= nImageGreen


    if( bPrintOutResult):
        if( nChannels == 3):
            print('==Original images [{}]=='.format(aImage.shape))
            plt.imshow(cv2.cvtColor(aImage, cv2.COLOR_BGR2RGB))
            plt.show()                 
            print('==Original images in a big matrix [{}]=='.format(ImageM.shape))
            plt.imshow(cv2.cvtColor(ImageM, cv2.COLOR_BGR2RGB))
            plt.show()                 
        else:
            print('==Original images [{}]=='.format(aImage.shape))
            plt.imshow(aImage)
            plt.show()                 
            print('==Original images in a big matrix [{}]=='.format(ImageM.shape))
            plt.imshow(ImageM)                            
            plt.show()                 


                        
    nNbrOfData=0
        
    #for nScale in range (100,140,15):
    #for nScale in range (85,120,15):
    for nScale in range (m_nScaleStart, m_nScaleEnd, m_nScaleStep):
                        
        #imageS = cv2.resize(ImageM, (int(ImageM.shape[0]*nScale*0.01),int(ImageM.shape[1]*nScale*0.01)))
        imageS = cv2.resize(ImageM, (int(ImageM.shape[1]*nScale*0.01),int(ImageM.shape[0]*nScale*0.01)))
                                 
        if( bPrintOutResult):
            if( nChannels == 3):
                print('==imageS Scale[{}] Size[{}]=='.format(nScale,imageS.shape))
                plt.imshow(cv2.cvtColor(imageS, cv2.COLOR_BGR2RGB))
                plt.show()                 
            else:
                print('==imageS Scale[{}] Size[{}]=='.format(nScale,imageS.shape))
                plt.imshow(imageS)
                plt.show()                 
        
        #for i in range(0,2): #flip x, Y, or X & Y 
        for i in range(m_nFlipStart, m_nFlipEnd, m_nFlipStep): #flip x, Y, or X & Y 
                               
            if( i == 0):                
                aImageF = imageS
                if(bPrintOutResult):
                    print("===original===")
            elif( i == 1 ):
                aImageF = cv2.flip(imageS, 0) #Horizonal flip
                if(bPrintOutResult):
                    print("===Flip Horizontal===")
            elif( i == 2 ):                    # do not use it because same as Horizontal flip  with 180 rotation
                aImageF = cv2.flip(imageS, 1) #vertical flip
                if(bPrintOutResult):
                    print("===Flip Vertical===")
            elif( i == 3 ):                     # do not use it because same as original image with 180 rotation
                aImageF = cv2.flip(imageS, -1) #Both flip
                if(bPrintOutResult):
                    print("===Flip Horizontal and Vertical===")
                
            if( bPrintOutResult):
                if( nChannels == 3):
                    print('==aImageF Scale[{}] Size[{}]=='.format(nScale,aImageF.shape))
                    plt.imshow(cv2.cvtColor(aImageF, cv2.COLOR_BGR2RGB))
                    plt.show()                 
                else:
                    print('==aImageF Scale[{}] Size[{}]=='.format(nScale,aImageF.shape))
                    plt.imshow(aImageF)
                    plt.show()               


            #20211227 to make less rotation 
            if( nAugOption == 125):
                nExcludeRange = m_nExcludeRange
            elif( nAugOption == 123):
                nExcludeRange = m_nExcludeRange - 10
            else:
                nExcludeRange = m_nExcludeRange
                
                       
            #for j in range (0,360,15):               
            for j in range (m_nRotationStart, m_nRotationEnd, m_nRotationStep):               
                if( j < nExcludeRange or j > 360-nExcludeRange ):
                    #M = cv2.getRotationMatrix2D((int(aImageF.shape[0]/2), int(aImageF.shape[1]/2)),j,1)
                    #imageR = cv2.warpAffine(aImageF,M,(aImageF.shape[0],aImageF.shape[1])) 
                    #ImageF = imageR[int(imageR.shape[0]/2-nRows/2):int(imageR.shape[0]/2+nRows/2),int(imageR.shape[1]/2-nCols/2):int(imageR.shape[1]/2+nCols/2) ]
                    
                    M = cv2.getRotationMatrix2D((int(aImageF.shape[1]/2), int(aImageF.shape[0]/2)),j,1)

                    if( nChannels == 3 ):
                        imageR = cv2.warpAffine(aImageF,M,(aImageF.shape[1],aImageF.shape[0]), borderValue=(nImageBlue,nImageGreen, nImageRed) ) 
                    else:
                        imageR = cv2.warpAffine(aImageF,M,(aImageF.shape[1],aImageF.shape[0]), borderValue=nImageGreen ) 
                        
                        
                    if( bPrintOutResult):
                        if( nChannels == 3):
                            print('==imageR Scale[{}] Size[{}]=='.format(nScale,imageR.shape))
                            plt.imshow(cv2.cvtColor(imageR, cv2.COLOR_BGR2RGB))
                            plt.show()                 
                        else:
                            print('==imageR Scale[{}] Size[{}]=='.format(nScale,imageR.shape))
                            plt.imshow(imageR)
                            plt.show()               
                                                    
                    ImageF = imageR[int(imageR.shape[0]/2-nRows/2):int(imageR.shape[0]/2+nRows/2),int(imageR.shape[1]/2-nCols/2):int(imageR.shape[1]/2+nCols/2) ]
                          
                    if(bPrintOutResult):
                        print("=Scale=[{}] Flip=[{}] Rotate [{}]  Imageshape [{}] ".format(nScale,i,j, ImageF.shape))
                    
                        if( nChannels == 3):
                            plt.imshow(cv2.cvtColor(ImageF, cv2.COLOR_BGR2RGB))
                        else:
                            plt.imshow(ImageF)
                            
                        plt.show()    
                    
                    
                    DataX[nNbrOfData] = np.array([ImageF])  
                    
                    nNbrOfData+=1
                            
                    if(bPrintOutResult):
                        print("=Scale=[{}] Flip=[{}] Rotate [{}]  Imageshape [{}] ".format(nScale,i,j, ImageF.shape))
                    
                        if( nChannels == 3):
                            plt.imshow(cv2.cvtColor(ImageF, cv2.COLOR_BGR2RGB))
                        else:
                            plt.imshow(ImageF)
                            
                        plt.show()    

                        sDirectoryForOutputImages ='./temp'
                        
                        sOutfileName='Sc{}-Fl{}-Ro{}-Real.jpg'.format(nScale,i,j)
                        cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), imageR )                
                        sOutfileName='Sc{}-Fl{}-Ro{}-Final.jpg'.format(nScale,i,j)
                        cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), ImageF )                

                    
    return DataX, nNbrOfImages












def GetAugmentedImagesRandom( nAugmentationNumber, DataX, DataY, nNbrOfDataTrain, nNbrOfClasses, nClassNumbers, nPrintOutResult ):
    
    nlClass = np.zeros(nNbrOfClasses, dtype = np.int64)
    
    nNbrOfDataTrainNew = 1
    if( nAugmentationNumber > 0  ):
                        
        if( nAugmentationNumber == 1461 ):            
            nNbrOfDataTrainNew = nClassNumbers[0]*1 + nClassNumbers[1]*4 + nClassNumbers[2]*6 + nClassNumbers[3]*1
            #nNbrOfDataTrainNew = nClassNumbers[0]*4 + nClassNumbers[1]*4 + nClassNumbers[2]*6 + nClassNumbers[3]*1
        else:
            nNbrOfDataTrainNew = nNbrOfDataTrain*nAugmentationNumber
            
        TrainX = np.ndarray((nNbrOfDataTrainNew, DataX.shape[1], DataX.shape[2], DataX.shape[3]), dtype=np.float32)
        TrainY = np.ndarray((nNbrOfDataTrainNew, DataY.shape[0] ), dtype=np.float32)
    

        nCountAug=0
        for i in range( int(DataX.shape[0])):
    
            if( i < nPrintOutResult ):
                bPrintOutResult = True
            else:
                bPrintOutResult = False                        
                
            if( DataY[i,0] == 1.0 ):
                nAugNumber = 1
                #nAugNumber = 4
                nClassIndex = 0
            elif( DataY[i,1] == 1.0 ):
                nAugNumber = 4
                nClassIndex = 1
            elif( DataY[i,2] == 1.0 ):
                nAugNumber = 6
                nClassIndex = 2
            else:
                nAugNumber = 1
                nClassIndex = 3       
                
            if( nAugmentationNumber >= 1 and  nAugmentationNumber <= 100 ):
                AugImages, nNbrOfImages = GetAugmentedImagesRegularOneInputRandom( DataX[i], nAugmentationNumber, bPrintOutResult)                                
            elif( nAugmentationNumber == 1461 ):                                    
                AugImages, nNbrOfImages = GetAugmentedImagesRegularOneInputRandom( DataX[i], nAugNumber, bPrintOutResult)
                
                
            for m in range(nNbrOfImages):
                TrainX[nCountAug] = AugImages[m]
                TrainY[nCountAug] = DataY[i]
                nCountAug += 1
                nlClass[nClassIndex] += 1
            
    else:
        TrainX = DataX.copy() 
        TrainY = DataY.copy()
        nNbrOfDataTrainNew = nNbrOfDataTrain 
        nlClass = nClassNumbers.copy()        
        
    return TrainX, TrainY, nNbrOfDataTrainNew, nlClass
         
 
    




def GetAugmentedImagesRandomForGeneralSet( nAugmentationNumber, DataX, DataY, nNbrOfDataTrain, nNbrOfClasses, nClassNumbers, nPrintOutResult ):
    
    nlClass = np.zeros(nNbrOfClasses, dtype = np.int64)
    
    nNbrOfDataTrainNew = 1
    if( nAugmentationNumber > 0  ):
                        
        if( nAugmentationNumber == 1461 ):            
            nNbrOfDataTrainNew = nClassNumbers[0]*1 + nClassNumbers[1]*4 + nClassNumbers[2]*6 + nClassNumbers[3]*1
            #nNbrOfDataTrainNew = nClassNumbers[0]*4 + nClassNumbers[1]*4 + nClassNumbers[2]*6 + nClassNumbers[3]*1
        else:
            nNbrOfDataTrainNew = nNbrOfDataTrain*nAugmentationNumber
            
        TrainX = np.ndarray((nNbrOfDataTrainNew, DataX.shape[1], DataX.shape[2], DataX.shape[3]), dtype=np.float32)
        
        if( len(DataY.shape) > 1 ):
            TrainY = np.ndarray((nNbrOfDataTrainNew, DataY.shape[1] ), dtype=np.float32)
        else:
            #TrainY = np.ndarray((nNbrOfDataTrainNew ), dtype=np.float32)
            TrainY = np.ndarray((nNbrOfDataTrainNew, 1 ), dtype=np.float32)
    

        nCountAug=0
        for i in range( int(DataX.shape[0])):
    
            if( i < nPrintOutResult ):
                bPrintOutResult = True
            else:
                bPrintOutResult = False                        
                
            if( nAugmentationNumber == 1461 ):            
            
                if( DataY[i,0] == 1.0 ):
                    nAugNumber = 1
                    #nAugNumber = 4
                    nClassIndex = 0
                elif( DataY[i,1] == 1.0 ):
                    nAugNumber = 4
                    nClassIndex = 1
                elif( DataY[i,2] == 1.0 ):
                    nAugNumber = 6
                    nClassIndex = 2
                else:
                    nAugNumber = 1
                    nClassIndex = 3  
            else:
                nAugNumber = 1
                
                for j in range( int(DataY.shape[1]) ):
                    if( DataY[i,j] == 1.0 ):
                        nClassIndex = j
                
                
            if( nAugmentationNumber >= 1 and  nAugmentationNumber <= 100 ):
                AugImages, nNbrOfImages = GetAugmentedImagesRegularOneInputRandom( DataX[i], nAugmentationNumber, bPrintOutResult)                                
            elif( nAugmentationNumber == 1461 ):                                    
                AugImages, nNbrOfImages = GetAugmentedImagesRegularOneInputRandom( DataX[i], nAugNumber, bPrintOutResult)
                
                
            for m in range(nNbrOfImages):
                TrainX[nCountAug] = AugImages[m]
                TrainY[nCountAug] = DataY[i]
                nCountAug += 1
                nlClass[nClassIndex] += 1
            
    else:
        TrainX = DataX.copy() 
        TrainY = DataY.copy()
        nNbrOfDataTrainNew = nNbrOfDataTrain 
        nlClass = nClassNumbers.copy()        
        
    return TrainX, TrainY, nNbrOfDataTrainNew, nlClass
         






def GetAugmentedImagesRegular( nAugmentationNumber, DataX, DataY, nNbrOfDataTrain, nNbrOfClasses, nClassNumbers, nPrintOutResult ):
    
    nlClass = np.zeros(nNbrOfClasses, dtype = np.int64)
    
    nNbrOfDataTrainNew = 1
    if( nAugmentationNumber > 0  ):
                        
        if( nAugmentationNumber == 1461 ):            
            nNbrOfDataTrainNew = nClassNumbers[0]*1 + nClassNumbers[1]*4 + nClassNumbers[2]*6 + nClassNumbers[3]*1
            #nNbrOfDataTrainNew = nClassNumbers[0]*4 + nClassNumbers[1]*4 + nClassNumbers[2]*6 + nClassNumbers[3]*1
        else:
            if( nAugmentationNumber == 125):
                nNbrOfDataTrainNew = nNbrOfDataTrain*10
            elif( nAugmentationNumber == 123):
                nNbrOfDataTrainNew = nNbrOfDataTrain*6
            elif( nAugmentationNumber == 121):
                nNbrOfDataTrainNew = nNbrOfDataTrain*2
            else:
                nNbrOfDataTrainNew = nNbrOfDataTrain*nAugmentationNumber
            
        TrainX = np.ndarray((nNbrOfDataTrainNew, DataX.shape[1], DataX.shape[2], DataX.shape[3]), dtype=np.float32)
        
        if( len(DataY.shape) > 1 ):
            TrainY = np.ndarray((nNbrOfDataTrainNew, DataY.shape[1] ), dtype=np.float32)
        else:
            #TrainY = np.ndarray((nNbrOfDataTrainNew ), dtype=np.float32)
            TrainY = np.ndarray((nNbrOfDataTrainNew, 1 ), dtype=np.float32)
    

        nCountAug=0
        for i in range( int(DataX.shape[0])):
    
            if( i < nPrintOutResult ):
                bPrintOutResult = True
            else:
                bPrintOutResult = False                        
                
            if( nAugmentationNumber == 1461 ):            
            
                if( DataY[i,0] == 1.0 ):
                    nAugNumber = 1
                    #nAugNumber = 4
                    nClassIndex = 0
                elif( DataY[i,1] == 1.0 ):
                    nAugNumber = 4
                    nClassIndex = 1
                elif( DataY[i,2] == 1.0 ):
                    nAugNumber = 6
                    nClassIndex = 2
                else:
                    nAugNumber = 1
                    nClassIndex = 3  
            else:
                nAugNumber = 1
                
                for j in range( int(DataY.shape[1]) ):
                    if( DataY[i,j] == 1.0 ):
                        nClassIndex = j
                
                
            if( nAugmentationNumber >= 1 and  nAugmentationNumber <= 200 ):
                AugImages, nNbrOfImages = GetAugmentedImagesRegularOneInput( DataX[i], nAugmentationNumber, bPrintOutResult)                                
            elif( nAugmentationNumber == 1461 ):                                    
                AugImages, nNbrOfImages = GetAugmentedImagesRegularOneInput( DataX[i], nAugNumber, bPrintOutResult)
                
                
            for m in range(nNbrOfImages):
                TrainX[nCountAug] = AugImages[m]
                TrainY[nCountAug] = DataY[i]
                nCountAug += 1
                nlClass[nClassIndex] += 1
            
    else:
        TrainX = DataX.copy() 
        TrainY = DataY.copy()
        nNbrOfDataTrainNew = nNbrOfDataTrain 
        nlClass = nClassNumbers.copy()        
        
    return TrainX, TrainY, nNbrOfDataTrainNew, nlClass
         

# receive a Image and the corresponding label image, create augmented images and label images, and return the images in numpy arrays
def GetAugmentedImagesRegularOneInputRandom( aImage, nNbrOfAugmentation, bPrintOutResult ):

    
    
    
    
    '''
    nNbrOfScales = 3    #(1.0,1.15, 1.30)
    nNbrOfFlips  = 2    #(original, flipX)
    nNbrOfRotations = 12 #( 360/30 =12)
    nNbrOfImages = nNbrOfScales*nNbrOfFlips*nNbrOfRotations
    '''
    
    #nNbrOfImages = GetNumberOfAugmentedImagesRegular()
    nNbrOfImages = nNbrOfAugmentation

    nRows = aImage.shape[0]
    nCols = aImage.shape[1] 
    nImageChannels = len(aImage.shape)   

    
    if( nImageChannels == 3):
        DataX = np.ndarray((nNbrOfImages, nRows, nCols, nImageChannels), dtype=np.uint8)
    else:
        DataX = np.ndarray((nNbrOfImages, nRows, nCols), dtype=np.uint8)
        

    if( nImageChannels == 3 ):
        nImageBlue  = int( (int(aImage[1,1,0]) + int(aImage[nRows-2,1,0]) + int(aImage[1,nCols-2,0]) + int(aImage[nRows-2,nCols-2,0]))/4)
        nImageGreen = int( (int(aImage[1,1,1]) + int(aImage[nRows-2,1,1]) + int(aImage[1,nCols-2,1]) + int(aImage[nRows-2,nCols-2,1]))/4)
        nImageRed   = int( (int(aImage[1,1,2]) + int(aImage[nRows-2,1,2]) + int(aImage[1,nCols-2,2]) + int(aImage[nRows-2,nCols-2,2]))/4)
                
        ImageM = np.zeros( (nRows*3, nCols*3,3), np.uint8)
        ImageM[:] = (nImageBlue,nImageGreen, nImageRed)
    else: 
        nImageGreen = int( (int(aImage[1,1]) + int(aImage[nRows-2,1]) + int(aImage[1,nCols-2]) + int(aImage[nRows-2,nCols-2]))/4)           
        ImageM = np.zeros( (nRows*3, nCols*3), np.uint8)
        ImageM[:] = nImageGreen

    
    ImageMCopy = ImageM.copy()
    
    
    
    for m in range (nRows):
        for n in range (nCols):
            ImageM[int(nRows)+m, int(nCols)+n] = aImage[m,n]
                
                        
    nNbrOfData=0
    
    
    '''    
    m_bScaleUsage   = True
    m_nScaleStart   = 95    
    c     = 105
    
    m_bFlipUsage    = True;
    m_nFlipStart    = 0   #(0:original, 1:flip using X axis, 2:flip based on Y axis)
    m_nFlipEnd      = 3
    
    
    m_bRotationUsage    = True;
    m_nRotationStart    = 0
    m_nRotationEnd      = 360
    
    
    m_bTranslationUsage    = True;
    m_nTranslationStart    = -10
    m_nTranslationEnd      =  11
    '''
    
    # do not flip if nbrOfAugmentation == 1
    if( nNbrOfAugmentation == 1 ):
        nFlipStart = 0
        nFlipEnd   = 2
        nFlipStep  = 1000
    else:
        nFlipStart, nFlipEnd, nFlipStep = GetAugmentationFlipVariable()
    
    
    
    
    for i in range(nFlipStart, nFlipEnd, nFlipStep): #flip x, Y, or X & Y 
                               
        if( i == 0):                
            aImageF         = ImageM
            if(bPrintOutResult):
                print("===original===")
        elif( i == 1 ):                    # do not use it because same as Horizontal flip  with 180 rotation
            aImageF         = cv2.flip(ImageM, 0) #vertical flip
            if( bPrintOutResult ):
                print("===Flip Vertical===")
        elif( i == 2 ):
            aImageF         = cv2.flip(ImageM, 1) #Horizonal flip
            if(bPrintOutResult):
                print("===Flip Horizontal===")
        elif( i == 3 ):                     # do not use it because same as original image with 180 rotation
            aImageF         = cv2.flip(ImageM, -1) #Both flip
            if(bPrintOutResult):
                print("===Flip Horizontal and Vertical===")    
       
    
        if( m_bFlipUsage == True):
            nNbrOfAugment = int(nNbrOfAugmentation/2)
        else:
            nNbrOfAugment = int(nNbrOfAugmentation)
            
        if( nNbrOfAugment < 1 ):
            nNbrOfAugment = 1
    
        for y in range( nNbrOfAugment):
            
            if( y == 0 and nNbrOfAugment > 1 ):
                                
                row = aImage.shape[0]
                col = aImage.shape[1]
            
                nTransRow = row 
                nTransCol = col 
                                                                        
                aImageT      = aImageF[nTransRow:nTransRow+row, nTransCol:nTransCol+col]
                                
                DataX[nNbrOfData] = np.array([aImageT]) 
                nNbrOfData+=1
                                                               
                if(bPrintOutResult):
                    nScale=100
                    nFlip=i
                    nRotation=0
                    nTransRow=0
                    nTransCol=0
                    
                    print("=Scale=[{}] Flip=[{}] Rotate [{}] Translation[{},{}] Imageshape [{}]   ".format(nScale,nFlip,nRotation,nTransRow, nTransCol, aImageT.shape))
                                
                    if( nImageChannels == 3):
                        plt.imshow(cv2.cvtColor(aImageT, cv2.COLOR_BGR2RGB))
                    else:
                        plt.imshow(aImageT)
                                        
                    plt.show()    
            
    
                    sDirectoryForOutputImages ='../temp'
                                    
                    sOutfileName='Sc{}-Fl{}-Ro{}-Trr{}-Trc{}-Image.jpg'.format(nScale,nFlip,nRotation,nTransRow, nTransCol)
                    cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), aImageT )
                                    
                    
                continue
                
                
            
            if(m_bScaleUsage == True):
                nScale = random.randint(m_nScaleStart,m_nScaleEnd)
            else:
                nScale = 100
                
            aImageS = cv2.resize(aImageF, (int(aImageF.shape[1]*nScale*0.01),int(aImageF.shape[0]*nScale*0.01)))
            
            #20191225 move rescaled images in the middle
            if( nScale < 100 ):
                nReSize = int(aImageF.shape[0]*nScale*0.01)
                nRowEnd   = nReSize               
                nRowDif  = int(abs((int(aImageF.shape[0]) - int(aImageF.shape[0]*nScale*0.01))/2))
                
                nReSize = int(aImageF.shape[1]*nScale*0.01)
                nColEnd   = nReSize                
                nColDif  = int(abs((int(aImageF.shape[1]) - int(aImageF.shape[1]*nScale*0.01))/2))
                          
                aImageSM =ImageMCopy.copy()                                             
                aImageSM[nRowDif:nRowEnd+nRowDif, nColDif:nColEnd+nColDif] = aImageS[0:nRowEnd, 0:nColEnd]
                aImageS = aImageSM.copy()                
                                               
            elif( nScale > 100 ):
                nReSize = int(aImageF.shape[0]*nScale*0.01)
                nRowDif  = int(abs((int(aImageF.shape[0]) - int(aImageF.shape[0]*nScale*0.01))/2))
                
                nReSize = int(aImageF.shape[1]*nScale*0.01)
                nColDif  = int(abs((int(aImageF.shape[1]) - int(aImageF.shape[1]*nScale*0.01))/2))
                          
                aImageSM =ImageMCopy.copy()                                             
                aImageSM[0:int(aImageF.shape[0])-nRowDif, 0:int(aImageF.shape[1])-nColDif] = aImageS[nRowDif:int(aImageF.shape[0]), nColDif:int(aImageF.shape[1])]
                aImageS = aImageSM.copy()                
                

            if(m_bRotationUsage == True):
                
                nRotation = 180
                while ( nRotation > m_nExcludeRange and nRotation < 360-m_nExcludeRange ):
                    nRotation = random.randint(m_nRotationStart,m_nRotationEnd)                    
                    
                M = cv2.getRotationMatrix2D((int(aImageS.shape[1]/2), int(aImageS.shape[0]/2)),nRotation,1)
                aImageR      = cv2.warpAffine(aImageS,M,(aImageS.shape[1],aImageS.shape[0]))
                    
            else:
                nRotation = 0
                aImageR      = aImageS
                
                            
            #ListRotation.insert(0,0) 
            row = aImage.shape[0]
            col = aImage.shape[1]
            
            if (m_bTranslationUsage == True ):
                                            
                nTransRow = row - random.randint(m_nTranslationStart,m_nTranslationEnd)
                nTransCol = col - random.randint(m_nTranslationStart,m_nTranslationEnd)
                                                                        
                aImageT      = aImageR[nTransRow:nTransRow+row, nTransCol:nTransCol+col]
                
            else:
                nTransRow = row 
                nTransCol = col 
                                                                        
                aImageT      = aImageR[nTransRow:nTransRow+row, nTransCol:nTransCol+col]
                
                    
            DataX[nNbrOfData] = np.array([aImageT]) 
            nNbrOfData+=1
                                                           
            if(bPrintOutResult):
                print("=Scale=[{}] Flip=[{}] Rotate [{}] Translation[{},{}] Imageshape [{}]  ".format(nScale,i,nRotation,nTransRow, nTransCol, aImageT.shape))
                            
                if( nImageChannels == 3):
                    plt.imshow(cv2.cvtColor(aImageT, cv2.COLOR_BGR2RGB))
                else:
                    plt.imshow(aImageT)
                                    
                plt.show()    
        

                sDirectoryForOutputImages ='../temp'
                                
                sOutfileName='Sc{}-Fl{}-Ro{}-Trr{}-Trc{}-Image.jpg'.format(nScale,i,nRotation,nTransRow, nTransCol)
                cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), aImageT )
                                
                    
                            
                    
                    
    #return DataX, DataLabelX, nNbrOfImages
    return DataX, nNbrOfData












# receive a Image, create augmented images, and return the images in numpy array
#Add translation function
def GetAugmentedImagesRegularOneInput_20191216( aImage, bPrintOutResult ):
    
    '''
    nNbrOfScales = 3    #(1.0,1.15, 1.30)
    nNbrOfFlips  = 2    #(original, flipX)
    #nNbrOfRotations = 12 #( 360/30 =12)
    nNbrOfRotations = 5 #( 0,15,30, 330, 345)
    nNbrOfTranslation = 9 # (-10, 10)*(-10, 10)

    nNbrOfImages = nNbrOfScales*nNbrOfFlips*nNbrOfRotations
    '''
    
    nNbrOfImages = GetNumberOfAugmentedImagesRegular()
    
    nRows           = aImage.shape[0]
    nCols           = aImage.shape[1] 
    nImageLength    = len(aImage.shape)   
    
    if( nImageLength == 3):
        nChannels       = aImage.shape[2] 
    else:
        nChannels       = 1 
        
    
    if( nImageLength == 3):
        DataX = np.ndarray((nNbrOfImages, nRows, nCols, nChannels), dtype=np.uint8)
    else:
        DataX = np.ndarray((nNbrOfImages, nRows, nCols), dtype=np.uint8)
        
    nImageBlue=nImageGreen=nImageRed=0
    nTRange=5

    if( nChannels == 3 ):
        #nImageBlue  = int( (int(aImage[1,1,0]) + int(aImage[nRows-2,1,0]) + int(aImage[1,nCols-2,0]) + int(aImage[nRows-2,nCols-2,0]))/4)
        #nImageGreen = int( (int(aImage[1,1,1]) + int(aImage[nRows-2,1,1]) + int(aImage[1,nCols-2,1]) + int(aImage[nRows-2,nCols-2,1]))/4)
        #nImageRed   = int( (int(aImage[1,1,2]) + int(aImage[nRows-2,1,2]) + int(aImage[1,nCols-2,2]) + int(aImage[nRows-2,nCols-2,2]))/4)
        nTRange=10
        nImageBlue  = int( (int(aImage[nTRange,nTRange,0]) + int(aImage[nRows-nTRange,nTRange,0]) + int(aImage[nTRange,nCols-nTRange,0]) + int(aImage[nRows-nTRange,nCols-nTRange,0]))/4)
        nImageGreen = int( (int(aImage[nTRange,nTRange,1]) + int(aImage[nRows-nTRange,nTRange,1]) + int(aImage[nTRange,nCols-nTRange,1]) + int(aImage[nRows-nTRange,nCols-nTRange,1]))/4)
        nImageRed   = int( (int(aImage[nTRange,nTRange,2]) + int(aImage[nRows-nTRange,nTRange,2]) + int(aImage[nTRange,nCols-nTRange,2]) + int(aImage[nRows-nTRange,nCols-nTRange,2]))/4)
                        
        ImageM = np.zeros( (nRows*3, nCols*3, nChannels), np.uint8)
        ImageM[:] = (nImageBlue,nImageGreen, nImageRed)
    elif( nChannels == 4):            
        ImageM = np.zeros( (nRows*3, nCols*3, nChannels), np.uint8)
        ImageM[:] = (0, 0, 0, 0)
    elif( nChannels == 5):            
        ImageM = np.zeros( (nRows*3, nCols*3, nChannels), np.uint8)
        ImageM[:] = (0, 0, 0, 0, 0)
    elif( nChannels == 1):  
        nTRange=10
        nImageGreen = int( (int(aImage[nTRange,nTRange]) + int(aImage[nRows-nTRange,nTRange]) + int(aImage[nTRange,nCols-nTRange]) + int(aImage[nRows-nTRange,nCols-nTRange]))/4)          
        ImageM = np.zeros( (nRows*3, nCols*3), np.uint8)
        ImageM[:] = nImageGreen

    
    for m in range (nRows):
        for n in range (nCols):
            ImageM[int(nRows)+m, int(nCols)+n] = aImage[m,n]
                
            
    #To remove white boundary (intensity= 25) if exist
    for i in range(0, ImageM.shape[0]): #flip x, Y, or X & Y 
        if( abs(i-aImage.shape[0]) <= nTRange or abs(i-2*aImage.shape[0]) <= nTRange):
            for j in range( 0, ImageM.shape[1]):
                if( nChannels == 3):                
                    ImageM[i,j]= (nImageBlue,nImageGreen, nImageRed)
                else:
                    ImageM[i,j]= nImageGreen
                    
        
    for j in range(0, ImageM.shape[1]): #flip x, Y, or X & Y 
        if( abs(j-aImage.shape[1]) <= nTRange or abs(j-2*aImage.shape[1]) <= nTRange):
            for i in range( 0, ImageM.shape[0]):
                if( nChannels == 3):                    
                    ImageM[i,j]= (nImageBlue,nImageGreen, nImageRed)
                else:
                    ImageM[i,j]= nImageGreen


    if( bPrintOutResult):
        if( nChannels == 3):
            print('==Original images [{}]=='.format(aImage.shape))
            plt.imshow(cv2.cvtColor(aImage, cv2.COLOR_BGR2RGB))
            plt.show()                 
            print('==Original images in a big matrix [{}]=='.format(ImageM.shape))
            plt.imshow(cv2.cvtColor(ImageM, cv2.COLOR_BGR2RGB))
            plt.show()                 
        else:
            print('==Original images [{}]=='.format(aImage.shape))
            plt.imshow(aImage)
            plt.show()                 
            print('==Original images in a big matrix [{}]=='.format(ImageM.shape))
            plt.imshow(ImageM)                            
            plt.show()                 


                        
    nNbrOfData=0
        
    #for nScale in range (100,140,15):
    #for nScale in range (85,120,15):
    for nScale in range (m_nScaleStart, m_nScaleEnd, m_nScaleStep):
                        
        #imageS = cv2.resize(ImageM, (int(ImageM.shape[0]*nScale*0.01),int(ImageM.shape[1]*nScale*0.01)))
        imageS = cv2.resize(ImageM, (int(ImageM.shape[1]*nScale*0.01),int(ImageM.shape[0]*nScale*0.01)))
                                 
        if( bPrintOutResult):
            if( nChannels == 3):
                print('==imageS Scale[{}] Size[{}]=='.format(nScale,imageS.shape))
                plt.imshow(cv2.cvtColor(imageS, cv2.COLOR_BGR2RGB))
                plt.show()                 
            else:
                print('==imageS Scale[{}] Size[{}]=='.format(nScale,imageS.shape))
                plt.imshow(imageS)
                plt.show()                 
        
        #for i in range(0,2): #flip x, Y, or X & Y 
        for i in range(m_nFlipStart, m_nFlipEnd, m_nFlipStep): #flip x, Y, or X & Y 
                               
            if( i == 0):                
                aImageF = imageS
                if(bPrintOutResult):
                    print("===original===")
            elif( i == 1 ):
                aImageF = cv2.flip(imageS, 0) #Horizonal flip
                if(bPrintOutResult):
                    print("===Flip Horizontal===")
            elif( i == 2 ):                    # do not use it because same as Horizontal flip  with 180 rotation
                aImageF = cv2.flip(imageS, 1) #vertical flip
                if(bPrintOutResult):
                    print("===Flip Vertical===")
            elif( i == 3 ):                     # do not use it because same as original image with 180 rotation
                aImageF = cv2.flip(imageS, -1) #Both flip
                if(bPrintOutResult):
                    print("===Flip Horizontal and Vertical===")
                
            if( False  ): #bPrintOutResult ):
                if( nChannels == 3):
                    print('==aImageF Scale[{}] Size[{}]=='.format(nScale,aImageF.shape))
                    plt.imshow(cv2.cvtColor(aImageF, cv2.COLOR_BGR2RGB))
                    plt.show()                 
                else:
                    print('==aImageF Scale[{}] Size[{}]=='.format(nScale,aImageF.shape))
                    plt.imshow(aImageF)
                    plt.show()               

                              
                    
                    
            ListRotation = []  
            for j in range (m_nRotationStart, m_nRotationEnd, m_nRotationStep):
                if( j == 0 ):
                    continue
                elif( j < m_nExcludeRange or j > 360-m_nExcludeRange ):
                    ListRotation.append(j)
            
            if(bPrintOutResult):
                print("=== Before shuffling [ListRotation]===" )
                print(ListRotation)
            
            
            
            random.shuffle(ListRotation) 
            
            ListRotation.insert(0,0) 
            
            if(bPrintOutResult):
                print("=== After shuffling [ListRotation]===" )
                print(ListRotation)
                    
                    
                    
            #for j in range (0,360,15):               
            #for j in range (m_nRotationStart, m_nRotationEnd, m_nRotationStep):               
            #    if( j < m_nExcludeRange or j > 360-m_nExcludeRange ):
                    #M = cv2.getRotationMatrix2D((int(aImageF.shape[0]/2), int(aImageF.shape[1]/2)),j,1)
                    #imageR = cv2.warpAffine(aImageF,M,(aImageF.shape[0],aImageF.shape[1])) 
                    #ImageF = imageR[int(imageR.shape[0]/2-nRows/2):int(imageR.shape[0]/2+nRows/2),int(imageR.shape[1]/2-nCols/2):int(imageR.shape[1]/2+nCols/2) ]

            for j in range(int(m_nRotationNumber)):
                if( True ):

                    nRotationValue = ListRotation[j]
                    
                    M = cv2.getRotationMatrix2D((int(aImageF.shape[1]/2), int(aImageF.shape[0]/2)),nRotationValue,1)

                    if( nChannels == 3 ):
                        imageR = cv2.warpAffine(aImageF,M,(aImageF.shape[1],aImageF.shape[0]), borderValue=(nImageBlue,nImageGreen, nImageRed) ) 
                    else:
                        imageR = cv2.warpAffine(aImageF,M,(aImageF.shape[1],aImageF.shape[0]), borderValue=nImageGreen ) 
                        
                        
                    #if( False ): #bPrintOutResult):
                    #    if( nChannels == 3):
                    #        print('==imageR Scale[{}] Size[{}]=='.format(nScale,imageR.shape))
                    #        plt.imshow(cv2.cvtColor(imageR, cv2.COLOR_BGR2RGB))
                    #        plt.show()                 
                    #    else:
                    #        print('==imageR Scale[{}] Size[{}]=='.format(nScale,imageR.shape))
                    #        plt.imshow(imageR)
                    #        plt.show()               
                       
                        
                    
                    ListTranslation = []  
                    for m in range (m_nTranslationStart, m_nTranslationEnd, m_nTranslationStep):
                        for n in range (m_nTranslationStart, m_nTranslationEnd, m_nTranslationStep):
                            if( m == 0 and n == 0):
                                continue
                            else:
                                ListTranslation.append( [m,n])
                         
                    if(bPrintOutResult):
                        print("=== Before shuffling [ListTranslation]===" )
                        print(ListTranslation)
                            
                            
                    random.shuffle(ListTranslation) 
                    
                    ListTranslation.insert(0,[0,0])
                    
                    if(bPrintOutResult):
                        print("=== After shuffling [ListTranslation]===" )
                        print(ListTranslation)
                            
                                                 

                    #for m in range (m_nTranslationStart, m_nTranslationEnd, m_nTranslationStep):
                    #    for n in range (m_nTranslationStart, m_nTranslationEnd, m_nTranslationStep):
                    
                    for k in range( int(m_nTranslationNumber) ):
                        if( True ):
                                    
                            row = aImage.shape[0]
                            col = aImage.shape[1]
                            
                            #nTransRow = row - m
                            #nTransCol = col - n

                            nTransRow = row - ListTranslation[k][0]
                            nTransCol = col - ListTranslation[k][1]
                                                
                            #ImageF = imageR[int(imageR.shape[0]/2-nRows/2):int(imageR.shape[0]/2+nRows/2),int(imageR.shape[1]/2-nCols/2):int(imageR.shape[1]/2+nCols/2) ]
                        
                            ImageT = imageR[nTransRow:nTransRow+row, nTransCol:nTransCol+col]
                        
                            DataX[nNbrOfData] = np.array([ImageT]) 
                            nNbrOfData+=1
                                
                            if(bPrintOutResult):
                                print("=Scale=[{}] Flip=[{}] Rotate [{}] Translation[{},{}] Imageshape [{}] ".format(nScale,i,j,m,n, ImageT.shape))
                            
                                if( nChannels == 3):
                                    plt.imshow(cv2.cvtColor(ImageT, cv2.COLOR_BGR2RGB))
                                else:
                                    plt.imshow(ImageT)
                                    
                                plt.show()    
        
                                sDirectoryForOutputImages ='../temp'
                                #sDirectoryForOutputImages =m_sDirectoryForInputImagesAugment
                                
                                #sOutfileName='Sc{}-Fl{}-Ro{}-Trr{}-Trc{}-Real.jpg'.format(nScale,i,j,m,n)
                                #cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), imageR )                
                                sOutfileName='Sc{}-Fl{}-Ro{}-Trr{}-Trc{}-Final.jpg'.format(nScale,i,j,m,n)
                                cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), ImageT )                

                    
    return DataX, nNbrOfImages




# receive a Image, create augmented images, and return the images in numpy array
#Add translation function
def GetAugmentedImagesRegularOneInputS1F2R3T2( aImage, bPrintOutResult ):
    
    '''
    nNbrOfScales = 3    #(1.0,1.15, 1.30)
    nNbrOfFlips  = 2    #(original, flipX)
    #nNbrOfRotations = 12 #( 360/30 =12)
    nNbrOfRotations = 5 #( 0,15,30, 330, 345)
    nNbrOfTranslation = 9 # (-10, 10)*(-10, 10)

    nNbrOfImages = nNbrOfScales*nNbrOfFlips*nNbrOfRotations
    '''
    
    nNbrOfImages = GetNumberOfAugmentedImagesRegular()
    
    nRows           = aImage.shape[0]
    nCols           = aImage.shape[1] 
    nImageLength    = len(aImage.shape)   
    
    if( nImageLength == 3):
        nChannels       = aImage.shape[2] 
    else:
        nChannels       = 1 
        
    
    if( nImageLength == 3):
        DataX = np.ndarray((nNbrOfImages, nRows, nCols, nChannels), dtype=np.uint8)
    else:
        DataX = np.ndarray((nNbrOfImages, nRows, nCols), dtype=np.uint8)
        
    nImageBlue=nImageGreen=nImageRed=0
    nTRange=5

    if( nChannels == 3 ):
        #nImageBlue  = int( (int(aImage[1,1,0]) + int(aImage[nRows-2,1,0]) + int(aImage[1,nCols-2,0]) + int(aImage[nRows-2,nCols-2,0]))/4)
        #nImageGreen = int( (int(aImage[1,1,1]) + int(aImage[nRows-2,1,1]) + int(aImage[1,nCols-2,1]) + int(aImage[nRows-2,nCols-2,1]))/4)
        #nImageRed   = int( (int(aImage[1,1,2]) + int(aImage[nRows-2,1,2]) + int(aImage[1,nCols-2,2]) + int(aImage[nRows-2,nCols-2,2]))/4)
        nTRange=10
        nImageBlue  = int( (int(aImage[nTRange,nTRange,0]) + int(aImage[nRows-nTRange,nTRange,0]) + int(aImage[nTRange,nCols-nTRange,0]) + int(aImage[nRows-nTRange,nCols-nTRange,0]))/4)
        nImageGreen = int( (int(aImage[nTRange,nTRange,1]) + int(aImage[nRows-nTRange,nTRange,1]) + int(aImage[nTRange,nCols-nTRange,1]) + int(aImage[nRows-nTRange,nCols-nTRange,1]))/4)
        nImageRed   = int( (int(aImage[nTRange,nTRange,2]) + int(aImage[nRows-nTRange,nTRange,2]) + int(aImage[nTRange,nCols-nTRange,2]) + int(aImage[nRows-nTRange,nCols-nTRange,2]))/4)
                        
        ImageM = np.zeros( (nRows*3, nCols*3, nChannels), np.uint8)
        ImageM[:] = (nImageBlue,nImageGreen, nImageRed)
    elif( nChannels == 4):            
        ImageM = np.zeros( (nRows*3, nCols*3, nChannels), np.uint8)
        ImageM[:] = (0, 0, 0, 0)
    elif( nChannels == 5):            
        ImageM = np.zeros( (nRows*3, nCols*3, nChannels), np.uint8)
        ImageM[:] = (0, 0, 0, 0, 0)
    elif( nChannels == 1):  
        nTRange=10
        nImageGreen = int( (int(aImage[nTRange,nTRange]) + int(aImage[nRows-nTRange,nTRange]) + int(aImage[nTRange,nCols-nTRange]) + int(aImage[nRows-nTRange,nCols-nTRange]))/4)          
        ImageM = np.zeros( (nRows*3, nCols*3), np.uint8)
        ImageM[:] = nImageGreen

    
    for m in range (nRows):
        for n in range (nCols):
            ImageM[int(nRows)+m, int(nCols)+n] = aImage[m,n]
                
            
    #To remove white boundary (intensity= 25) if exist
    for i in range(0, ImageM.shape[0]): #flip x, Y, or X & Y 
        if( abs(i-aImage.shape[0]) <= nTRange or abs(i-2*aImage.shape[0]) <= nTRange):
            for j in range( 0, ImageM.shape[1]):
                if( nChannels == 3):                
                    ImageM[i,j]= (nImageBlue,nImageGreen, nImageRed)
                else:
                    ImageM[i,j]= nImageGreen
                    
        
    for j in range(0, ImageM.shape[1]): #flip x, Y, or X & Y 
        if( abs(j-aImage.shape[1]) <= nTRange or abs(j-2*aImage.shape[1]) <= nTRange):
            for i in range( 0, ImageM.shape[0]):
                if( nChannels == 3):                    
                    ImageM[i,j]= (nImageBlue,nImageGreen, nImageRed)
                else:
                    ImageM[i,j]= nImageGreen


    if( bPrintOutResult):
        if( nChannels == 3):
            print('==Original images [{}]=='.format(aImage.shape))
            plt.imshow(cv2.cvtColor(aImage, cv2.COLOR_BGR2RGB))
            plt.show()                 
            print('==Original images in a big matrix [{}]=='.format(ImageM.shape))
            plt.imshow(cv2.cvtColor(ImageM, cv2.COLOR_BGR2RGB))
            plt.show()                 
        else:
            print('==Original images [{}]=='.format(aImage.shape))
            plt.imshow(aImage)
            plt.show()                 
            print('==Original images in a big matrix [{}]=='.format(ImageM.shape))
            plt.imshow(ImageM)                            
            plt.show()                 


                        
    nNbrOfData=0
        
    #for nScale in range (100,140,15):
    #for nScale in range (85,120,15):
    for nScale in range (m_nScaleStart, m_nScaleEnd, m_nScaleStep):
                        
        #imageS = cv2.resize(ImageM, (int(ImageM.shape[0]*nScale*0.01),int(ImageM.shape[1]*nScale*0.01)))
        imageS = cv2.resize(ImageM, (int(ImageM.shape[1]*nScale*0.01),int(ImageM.shape[0]*nScale*0.01)))
                                 
        if( bPrintOutResult):
            if( nChannels == 3):
                print('==imageS Scale[{}] Size[{}]=='.format(nScale,imageS.shape))
                plt.imshow(cv2.cvtColor(imageS, cv2.COLOR_BGR2RGB))
                plt.show()                 
            else:
                print('==imageS Scale[{}] Size[{}]=='.format(nScale,imageS.shape))
                plt.imshow(imageS)
                plt.show()                 
        
        #for i in range(0,2): #flip x, Y, or X & Y 
        for i in range(m_nFlipStart, m_nFlipEnd, m_nFlipStep): #flip x, Y, or X & Y 
                               
            if( i == 0):                
                aImageF = imageS
                if(bPrintOutResult):
                    print("===original===")
            elif( i == 1 ):
                aImageF = cv2.flip(imageS, 0) #Horizonal flip
                if(bPrintOutResult):
                    print("===Flip Horizontal===")
            elif( i == 2 ):                    # do not use it because same as Horizontal flip  with 180 rotation
                aImageF = cv2.flip(imageS, 1) #vertical flip
                if(bPrintOutResult):
                    print("===Flip Vertical===")
            elif( i == 3 ):                     # do not use it because same as original image with 180 rotation
                aImageF = cv2.flip(imageS, -1) #Both flip
                if(bPrintOutResult):
                    print("===Flip Horizontal and Vertical===")
                
            if( False  ): #bPrintOutResult ):
                if( nChannels == 3):
                    print('==aImageF Scale[{}] Size[{}]=='.format(nScale,aImageF.shape))
                    plt.imshow(cv2.cvtColor(aImageF, cv2.COLOR_BGR2RGB))
                    plt.show()                 
                else:
                    print('==aImageF Scale[{}] Size[{}]=='.format(nScale,aImageF.shape))
                    plt.imshow(aImageF)
                    plt.show()               

                                    
            #for j in range (0,360,15):               
            for j in range (m_nRotationStart, m_nRotationEnd, m_nRotationStep):               
                if( j < m_nExcludeRange or j > 360-m_nExcludeRange ):
                    #M = cv2.getRotationMatrix2D((int(aImageF.shape[0]/2), int(aImageF.shape[1]/2)),j,1)
                    #imageR = cv2.warpAffine(aImageF,M,(aImageF.shape[0],aImageF.shape[1])) 
                    #ImageF = imageR[int(imageR.shape[0]/2-nRows/2):int(imageR.shape[0]/2+nRows/2),int(imageR.shape[1]/2-nCols/2):int(imageR.shape[1]/2+nCols/2) ]
                    
                    M = cv2.getRotationMatrix2D((int(aImageF.shape[1]/2), int(aImageF.shape[0]/2)),j,1)

                    if( nChannels == 3 ):
                        imageR = cv2.warpAffine(aImageF,M,(aImageF.shape[1],aImageF.shape[0]), borderValue=(nImageBlue,nImageGreen, nImageRed) ) 
                    else:
                        imageR = cv2.warpAffine(aImageF,M,(aImageF.shape[1],aImageF.shape[0]), borderValue=nImageGreen ) 
                        
                        
                    if( False ): #bPrintOutResult):
                        if( nChannels == 3):
                            print('==imageR Scale[{}] Size[{}]=='.format(nScale,imageR.shape))
                            plt.imshow(cv2.cvtColor(imageR, cv2.COLOR_BGR2RGB))
                            plt.show()                 
                        else:
                            print('==imageR Scale[{}] Size[{}]=='.format(nScale,imageR.shape))
                            plt.imshow(imageR)
                            plt.show()               
                       
                        
                    nCount=0
                    for i in range (100):
                        nRandom = random.randint(0,8)
                        if (nRandom != int(9/2)):
                            break
                                                 

                    for m in range (m_nTranslationStart, m_nTranslationEnd, m_nTranslationStep):
                        for n in range (m_nTranslationStart, m_nTranslationEnd, m_nTranslationStep):
                            row = aImage.shape[0]
                            col = aImage.shape[1]
                            
                            nTransRow = row - m
                            nTransCol = col - n
                                                
                            #ImageF = imageR[int(imageR.shape[0]/2-nRows/2):int(imageR.shape[0]/2+nRows/2),int(imageR.shape[1]/2-nCols/2):int(imageR.shape[1]/2+nCols/2) ]
                        
                            ImageT = imageR[nTransRow:nTransRow+row, nTransCol:nTransCol+col]
                        
                            if( m == 0 and n == 0):
                                DataX[nNbrOfData] = np.array([ImageT]) 
                                nNbrOfData+=1
                            elif ( nCount == nRandom ):
                                DataX[nNbrOfData] = np.array([ImageT]) 
                                nNbrOfData+=1
                            
                            nCount += 1
                                
                            if(bPrintOutResult):
                                print("=Scale=[{}] Flip=[{}] Rotate [{}] Translation[{},{}] Imageshape [{}] ".format(nScale,i,j,m,n, ImageT.shape))
                            
                                if( nChannels == 3):
                                    plt.imshow(cv2.cvtColor(ImageT, cv2.COLOR_BGR2RGB))
                                else:
                                    plt.imshow(ImageT)
                                    
                                plt.show()    
        
                                sDirectoryForOutputImages ='../temp'
                                #sDirectoryForOutputImages =m_sDirectoryForInputImagesAugment
                                
                                #sOutfileName='Sc{}-Fl{}-Ro{}-Trr{}-Trc{}-Real.jpg'.format(nScale,i,j,m,n)
                                #cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), imageR )                
                                sOutfileName='Sc{}-Fl{}-Ro{}-Trr{}-Trc{}-Final.jpg'.format(nScale,i,j,m,n)
                                cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), ImageT )                

                    
    return DataX, nNbrOfImages







# receive a Image, create augmented images, and return the images in numpy array
#Add translation function
def GetAugmentedImagesRegularOneInput( aImage,  nAugOption,  bPrintOutResult ):
    
    '''
    nNbrOfScales = 3    #(1.0,1.15, 1.30)
    nNbrOfFlips  = 2    #(original, flipX)
    #nNbrOfRotations = 12 #( 360/30 =12)
    nNbrOfRotations = 5 #( 0,15,30, 330, 345): -> ( 0,10,20, 340, 350)
    nNbrOfTranslation = 9 # (-10, 10)*(-10, 10)

    nNbrOfImages = nNbrOfScales*nNbrOfFlips*nNbrOfRotations
    '''
    
   #nNbrOfImages = GetNumberOfAugmentedImagesRegular()
     
    if( nAugOption == 125 ):
        nNbrOfImages = GetNumberOfAugmentedImagesRegular() 
    elif( nAugOption == 123 ):
        nNbrOfImages = 6  #nNbrOfAugmentations 
    elif( nAugOption == 121 ):
        nNbrOfImages = 2  #nNbrOfAugmentations 
    else:
        nNbrOfImages = GetNumberOfAugmentedImagesRegular()     
    
    
    nRows           = aImage.shape[0]
    nCols           = aImage.shape[1] 
    nImageLength    = len(aImage.shape)   
    
    if( nImageLength == 3):
        nChannels       = aImage.shape[2] 
    else:
        nChannels       = 1 
        
    
    if( nImageLength == 3):
        DataX = np.ndarray((nNbrOfImages, nRows, nCols, nChannels), dtype=np.uint8)
    else:
        DataX = np.ndarray((nNbrOfImages, nRows, nCols), dtype=np.uint8)
        
    nImageBlue=nImageGreen=nImageRed=0
    nTRange=5

    if( nChannels == 3 ):
        #nImageBlue  = int( (int(aImage[1,1,0]) + int(aImage[nRows-2,1,0]) + int(aImage[1,nCols-2,0]) + int(aImage[nRows-2,nCols-2,0]))/4)
        #nImageGreen = int( (int(aImage[1,1,1]) + int(aImage[nRows-2,1,1]) + int(aImage[1,nCols-2,1]) + int(aImage[nRows-2,nCols-2,1]))/4)
        #nImageRed   = int( (int(aImage[1,1,2]) + int(aImage[nRows-2,1,2]) + int(aImage[1,nCols-2,2]) + int(aImage[nRows-2,nCols-2,2]))/4)
        nTRange=10
        nImageBlue  = int( (int(aImage[nTRange,nTRange,0]) + int(aImage[nRows-nTRange,nTRange,0]) + int(aImage[nTRange,nCols-nTRange,0]) + int(aImage[nRows-nTRange,nCols-nTRange,0]))/4)
        nImageGreen = int( (int(aImage[nTRange,nTRange,1]) + int(aImage[nRows-nTRange,nTRange,1]) + int(aImage[nTRange,nCols-nTRange,1]) + int(aImage[nRows-nTRange,nCols-nTRange,1]))/4)
        nImageRed   = int( (int(aImage[nTRange,nTRange,2]) + int(aImage[nRows-nTRange,nTRange,2]) + int(aImage[nTRange,nCols-nTRange,2]) + int(aImage[nRows-nTRange,nCols-nTRange,2]))/4)
                        
        ImageM = np.zeros( (nRows*3, nCols*3, nChannels), np.uint8)
        ImageM[:] = (nImageBlue,nImageGreen, nImageRed)
    elif( nChannels == 4):            
        ImageM = np.zeros( (nRows*3, nCols*3, nChannels), np.uint8)
        ImageM[:] = (0, 0, 0, 0)
    elif( nChannels == 5):            
        ImageM = np.zeros( (nRows*3, nCols*3, nChannels), np.uint8)
        ImageM[:] = (0, 0, 0, 0, 0)
    elif( nChannels == 1):  
        nTRange=10
        nImageGreen = int( (int(aImage[nTRange,nTRange]) + int(aImage[nRows-nTRange,nTRange]) + int(aImage[nTRange,nCols-nTRange]) + int(aImage[nRows-nTRange,nCols-nTRange]))/4)          
        ImageM = np.zeros( (nRows*3, nCols*3), np.uint8)
        ImageM[:] = nImageGreen

    
    for m in range (nRows):
        for n in range (nCols):
            ImageM[int(nRows)+m, int(nCols)+n] = aImage[m,n]
                
            
    #To remove white boundary (intensity= 25) if exist
    for i in range(0, ImageM.shape[0]): #flip x, Y, or X & Y 
        if( abs(i-aImage.shape[0]) <= nTRange or abs(i-2*aImage.shape[0]) <= nTRange):
            for j in range( 0, ImageM.shape[1]):
                if( nChannels == 3):                
                    ImageM[i,j]= (nImageBlue,nImageGreen, nImageRed)
                else:
                    ImageM[i,j]= nImageGreen
                    
        
    for j in range(0, ImageM.shape[1]): #flip x, Y, or X & Y 
        if( abs(j-aImage.shape[1]) <= nTRange or abs(j-2*aImage.shape[1]) <= nTRange):
            for i in range( 0, ImageM.shape[0]):
                if( nChannels == 3):                    
                    ImageM[i,j]= (nImageBlue,nImageGreen, nImageRed)
                else:
                    ImageM[i,j]= nImageGreen


    if( bPrintOutResult):
        if( nChannels == 3):
            print('==Original images [{}]=='.format(aImage.shape))
            plt.imshow(cv2.cvtColor(aImage, cv2.COLOR_BGR2RGB))
            plt.show()                 
            print('==Original images in a big matrix [{}]=='.format(ImageM.shape))
            plt.imshow(cv2.cvtColor(ImageM, cv2.COLOR_BGR2RGB))
            plt.show()                 
        else:
            print('==Original images [{}]=='.format(aImage.shape))
            plt.imshow(aImage)
            plt.show()                 
            print('==Original images in a big matrix [{}]=='.format(ImageM.shape))
            plt.imshow(ImageM)                            
            plt.show()                 


                        
    nNbrOfData=0
        
    #for nScale in range (100,140,15):
    #for nScale in range (85,120,15):
    for nScale in range (m_nScaleStart, m_nScaleEnd, m_nScaleStep):
                        
        #imageS = cv2.resize(ImageM, (int(ImageM.shape[0]*nScale*0.01),int(ImageM.shape[1]*nScale*0.01)))
        imageS = cv2.resize(ImageM, (int(ImageM.shape[1]*nScale*0.01),int(ImageM.shape[0]*nScale*0.01)))
                                 
        if( bPrintOutResult):
            if( nChannels == 3):
                print('==imageS Scale[{}] Size[{}]=='.format(nScale,imageS.shape))
                plt.imshow(cv2.cvtColor(imageS, cv2.COLOR_BGR2RGB))
                plt.show()                 
            else:
                print('==imageS Scale[{}] Size[{}]=='.format(nScale,imageS.shape))
                plt.imshow(imageS)
                plt.show()                 
        
        #for i in range(0,2): #flip x, Y, or X & Y 
        for nFlip in range(m_nFlipStart, m_nFlipEnd, m_nFlipStep): #flip x, Y, or X & Y 
                               
            if( nFlip == 0):                
                aImageF = imageS
                if(bPrintOutResult):
                    print("===original===")
            elif( nFlip == 1 ):
                aImageF = cv2.flip(imageS, 0) #Horizonal flip
                if(bPrintOutResult):
                    print("===Flip Horizontal===")
            elif( nFlip == 2 ):                    # do not use it because same as Horizontal flip  with 180 rotation
                aImageF = cv2.flip(imageS, 1) #vertical flip
                if(bPrintOutResult):
                    print("===Flip Vertical===")
            elif( nFlip == 3 ):                     # do not use it because same as original image with 180 rotation
                aImageF = cv2.flip(imageS, -1) #Both flip
                if(bPrintOutResult):
                    print("===Flip Horizontal and Vertical===")
                
            if( False  ): #bPrintOutResult ):
                if( nChannels == 3):
                    print('==aImageF Scale[{}] Size[{}]=='.format(nScale,aImageF.shape))
                    plt.imshow(cv2.cvtColor(aImageF, cv2.COLOR_BGR2RGB))
                    plt.show()                 
                else:
                    print('==aImageF Scale[{}] Size[{}]=='.format(nScale,aImageF.shape))
                    plt.imshow(aImageF)
                    plt.show()               


            #20211227 to make less rotation 
            if( nAugOption == 125):
                nExcludeRange = m_nExcludeRange
            elif( nAugOption == 123):
                nExcludeRange = m_nExcludeRange - 10
            elif( nAugOption == 121):
                nExcludeRange = m_nExcludeRange - 23
            else:
                nExcludeRange = m_nExcludeRange
                

                                    
            #for j in range (0,360,15):               
            for j in range (m_nRotationStart, m_nRotationEnd, m_nRotationStep):               
                if( j < nExcludeRange or j > 360-nExcludeRange ):
                    #M = cv2.getRotationMatrix2D((int(aImageF.shape[0]/2), int(aImageF.shape[1]/2)),j,1)
                    #imageR = cv2.warpAffine(aImageF,M,(aImageF.shape[0],aImageF.shape[1])) 
                    #ImageF = imageR[int(imageR.shape[0]/2-nRows/2):int(imageR.shape[0]/2+nRows/2),int(imageR.shape[1]/2-nCols/2):int(imageR.shape[1]/2+nCols/2) ]
                    
                    M = cv2.getRotationMatrix2D((int(aImageF.shape[1]/2), int(aImageF.shape[0]/2)),j,1)

                    if( nChannels == 3 ):
                        imageR = cv2.warpAffine(aImageF,M,(aImageF.shape[1],aImageF.shape[0]), borderValue=(nImageBlue,nImageGreen, nImageRed) ) 
                    else:
                        imageR = cv2.warpAffine(aImageF,M,(aImageF.shape[1],aImageF.shape[0]), borderValue=nImageGreen ) 
                        
                        
                    if( False ): #bPrintOutResult):
                        if( nChannels == 3):
                            print('==imageR Scale[{}] Size[{}]=='.format(nScale,imageR.shape))
                            plt.imshow(cv2.cvtColor(imageR, cv2.COLOR_BGR2RGB))
                            plt.show()                 
                        else:
                            print('==imageR Scale[{}] Size[{}]=='.format(nScale,imageR.shape))
                            plt.imshow(imageR)
                            plt.show()               
                       
                        
                    nCount=0
                    for m in range (100):
                        nRandom = random.randint(0,8)
                        if (nRandom != int(9/2)):
                            break
                                                 

                    for m in range (m_nTranslationStart, m_nTranslationEnd, m_nTranslationStep):
                        for n in range (m_nTranslationStart, m_nTranslationEnd, m_nTranslationStep):
                            row = aImage.shape[0]
                            col = aImage.shape[1]
                            
                            nTransRow = row - m
                            nTransCol = col - n
                                                
                            #ImageF = imageR[int(imageR.shape[0]/2-nRows/2):int(imageR.shape[0]/2+nRows/2),int(imageR.shape[1]/2-nCols/2):int(imageR.shape[1]/2+nCols/2) ]
                        
                            ImageT = imageR[nTransRow:nTransRow+row, nTransCol:nTransCol+col]
                        
                            if( m == 0 and n == 0):
                                DataX[nNbrOfData] = np.array([ImageT]) 
                                nNbrOfData+=1
                            elif ( nCount == nRandom ):
                                DataX[nNbrOfData] = np.array([ImageT]) 
                                nNbrOfData+=1
                            
                            nCount += 1
                                
                            if(bPrintOutResult):
                                print("=Scale=[{}] Flip=[{}] Rotate [{}] Translation[{},{}] Imageshape [{}] ".format(nScale,nFlip,j,m,n, ImageT.shape))
                            
                                if( nChannels == 3):
                                    plt.imshow(cv2.cvtColor(ImageT, cv2.COLOR_BGR2RGB))
                                else:
                                    plt.imshow(ImageT)
                                    
                                plt.show()    
        
                                sDirectoryForOutputImages ='../temp'
                                #sDirectoryForOutputImages =m_sDirectoryForInputImagesAugment
                                
                                #sOutfileName='Sc{}-Fl{}-Ro{}-Trr{}-Trc{}-Real.jpg'.format(nScale,i,j,m,n)
                                #cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), imageR )                
                                sOutfileName='Sc{}-Fl{}-Ro{}-Trr{}-Trc{}-Final.jpg'.format(nScale,nFlip,j,m,n)
                                cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), ImageT )                

                    
    return DataX, nNbrOfImages



# receive a Image, create augmented images, and return the images in numpy array
def GetAugmentedImagesRegularOneInput_20191209( aImage, bPrintOutResult ):
    
    '''
    nNbrOfScales = 3    #(1.0,1.15, 1.30)
    nNbrOfFlips  = 2    #(original, flipX)
    #nNbrOfRotations = 12 #( 360/30 =12)
    nNbrOfRotations = 5 #( 0,15,30, 330, 345)

    nNbrOfImages = nNbrOfScales*nNbrOfFlips*nNbrOfRotations
    '''
    
    nNbrOfImages = GetNumberOfAugmentedImagesRegular()
    
    nRows           = aImage.shape[0]
    nCols           = aImage.shape[1] 
    nImageLength    = len(aImage.shape)   
    
    if( nImageLength == 3):
        nChannels       = aImage.shape[2] 
    else:
        nChannels       = 1 
        
    
    if( nImageLength == 3):
        DataX = np.ndarray((nNbrOfImages, nRows, nCols, nChannels), dtype=np.uint8)
    else:
        DataX = np.ndarray((nNbrOfImages, nRows, nCols), dtype=np.uint8)
        
    nImageBlue=nImageGreen=nImageRed=0
    nTRange=5

    if( nChannels == 3 ):
        #nImageBlue  = int( (int(aImage[1,1,0]) + int(aImage[nRows-2,1,0]) + int(aImage[1,nCols-2,0]) + int(aImage[nRows-2,nCols-2,0]))/4)
        #nImageGreen = int( (int(aImage[1,1,1]) + int(aImage[nRows-2,1,1]) + int(aImage[1,nCols-2,1]) + int(aImage[nRows-2,nCols-2,1]))/4)
        #nImageRed   = int( (int(aImage[1,1,2]) + int(aImage[nRows-2,1,2]) + int(aImage[1,nCols-2,2]) + int(aImage[nRows-2,nCols-2,2]))/4)
        nTRange=10
        nImageBlue  = int( (int(aImage[nTRange,nTRange,0]) + int(aImage[nRows-nTRange,nTRange,0]) + int(aImage[nTRange,nCols-nTRange,0]) + int(aImage[nRows-nTRange,nCols-nTRange,0]))/4)
        nImageGreen = int( (int(aImage[nTRange,nTRange,1]) + int(aImage[nRows-nTRange,nTRange,1]) + int(aImage[nTRange,nCols-nTRange,1]) + int(aImage[nRows-nTRange,nCols-nTRange,1]))/4)
        nImageRed   = int( (int(aImage[nTRange,nTRange,2]) + int(aImage[nRows-nTRange,nTRange,2]) + int(aImage[nTRange,nCols-nTRange,2]) + int(aImage[nRows-nTRange,nCols-nTRange,2]))/4)
                        
        ImageM = np.zeros( (nRows*3, nCols*3, nChannels), np.uint8)
        ImageM[:] = (nImageBlue,nImageGreen, nImageRed)
    elif( nChannels == 4):            
        ImageM = np.zeros( (nRows*3, nCols*3, nChannels), np.uint8)
        ImageM[:] = (0, 0, 0, 0)
    elif( nChannels == 5):            
        ImageM = np.zeros( (nRows*3, nCols*3, nChannels), np.uint8)
        ImageM[:] = (0, 0, 0, 0, 0)
    elif( nChannels == 1):  
        nTRange=10
        nImageGreen = int( (int(aImage[nTRange,nTRange]) + int(aImage[nRows-nTRange,nTRange]) + int(aImage[nTRange,nCols-nTRange]) + int(aImage[nRows-nTRange,nCols-nTRange]))/4)          
        ImageM = np.zeros( (nRows*3, nCols*3), np.uint8)
        ImageM[:] = nImageGreen

    
    for m in range (nRows):
        for n in range (nCols):
            ImageM[int(nRows)+m, int(nCols)+n] = aImage[m,n]
                
            
    #To remove white boundary (intensity= 25) if exist
    for i in range(0, ImageM.shape[0]): #flip x, Y, or X & Y 
        if( abs(i-aImage.shape[0]) <= nTRange or abs(i-2*aImage.shape[0]) <= nTRange):
            for j in range( 0, ImageM.shape[1]):
                if( nChannels == 3):                
                    ImageM[i,j]= (nImageBlue,nImageGreen, nImageRed)
                else:
                    ImageM[i,j]= nImageGreen
                    
        
    for j in range(0, ImageM.shape[1]): #flip x, Y, or X & Y 
        if( abs(j-aImage.shape[1]) <= nTRange or abs(j-2*aImage.shape[1]) <= nTRange):
            for i in range( 0, ImageM.shape[0]):
                if( nChannels == 3):                    
                    ImageM[i,j]= (nImageBlue,nImageGreen, nImageRed)
                else:
                    ImageM[i,j]= nImageGreen


    if( bPrintOutResult):
        if( nChannels == 3):
            print('==Original images [{}]=='.format(aImage.shape))
            plt.imshow(cv2.cvtColor(aImage, cv2.COLOR_BGR2RGB))
            plt.show()                 
            print('==Original images in a big matrix [{}]=='.format(ImageM.shape))
            plt.imshow(cv2.cvtColor(ImageM, cv2.COLOR_BGR2RGB))
            plt.show()                 
        else:
            print('==Original images [{}]=='.format(aImage.shape))
            plt.imshow(aImage)
            plt.show()                 
            print('==Original images in a big matrix [{}]=='.format(ImageM.shape))
            plt.imshow(ImageM)                            
            plt.show()                 


                        
    nNbrOfData=0
        
    #for nScale in range (100,140,15):
    #for nScale in range (85,120,15):
    for nScale in range (m_nScaleStart, m_nScaleEnd, m_nScaleStep):
                        
        #imageS = cv2.resize(ImageM, (int(ImageM.shape[0]*nScale*0.01),int(ImageM.shape[1]*nScale*0.01)))
        imageS = cv2.resize(ImageM, (int(ImageM.shape[1]*nScale*0.01),int(ImageM.shape[0]*nScale*0.01)))
                                 
        if( bPrintOutResult):
            if( nChannels == 3):
                print('==imageS Scale[{}] Size[{}]=='.format(nScale,imageS.shape))
                plt.imshow(cv2.cvtColor(imageS, cv2.COLOR_BGR2RGB))
                plt.show()                 
            else:
                print('==imageS Scale[{}] Size[{}]=='.format(nScale,imageS.shape))
                plt.imshow(imageS)
                plt.show()                 
        
        #for i in range(0,2): #flip x, Y, or X & Y 
        for i in range(m_nFlipStart, m_nFlipEnd, m_nFlipStep): #flip x, Y, or X & Y 
                               
            if( i == 0):                
                aImageF = imageS
                if(bPrintOutResult):
                    print("===original===")
            elif( i == 1 ):
                aImageF = cv2.flip(imageS, 0) #Horizonal flip
                if(bPrintOutResult):
                    print("===Flip Horizontal===")
            elif( i == 2 ):                    # do not use it because same as Horizontal flip  with 180 rotation
                aImageF = cv2.flip(imageS, 1) #vertical flip
                if(bPrintOutResult):
                    print("===Flip Vertical===")
            elif( i == 3 ):                     # do not use it because same as original image with 180 rotation
                aImageF = cv2.flip(imageS, -1) #Both flip
                if(bPrintOutResult):
                    print("===Flip Horizontal and Vertical===")
                
            if( bPrintOutResult):
                if( nChannels == 3):
                    print('==aImageF Scale[{}] Size[{}]=='.format(nScale,aImageF.shape))
                    plt.imshow(cv2.cvtColor(aImageF, cv2.COLOR_BGR2RGB))
                    plt.show()                 
                else:
                    print('==aImageF Scale[{}] Size[{}]=='.format(nScale,aImageF.shape))
                    plt.imshow(aImageF)
                    plt.show()               

                                    
            #for j in range (0,360,15):               
            for j in range (m_nRotationStart, m_nRotationEnd, m_nRotationStep):               
                if( j < m_nExcludeRange or j > 360-m_nExcludeRange ):
                    #M = cv2.getRotationMatrix2D((int(aImageF.shape[0]/2), int(aImageF.shape[1]/2)),j,1)
                    #imageR = cv2.warpAffine(aImageF,M,(aImageF.shape[0],aImageF.shape[1])) 
                    #ImageF = imageR[int(imageR.shape[0]/2-nRows/2):int(imageR.shape[0]/2+nRows/2),int(imageR.shape[1]/2-nCols/2):int(imageR.shape[1]/2+nCols/2) ]
                    
                    M = cv2.getRotationMatrix2D((int(aImageF.shape[1]/2), int(aImageF.shape[0]/2)),j,1)

                    if( nChannels == 3 ):
                        imageR = cv2.warpAffine(aImageF,M,(aImageF.shape[1],aImageF.shape[0]), borderValue=(nImageBlue,nImageGreen, nImageRed) ) 
                    else:
                        imageR = cv2.warpAffine(aImageF,M,(aImageF.shape[1],aImageF.shape[0]), borderValue=nImageGreen ) 
                        
                        
                    if( bPrintOutResult):
                        if( nChannels == 3):
                            print('==imageR Scale[{}] Size[{}]=='.format(nScale,imageR.shape))
                            plt.imshow(cv2.cvtColor(imageR, cv2.COLOR_BGR2RGB))
                            plt.show()                 
                        else:
                            print('==imageR Scale[{}] Size[{}]=='.format(nScale,imageR.shape))
                            plt.imshow(imageR)
                            plt.show()               
                                                    
                    ImageF = imageR[int(imageR.shape[0]/2-nRows/2):int(imageR.shape[0]/2+nRows/2),int(imageR.shape[1]/2-nCols/2):int(imageR.shape[1]/2+nCols/2) ]
                          
                    if(bPrintOutResult):
                        print("=Scale=[{}] Flip=[{}] Rotate [{}]  Imageshape [{}] ".format(nScale,i,j, ImageF.shape))
                    
                        if( nChannels == 3):
                            plt.imshow(cv2.cvtColor(ImageF, cv2.COLOR_BGR2RGB))
                        else:
                            plt.imshow(ImageF)
                            
                        plt.show()    
                    
                    
                    DataX[nNbrOfData] = np.array([ImageF])  
                    
                    nNbrOfData+=1
                            
                    if(bPrintOutResult):
                        print("=Scale=[{}] Flip=[{}] Rotate [{}]  Imageshape [{}] ".format(nScale,i,j, ImageF.shape))
                    
                        if( nChannels == 3):
                            plt.imshow(cv2.cvtColor(ImageF, cv2.COLOR_BGR2RGB))
                        else:
                            plt.imshow(ImageF)
                            
                        plt.show()    

                        sDirectoryForOutputImages ='./temp'
                        
                        sOutfileName='Sc{}-Fl{}-Ro{}-Real.jpg'.format(nScale,i,j)
                        cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), imageR )                
                        sOutfileName='Sc{}-Fl{}-Ro{}-Final.jpg'.format(nScale,i,j)
                        cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), ImageF )                

                    
    return DataX, nNbrOfImages





# receive a Image and the corresponding label image, create augmented images and label images, and return the images in numpy arrays
def GetAugmentedImagesRegularTwoInputs( aImage, aImageLabel):

    bPrintOut = m_bPrintOutResult
    

    
    #nNbrOfImages = GetNumberOfAugmentedImagesRegular()
    nNbrOfImages = m_nNbrOfAugmentation

    nRows = aImage.shape[0]
    nCols = aImage.shape[1] 
    nImageChannels = len(aImage.shape)   
    nImageLabelChannels = len(aImageLabel.shape)   

    
    if( nImageChannels == 3):
        DataX = np.ndarray((nNbrOfImages, nRows, nCols, nImageChannels), dtype=np.uint8)
    else:
        DataX = np.ndarray((nNbrOfImages, nRows, nCols), dtype=np.uint8)
        
    if( nImageLabelChannels == 3):
        DataLabelX = np.ndarray((nNbrOfImages, nRows, nCols, nImageLabelChannels), dtype=np.uint8)
    else:
        DataLabelX = np.ndarray((nNbrOfImages, nRows, nCols), dtype=np.uint8)

    if( nImageChannels == 3 ):
        nImageBlue  = int( (int(aImage[1,1,0]) + int(aImage[nRows-2,1,0]) + int(aImage[1,nCols-2,0]) + int(aImage[nRows-2,nCols-2,0]))/4)
        nImageGreen = int( (int(aImage[1,1,1]) + int(aImage[nRows-2,1,1]) + int(aImage[1,nCols-2,1]) + int(aImage[nRows-2,nCols-2,1]))/4)
        nImageRed   = int( (int(aImage[1,1,2]) + int(aImage[nRows-2,1,2]) + int(aImage[1,nCols-2,2]) + int(aImage[nRows-2,nCols-2,2]))/4)
                
        ImageM = np.zeros( (nRows*3, nCols*3,3), np.uint8)
        ImageM[:] = (nImageBlue,nImageGreen, nImageRed)
    else: 
        nImageGreen = int( (int(aImage[1,1]) + int(aImage[nRows-2,1]) + int(aImage[1,nCols-2]) + int(aImage[nRows-2,nCols-2]))/4)           
        ImageM = np.zeros( (nRows*3, nCols*3), np.uint8)
        ImageM[:] = nImageGreen

    ImageLabelM = np.zeros( (nRows*3, nCols*3), np.uint8)
    ImageLabelM[:] = 0
    
    ImageMCopy = ImageM.copy()
    ImageLabelMCopy = ImageLabelM.copy()
    
    
    
    for m in range (nRows):
        for n in range (nCols):
            ImageM[int(nRows)+m, int(nCols)+n] = aImage[m,n]
            ImageLabelM[int(nRows)+m, int(nCols)+n] = aImageLabel[m,n]
                
                        
    nNbrOfData=0
    
    
    '''    
    m_bScaleUsage   = True
    m_nScaleStart   = 95    
    c     = 105
    
    m_bFlipUsage    = True;
    m_nFlipStart    = 0   #(0:original, 1:flip using X axis, 2:flip based on Y axis)
    m_nFlipEnd      = 3
    
    
    m_bRotationUsage    = True;
    m_nRotationStart    = 0
    m_nRotationEnd      = 360
    
    
    m_bTranslationUsage    = True;
    m_nTranslationStart    = -10
    m_nTranslationEnd      =  11
    '''
    
    
    for i in range(m_nFlipStart, m_nFlipEnd, m_nFlipStep): #flip x, Y, or X & Y 
                               
        if( i == 0):                
            aImageF         = ImageM
            aImageLabelF    = ImageLabelM
            if(bPrintOut):
                print("===original===")
        elif( i == 1 ):                    # do not use it because same as Horizontal flip  with 180 rotation
            aImageF         = cv2.flip(ImageM, 0) #vertical flip
            aImageLabelF    = cv2.flip(ImageLabelM, 0) #vertical flip
            if(bPrintOut):
                print("===Flip Vertical===")
        elif( i == 2 ):
            aImageF         = cv2.flip(ImageM, 1) #Horizonal flip
            aImageLabelF    = cv2.flip(ImageLabelM, 1) #Horizonal flip
            if(bPrintOut):
                print("===Flip Horizontal===")
        elif( i == 3 ):                     # do not use it because same as original image with 180 rotation
            aImageF         = cv2.flip(ImageM, -1) #Both flip
            aImageLabelF    = cv2.flip(ImageLabelM, -1) #Both flip
            if(bPrintOut):
                print("===Flip Horizontal and Vertical===")    
    
    
    
        if( m_bFlipUsage == True):
            nNbrOfAugmentation = int(m_nNbrOfAugmentation/2)
        else:
            nNbrOfAugmentation = int(m_nNbrOfAugmentation)
            
    
    
        for y in range( nNbrOfAugmentation):
            
            if( y == 0 ):
                
                
                row = aImage.shape[0]
                col = aImage.shape[1]
            
                nTransRow = row 
                nTransCol = col 
                                                                        
                aImageT      = aImageF[nTransRow:nTransRow+row, nTransCol:nTransCol+col]
                aImageLabelT = aImageLabelF[nTransRow:nTransRow+row, nTransCol:nTransCol+col]                
                
                
                DataX[nNbrOfData] = np.array([aImageT]) 
                DataLabelX[nNbrOfData] = np.array([aImageLabelT])  
                nNbrOfData+=1
                                                               
                if(bPrintOut):
                    nScale=100
                    nFlip=i
                    nRotation=0
                    nTransRow=0
                    nTransCol=0
                    
                    print("=Scale=[{}] Flip=[{}] Rotate [{}] Translation[{},{}] Imageshape [{}]  ImageLabelshape [{}] ".format(nScale,nFlip,nRotation,nTransRow, nTransCol, aImageT.shape, aImageLabelT.shape))
                                
                    if( nImageChannels == 3):
                        plt.imshow(cv2.cvtColor(aImageT, cv2.COLOR_BGR2RGB))
                    else:
                        plt.imshow(aImageT)
                                        
                    plt.show()    
            
                    plt.imshow(aImageLabelT)
                    plt.show()  
    
                    sDirectoryForOutputImages ='../temp'
                                    
                    sOutfileName='Sc{}-Fl{}-Ro{}-Trr{}-Trc{}-Image.jpg'.format(nScale,nFlip,nRotation,nTransRow, nTransCol)
                    cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), aImageT )
                                    
                    sOutfileName='Sc{}-Fl{}-Ro{}-Trr{}-Trc{}-Label.jpg'.format(nScale,nFlip,nRotation,nTransRow, nTransCol)                                
                    cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), aImageLabelT )  
                    
                continue
                
                
            
            if(m_bScaleUsage == True):
                nScale = random.randint(m_nScaleStart,m_nScaleEnd)
            else:
                nScale = 100
                
            aImageS = cv2.resize(aImageF, (int(aImageF.shape[1]*nScale*0.01),int(aImageF.shape[0]*nScale*0.01)))
            aImageLabelS = cv2.resize(aImageLabelF, (int(aImageLabelF.shape[1]*nScale*0.01),int(aImageLabelF.shape[0]*nScale*0.01)))
            
            #20191225 move rescaled images in the middle
            if( nScale < 100 ):
                nReSize = int(aImageF.shape[0]*nScale*0.01)
                nRowEnd   = nReSize               
                nRowDif  = int(abs((int(aImageF.shape[0]) - int(aImageF.shape[0]*nScale*0.01))/2))
                
                nReSize = int(aImageF.shape[1]*nScale*0.01)
                nColEnd   = nReSize                
                nColDif  = int(abs((int(aImageF.shape[1]) - int(aImageF.shape[1]*nScale*0.01))/2))
                          
                aImageSM =ImageMCopy.copy()                                             
                aImageSM[nRowDif:nRowEnd+nRowDif, nColDif:nColEnd+nColDif] = aImageS[0:nRowEnd, 0:nColEnd]
                aImageS = aImageSM.copy()                
                
                aImageLabelSM =ImageLabelMCopy.copy()                                                             
                aImageLabelSM[nRowDif:nRowEnd+nRowDif, nColDif:nColEnd+nColDif] = aImageLabelS[0:nRowEnd, 0:nColEnd]
                aImageLabelS = aImageLabelSM.copy()
                
                
            elif( nScale > 100 ):
                nReSize = int(aImageF.shape[0]*nScale*0.01)
                nRowDif  = int(abs((int(aImageF.shape[0]) - int(aImageF.shape[0]*nScale*0.01))/2))
                
                nReSize = int(aImageF.shape[1]*nScale*0.01)
                nColDif  = int(abs((int(aImageF.shape[1]) - int(aImageF.shape[1]*nScale*0.01))/2))
                          
                aImageSM =ImageMCopy.copy()                                             
                aImageSM[0:int(aImageF.shape[0])-nRowDif, 0:int(aImageF.shape[1])-nColDif] = aImageS[nRowDif:int(aImageF.shape[0]), nColDif:int(aImageF.shape[1])]
                aImageS = aImageSM.copy()                
                
                aImageLabelSM =ImageLabelMCopy.copy()                                                             
                aImageLabelSM[0:int(aImageF.shape[0])-nRowDif, 0:int(aImageF.shape[1])-nColDif] = aImageLabelS[nRowDif:int(aImageF.shape[0]), nColDif:int(aImageF.shape[1])]
                aImageLabelS = aImageLabelSM.copy()

            if(m_bRotationUsage == True):
                
                nRotation = 180
                while ( nRotation > m_nExcludeRange and nRotation < 360-m_nExcludeRange ):
                    nRotation = random.randint(m_nRotationStart,m_nRotationEnd)                    
                    
                M = cv2.getRotationMatrix2D((int(aImageS.shape[1]/2), int(aImageS.shape[0]/2)),nRotation,1)
                aImageR      = cv2.warpAffine(aImageS,M,(aImageS.shape[1],aImageS.shape[0]))
                aImageLabelR = cv2.warpAffine(aImageLabelS,M,(aImageLabelS.shape[1],aImageLabelS.shape[0]))
                    
            else:
                nRotation = 0
                aImageR      = aImageS
                aImageLabelR = aImageLabelS
                
                            
            #ListRotation.insert(0,0) 
            row = aImage.shape[0]
            col = aImage.shape[1]
            
            if (m_bTranslationUsage == True ):
                                            
                nTransRow = row - random.randint(m_nTranslationStart,m_nTranslationEnd)
                nTransCol = col - random.randint(m_nTranslationStart,m_nTranslationEnd)
                                                                        
                aImageT      = aImageR[nTransRow:nTransRow+row, nTransCol:nTransCol+col]
                aImageLabelT = aImageLabelR[nTransRow:nTransRow+row, nTransCol:nTransCol+col]
                
            else:
                nTransRow = row 
                nTransCol = col 
                                                                        
                aImageT      = aImageR[nTransRow:nTransRow+row, nTransCol:nTransCol+col]
                aImageLabelT = aImageLabelR[nTransRow:nTransRow+row, nTransCol:nTransCol+col]
                
                    
            DataX[nNbrOfData] = np.array([aImageT]) 
            DataLabelX[nNbrOfData] = np.array([aImageLabelT])  
            nNbrOfData+=1
                                                           
            if(bPrintOut):
                print("=Scale=[{}] Flip=[{}] Rotate [{}] Translation[{},{}] Imageshape [{}]  ImageLabelshape [{}] ".format(nScale,i,nRotation,nTransRow, nTransCol, aImageT.shape, aImageLabelT.shape))
                            
                if( nImageChannels == 3):
                    plt.imshow(cv2.cvtColor(aImageT, cv2.COLOR_BGR2RGB))
                else:
                    plt.imshow(aImageT)
                                    
                plt.show()    
        
                plt.imshow(aImageLabelT)
                plt.show()  

                sDirectoryForOutputImages ='../temp'
                                
                sOutfileName='Sc{}-Fl{}-Ro{}-Trr{}-Trc{}-Image.jpg'.format(nScale,i,nRotation,nTransRow, nTransCol)
                cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), aImageT )
                                
                sOutfileName='Sc{}-Fl{}-Ro{}-Trr{}-Trc{}-Label.jpg'.format(nScale,i,nRotation,nTransRow, nTransCol)                                
                cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), aImageLabelT )                
                    
                            
                    
                    
    #return DataX, DataLabelX, nNbrOfImages
    return DataX, DataLabelX, nNbrOfData



# receive a Image and the corresponding label image, create augmented images and label images, and return the images in numpy arrays
def GetAugmentedImagesRegularTwoInputsS1F2R3T2( aImage, aImageLabel ):

    bPrintOut = m_bPrintOutResult
    
    '''
    nNbrOfScales = 3    #(1.0,1.15, 1.30)
    nNbrOfFlips  = 2    #(original, flipX)
    nNbrOfRotations = 12 #( 360/30 =12)
    nNbrOfImages = nNbrOfScales*nNbrOfFlips*nNbrOfRotations
    '''
    
    nNbrOfImages = GetNumberOfAugmentedImagesRegular()

    nRows = aImage.shape[0]
    nCols = aImage.shape[1] 
    nImageChannels = len(aImage.shape)   
    nImageLabelChannels = len(aImageLabel.shape)   

    
    if( nImageChannels == 3):
        DataX = np.ndarray((nNbrOfImages, nRows, nCols, nImageChannels), dtype=np.uint8)
    else:
        DataX = np.ndarray((nNbrOfImages, nRows, nCols), dtype=np.uint8)
        
    if( nImageLabelChannels == 3):
        DataLabelX = np.ndarray((nNbrOfImages, nRows, nCols, nImageLabelChannels), dtype=np.uint8)
    else:
        DataLabelX = np.ndarray((nNbrOfImages, nRows, nCols), dtype=np.uint8)

    if( nImageChannels == 3 ):
        nImageBlue  = int( (int(aImage[1,1,0]) + int(aImage[nRows-2,1,0]) + int(aImage[1,nCols-2,0]) + int(aImage[nRows-2,nCols-2,0]))/4)
        nImageGreen = int( (int(aImage[1,1,1]) + int(aImage[nRows-2,1,1]) + int(aImage[1,nCols-2,1]) + int(aImage[nRows-2,nCols-2,1]))/4)
        nImageRed   = int( (int(aImage[1,1,2]) + int(aImage[nRows-2,1,2]) + int(aImage[1,nCols-2,2]) + int(aImage[nRows-2,nCols-2,2]))/4)
                
        ImageM = np.zeros( (nRows*3, nCols*3,3), np.uint8)
        ImageM[:] = (nImageBlue,nImageGreen, nImageRed)
    else: 
        nImageGreen = int( (int(aImage[1,1]) + int(aImage[nRows-2,1]) + int(aImage[1,nCols-2]) + int(aImage[nRows-2,nCols-2]))/4)           
        ImageM = np.zeros( (nRows*3, nCols*3), np.uint8)
        ImageM[:] = nImageGreen

    ImageLabelM = np.zeros( (nRows*3, nCols*3), np.uint8)
    ImageLabelM[:] = 0
    
    for m in range (nRows):
        for n in range (nCols):
            ImageM[int(nRows)+m, int(nCols)+n] = aImage[m,n]
            ImageLabelM[int(nRows)+m, int(nCols)+n] = aImageLabel[m,n]
                
                        
    nNbrOfData=0
    #for nScale in range (100,141,15):
    #for nScale in range (85,120,15):
    for nScale in range (m_nScaleStart, m_nScaleEnd, m_nScaleStep):
                
        #imageS = cv2.resize(ImageM, (int(ImageM.shape[0]*nScale*0.01),int(ImageM.shape[1]*nScale*0.01)))
        #imageLabelS = cv2.resize(ImageLabelM, (int(ImageLabelM.shape[0]*nScale*0.01),int(ImageLabelM.shape[1]*nScale*0.01)))                                 
        imageS = cv2.resize(ImageM, (int(ImageM.shape[1]*nScale*0.01),int(ImageM.shape[0]*nScale*0.01)))
        imageLabelS = cv2.resize(ImageLabelM, (int(ImageLabelM.shape[1]*nScale*0.01),int(ImageLabelM.shape[0]*nScale*0.01)))

        #for i in range(0,2): #flip x, Y, or X & Y 
        for i in range(m_nFlipStart, m_nFlipEnd, m_nFlipStep): #flip x, Y, or X & Y 
                               
            if( i == 0):                
                aImageF         = imageS
                aImageLabelF    = imageLabelS
                if(bPrintOut):
                    print("===original===")
            elif( i == 1 ):                    # do not use it because same as Horizontal flip  with 180 rotation
                aImageF         = cv2.flip(imageS, 0) #vertical flip
                aImageLabelF    = cv2.flip(imageLabelS, 0) #vertical flip
                if(bPrintOut):
                    print("===Flip Vertical===")
            elif( i == 2 ):
                aImageF         = cv2.flip(imageS, 1) #Horizonal flip
                aImageLabelF    = cv2.flip(imageLabelS, 1) #Horizonal flip
                if(bPrintOut):
                    print("===Flip Horizontal===")
            elif( i == 3 ):                     # do not use it because same as original image with 180 rotation
                aImageF         = cv2.flip(imageS, -1) #Both flip
                aImageLabelF    = cv2.flip(imageLabelS, -1) #Both flip
                if(bPrintOut):
                    print("===Flip Horizontal and Vertical===")
                
                                    
            #for j in range (0,360,15):
            for j in range (m_nRotationStart, m_nRotationEnd, m_nRotationStep):
                if( j < m_nExcludeRange or j > 360-m_nExcludeRange ):
                
                    #M = cv2.getRotationMatrix2D((int(aImageF.shape[0]/2), int(aImageF.shape[1]/2)),j,1)
                    #imageR      = cv2.warpAffine(aImageF,M,(aImageF.shape[0],aImageF.shape[1]))
                    #imageLabelR = cv2.warpAffine(aImageLabelF,M,(aImageLabelF.shape[0],aImageLabelF.shape[1]))
                    M = cv2.getRotationMatrix2D((int(aImageF.shape[1]/2), int(aImageF.shape[0]/2)),j,1)
                    imageR      = cv2.warpAffine(aImageF,M,(aImageF.shape[1],aImageF.shape[0]))
                    imageLabelR = cv2.warpAffine(aImageLabelF,M,(aImageLabelF.shape[1],aImageLabelF.shape[0]))
                                  
                    #ImageF      = imageR[int(imageR.shape[0]/2-nRows/2):int(imageR.shape[0]/2+nRows/2),int(imageR.shape[1]/2-nCols/2):int(imageR.shape[1]/2+nCols/2) ]
                    #ImageLabelF = imageLabelR[int(imageLabelR.shape[0]/2-nRows/2):int(imageLabelR.shape[0]/2+nRows/2),int(imageLabelR.shape[1]/2-nCols/2):int(imageLabelR.shape[1]/2+nCols/2) ]
                          
                    #DataX[nNbrOfData] = np.array([ImageF])  
                    #DataLabelX[nNbrOfData] = np.array([ImageLabelF])  
                    #nNbrOfData+=1

                    #if(bPrintOut):
                    #    print("[{}] =Scale=[{}] Flip=[{}] Rotate [{}]  Imageshape [{}] ImageLabelshape [{}] ".format(nNbrOfData, nScale,i,j, ImageF.shape, ImageLabelF.shape))
                    #
                    #    if( nImageChannels == 3):
                    #        plt.imshow(cv2.cvtColor(ImageF, cv2.COLOR_BGR2RGB))
                    #    else:
                    #        plt.imshow(ImageF)
                    #        
                    #    plt.show()  
                    #    
                    #    plt.imshow(ImageLabelF)
                    #    plt.show()  

                    
                    nCount=0
                    for i in range (100):
                        nRandom = random.randint(0,8)
                        if (nRandom != int(9/2)):
                            break
                        
                    
                    for m in range (m_nTranslationStart, m_nTranslationEnd, m_nTranslationStep):
                        for n in range (m_nTranslationStart, m_nTranslationEnd, m_nTranslationStep):
                            row = aImage.shape[0]
                            col = aImage.shape[1]
                            
                            nTransRow = row - m
                            nTransCol = col - n
                                                
                            #ImageF = imageR[int(imageR.shape[0]/2-nRows/2):int(imageR.shape[0]/2+nRows/2),int(imageR.shape[1]/2-nCols/2):int(imageR.shape[1]/2+nCols/2) ]
                        
                            ImageT      = imageR[nTransRow:nTransRow+row, nTransCol:nTransCol+col]
                            ImageLabelT = imageLabelR[nTransRow:nTransRow+row, nTransCol:nTransCol+col]
                        
                            #DataX[nNbrOfData] = np.array([ImageT])  
                            #DataLabelX[nNbrOfData] = np.array([ImageLabelT])                                                                           
                            #nNbrOfData+=1
                    
                            if( m == 0 and n == 0):
                                DataX[nNbrOfData] = np.array([ImageT]) 
                                DataLabelX[nNbrOfData] = np.array([ImageLabelT])  
                                nNbrOfData+=1
                            elif ( nCount == nRandom ):
                                DataX[nNbrOfData] = np.array([ImageT]) 
                                DataLabelX[nNbrOfData] = np.array([ImageLabelT])  
                                nNbrOfData+=1
                            
                            nCount += 1
                    
                    
                    
                            if(bPrintOut):
                                print("=Scale=[{}] Flip=[{}] Rotate [{}] Translation[{},{}] Imageshape [{}]  ImageLabelshape [{}] ".format(nScale,i,j,m,n, ImageT.shape, ImageLabelT.shape))
                            
                                if( nImageChannels == 3):
                                    plt.imshow(cv2.cvtColor(ImageT, cv2.COLOR_BGR2RGB))
                                else:
                                    plt.imshow(ImageT)
                                    
                                plt.show()    
        
                                plt.imshow(ImageLabelT)
                                plt.show()  

                                sDirectoryForOutputImages ='../temp'
                                #sDirectoryForOutputImages =m_sDirectoryForInputImagesAugment
                                
                                #sOutfileName='Sc{}-Fl{}-Ro{}-Trr{}-Trc{}-Real.jpg'.format(nScale,i,j,m,n)
                                #cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), imageR )                
                                sOutfileName='Sc{}-Fl{}-Ro{}-Trr{}-Trc{}-Image.jpg'.format(nScale,i,j,m,n)
                                cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), ImageT )
                                
                                sOutfileName='Sc{}-Fl{}-Ro{}-Trr{}-Trc{}-Label.jpg'.format(nScale,i,j,m,n)                                
                                cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), ImageLabelT )                
                    
                            
                    
                    
    #return DataX, DataLabelX, nNbrOfImages
    return DataX, DataLabelX, nNbrOfData



# receive a Image and the corresponding label image, create augmented images and label images, and return the images in numpy arrays
#This is for Scale, Flip, and Rotation
def GetAugmentedImagesRegularTwoInputsS1F2R5( aImage, aImageLabel ):

    bPrintOut = m_bPrintOutResult
    
    '''
    nNbrOfScales = 3    #(1.0,1.15, 1.30)
    nNbrOfFlips  = 2    #(original, flipX)
    nNbrOfRotations = 12 #( 360/30 =12)
    nNbrOfImages = nNbrOfScales*nNbrOfFlips*nNbrOfRotations
    '''
    
    nNbrOfImages = GetNumberOfAugmentedImagesRegular()

    nRows = aImage.shape[0]
    nCols = aImage.shape[1] 
    nImageChannels = len(aImage.shape)   
    nImageLabelChannels = len(aImageLabel.shape)   

    
    if( nImageChannels == 3):
        DataX = np.ndarray((nNbrOfImages, nRows, nCols, nImageChannels), dtype=np.uint8)
    else:
        DataX = np.ndarray((nNbrOfImages, nRows, nCols), dtype=np.uint8)
        
    if( nImageLabelChannels == 3):
        DataLabelX = np.ndarray((nNbrOfImages, nRows, nCols, nImageLabelChannels), dtype=np.uint8)
    else:
        DataLabelX = np.ndarray((nNbrOfImages, nRows, nCols), dtype=np.uint8)

    if( nImageChannels == 3 ):
        nImageBlue  = int( (int(aImage[1,1,0]) + int(aImage[nRows-2,1,0]) + int(aImage[1,nCols-2,0]) + int(aImage[nRows-2,nCols-2,0]))/4)
        nImageGreen = int( (int(aImage[1,1,1]) + int(aImage[nRows-2,1,1]) + int(aImage[1,nCols-2,1]) + int(aImage[nRows-2,nCols-2,1]))/4)
        nImageRed   = int( (int(aImage[1,1,2]) + int(aImage[nRows-2,1,2]) + int(aImage[1,nCols-2,2]) + int(aImage[nRows-2,nCols-2,2]))/4)
                
        ImageM = np.zeros( (nRows*3, nCols*3,3), np.uint8)
        ImageM[:] = (nImageBlue,nImageGreen, nImageRed)
    else: 
        nImageGreen = int( (int(aImage[1,1]) + int(aImage[nRows-2,1]) + int(aImage[1,nCols-2]) + int(aImage[nRows-2,nCols-2]))/4)           
        ImageM = np.zeros( (nRows*3, nCols*3), np.uint8)
        ImageM[:] = nImageGreen

    ImageLabelM = np.zeros( (nRows*3, nCols*3), np.uint8)
    ImageLabelM[:] = 0
    
    for m in range (nRows):
        for n in range (nCols):
            ImageM[int(nRows)+m, int(nCols)+n] = aImage[m,n]
            ImageLabelM[int(nRows)+m, int(nCols)+n] = aImageLabel[m,n]
                
                        
    nNbrOfData=0
    #for nScale in range (100,141,15):
    #for nScale in range (85,120,15):
    for nScale in range (m_nScaleStart, m_nScaleEnd, m_nScaleStep):
                
        #imageS = cv2.resize(ImageM, (int(ImageM.shape[0]*nScale*0.01),int(ImageM.shape[1]*nScale*0.01)))
        #imageLabelS = cv2.resize(ImageLabelM, (int(ImageLabelM.shape[0]*nScale*0.01),int(ImageLabelM.shape[1]*nScale*0.01)))                                 
        imageS = cv2.resize(ImageM, (int(ImageM.shape[1]*nScale*0.01),int(ImageM.shape[0]*nScale*0.01)))
        imageLabelS = cv2.resize(ImageLabelM, (int(ImageLabelM.shape[1]*nScale*0.01),int(ImageLabelM.shape[0]*nScale*0.01)))

        #for i in range(0,2): #flip x, Y, or X & Y 
        for i in range(m_nFlipStart, m_nFlipEnd, m_nFlipStep): #flip x, Y, or X & Y 
                               
            if( i == 0):                
                aImageF         = imageS
                aImageLabelF    = imageLabelS
                if(bPrintOut):
                    print("===original===")
            elif( i == 1 ):                    # do not use it because same as Horizontal flip  with 180 rotation
                aImageF         = cv2.flip(imageS, 0) #vertical flip
                aImageLabelF    = cv2.flip(imageLabelS, 0) #vertical flip
                if(bPrintOut):
                    print("===Flip Vertical===")
            elif( i == 2 ):
                aImageF         = cv2.flip(imageS, 1) #Horizonal flip
                aImageLabelF    = cv2.flip(imageLabelS, 1) #Horizonal flip
                if(bPrintOut):
                    print("===Flip Horizontal===")
            elif( i == 3 ):                     # do not use it because same as original image with 180 rotation
                aImageF         = cv2.flip(imageS, -1) #Both flip
                aImageLabelF    = cv2.flip(imageLabelS, -1) #Both flip
                if(bPrintOut):
                    print("===Flip Horizontal and Vertical===")
                
                                    
            #for j in range (0,360,15):
            for j in range (m_nRotationStart, m_nRotationEnd, m_nRotationStep):
                if( j < m_nExcludeRange or j > 360-m_nExcludeRange ):
                
                    #M = cv2.getRotationMatrix2D((int(aImageF.shape[0]/2), int(aImageF.shape[1]/2)),j,1)
                    #imageR      = cv2.warpAffine(aImageF,M,(aImageF.shape[0],aImageF.shape[1]))
                    #imageLabelR = cv2.warpAffine(aImageLabelF,M,(aImageLabelF.shape[0],aImageLabelF.shape[1]))
                    M = cv2.getRotationMatrix2D((int(aImageF.shape[1]/2), int(aImageF.shape[0]/2)),j,1)
                    imageR      = cv2.warpAffine(aImageF,M,(aImageF.shape[1],aImageF.shape[0]))
                    imageLabelR = cv2.warpAffine(aImageLabelF,M,(aImageLabelF.shape[1],aImageLabelF.shape[0]))
                                                    
                    ImageF      = imageR[int(imageR.shape[0]/2-nRows/2):int(imageR.shape[0]/2+nRows/2),int(imageR.shape[1]/2-nCols/2):int(imageR.shape[1]/2+nCols/2) ]
                    ImageLabelF = imageLabelR[int(imageLabelR.shape[0]/2-nRows/2):int(imageLabelR.shape[0]/2+nRows/2),int(imageLabelR.shape[1]/2-nCols/2):int(imageLabelR.shape[1]/2+nCols/2) ]
                          
                    DataX[nNbrOfData] = np.array([ImageF])  
                    DataLabelX[nNbrOfData] = np.array([ImageLabelF])  
                    
                    nNbrOfData+=1
                            
                    if(bPrintOut):
                        print("[{}] =Scale=[{}] Flip=[{}] Rotate [{}]  Imageshape [{}] ImageLabelshape [{}] ".format(nNbrOfData, nScale,i,j, ImageF.shape, ImageLabelF.shape))
                    
                        if( nImageChannels == 3):
                            plt.imshow(cv2.cvtColor(ImageF, cv2.COLOR_BGR2RGB))
                        else:
                            plt.imshow(ImageF)
                            
                        plt.show()  
                        
                        plt.imshow(ImageLabelF)
                        plt.show()  
                    
                    
    #return DataX, DataLabelX, nNbrOfImages
    return DataX, DataLabelX, nNbrOfData


# 20190502 added: To remove white boundary points
def AdjustBoundaryPixelValuesForAugmentation( aImage, nBoundaryRangeToAdjust ):

    nRows = aImage.shape[0]
    nCols = aImage.shape[1] 
    nImageChannels = len(aImage.shape)    

    OutImage = aImage.copy()

    if( nBoundaryRangeToAdjust > 0 ):
        
        if( nImageChannels == 3 ):
            nImageBlue  = int( (int(aImage[nBoundaryRangeToAdjust,nBoundaryRangeToAdjust,0]) + int(aImage[nRows-nBoundaryRangeToAdjust,nBoundaryRangeToAdjust,0]) + int(aImage[nBoundaryRangeToAdjust,nCols-nBoundaryRangeToAdjust,0]) + int(aImage[nRows-nBoundaryRangeToAdjust,nCols-nBoundaryRangeToAdjust,0]))/4)
            nImageGreen = int( (int(aImage[nBoundaryRangeToAdjust,nBoundaryRangeToAdjust,1]) + int(aImage[nRows-nBoundaryRangeToAdjust,nBoundaryRangeToAdjust,1]) + int(aImage[nBoundaryRangeToAdjust,nCols-nBoundaryRangeToAdjust,1]) + int(aImage[nRows-nBoundaryRangeToAdjust,nCols-nBoundaryRangeToAdjust,1]))/4)
            nImageRed   = int( (int(aImage[nBoundaryRangeToAdjust,nBoundaryRangeToAdjust,2]) + int(aImage[nRows-nBoundaryRangeToAdjust,nBoundaryRangeToAdjust,2]) + int(aImage[nBoundaryRangeToAdjust,nCols-nBoundaryRangeToAdjust,2]) + int(aImage[nRows-nBoundaryRangeToAdjust,nCols-nBoundaryRangeToAdjust,2]))/4)                
        else: 
            nImageGreen = int( (int(aImage[nBoundaryRangeToAdjust,nBoundaryRangeToAdjust]) + int(aImage[nRows-nBoundaryRangeToAdjust,nBoundaryRangeToAdjust]) + int(aImage[nBoundaryRangeToAdjust,nCols-nBoundaryRangeToAdjust]) + int(aImage[nRows-nBoundaryRangeToAdjust,nCols-nBoundaryRangeToAdjust]))/4)           
           
        for i in range(nRows):
            for j in range(nCols):
                if( i < nBoundaryRangeToAdjust or i > nRows-nBoundaryRangeToAdjust or j < nBoundaryRangeToAdjust or j > nCols-nBoundaryRangeToAdjust ):
                    if( nImageChannels == 3 ):
                        OutImage[i,j] = (nImageBlue,nImageGreen,nImageRed)
                    else:
                        OutImage[i,j] = nImageGreen
                    
    return OutImage



def GetCropNumber(nImageSize, nCropSize, nImageStrideToCrop ):

    nCount=0
    for i in range(0, nImageSize, nImageStrideToCrop):
        nLength = i + nCropSize-1
        if( nLength < nImageSize-1 ):
            nCount+=1
        elif( nLength >= nImageSize-1 ):
            nCount+=1
            break
        else:
            break
        
    return nCount

        

# receive a Image, create augmented images, and return the images in numpy array
def GetCropImagesRegularOneInput( aImage, cfg, bPrintOutResult ):

    bPrintOutResult= False
            
    if( len(aImage.shape) == 3):
        nChannels       = 3 
    else:
        nChannels       = 1 
    
    
    nNbrForRow      = GetCropNumber(aImage.shape[0], cfg.nImageRowsToCrop, cfg.nImageStrideToCrop )
    nNbrForColumn   = GetCropNumber(aImage.shape[1], cfg.nImageColumnsToCrop, cfg.nImageStrideToCrop )        
    nNbrOfImages    = nNbrForRow*nNbrForColumn
   
        
    if( nChannels == 3):
        DataX = np.ndarray((nNbrOfImages, cfg.nImageRowsToCrop, cfg.nImageColumnsToCrop, nChannels), dtype=np.uint8)
    else:
        DataX = np.ndarray((nNbrOfImages, cfg.nImageRowsToCrop, cfg.nImageColumnsToCrop), dtype=np.uint8)
        
    nNbrOfData=0  
    bCropRow=True
    for i in range(0, aImage.shape[0], cfg.nImageStrideToCrop):

        nRowStart = i 
        nRowEnd   = i+cfg.nImageRowsToCrop
            
        if( nRowEnd >= aImage.shape[0]-1):
            nRowStart = aImage.shape[0]-1-cfg.nImageRowsToCrop
            nRowEnd   = aImage.shape[0]-1
            bCropRow = False        

        bCropCol=True              
        for j in range(0, aImage.shape[1], cfg.nImageStrideToCrop):
                               
            nColumnStart = j
            nColumnEnd   = j+cfg.nImageColumnsToCrop
            
            if( nColumnEnd > aImage.shape[1]-1):
                nColumnStart = aImage.shape[1]-1-cfg.nImageColumnsToCrop
                nColumnEnd   = aImage.shape[1]-1
                bCropCol = False
                        
            aImageCrop = aImage[nRowStart:nRowEnd,nColumnStart:nColumnEnd ]
                          
            DataX[nNbrOfData] = np.array([aImageCrop])  
            

            nNbrOfData+=1           
            
                                                
            if(bPrintOutResult):
                print("=Imageshape[{},{}] = [{}] ".format(i, j, aImageCrop.shape))
                    
                if( nChannels == 3):
                    plt.imshow(cv2.cvtColor(aImageCrop, cv2.COLOR_BGR2RGB))
                else:
                    plt.imshow(aImageCrop)
                            
                plt.show()    
                
                sDirectoryForOutputImages ='./temp/Crop'
                        
                sOutfileName='Crop-Row{}-Col{}.jpg'.format(i,j)
                cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), aImageCrop )      
                
            if(bCropCol == False ):
                break

        if(bCropRow == False and bCropCol == False ):
            break

                    
    return DataX, nNbrOfImages


#Counting nNbrOfImages was wrong
# receive a Image, create augmented images, and return the images in numpy array
def GetCropImagesRegularOneInput_Old( aImage, cfg, bPrintOutResult ):

    bPrintOutResult= False
            
    if( len(aImage.shape) == 3):
        nChannels       = 3 
    else:
        nChannels       = 1 
    
    nNbrForRow      = int(float(aImage.shape[0])/float(cfg.nImageStrideToCrop) + 0.99999)     
    nNbrForColumn   = int(float(aImage.shape[1])/float(cfg.nImageStrideToCrop) + 0.99999)  
    nNbrOfImages    = nNbrForRow*nNbrForColumn
    
        
    if( nChannels == 3):
        DataX = np.ndarray((nNbrOfImages, cfg.nImageRowsToCrop, cfg.nImageColumnsToCrop, nChannels), dtype=np.uint8)
    else:
        DataX = np.ndarray((nNbrOfImages, cfg.nImageRowsToCrop, cfg.nImageColumnsToCrop), dtype=np.uint8)
        
    nNbrOfData=0        
    for i in range(0, aImage.shape[0], cfg.nImageStrideToCrop):
        for j in range(0, aImage.shape[1], cfg.nImageStrideToCrop):
            nRowStart = i 
            nRowEnd   = i+cfg.nImageRowsToCrop
            
            if( nRowEnd > aImage.shape[0]-1):
                nRowStart = aImage.shape[0]-1-cfg.nImageRowsToCrop
                nRowEnd   = aImage.shape[0]-1
                                
            nColumnStart = j
            nColumnEnd   = j+cfg.nImageColumnsToCrop
            
            if( nColumnEnd > aImage.shape[1]-1):
                nColumnStart = aImage.shape[1]-1-cfg.nImageColumnsToCrop
                nColumnEnd   = aImage.shape[1]-1
                        
            aImageCrop = aImage[nRowStart:nRowEnd,nColumnStart:nColumnEnd ]
                          
            DataX[nNbrOfData] = np.array([aImageCrop])  
            

            nNbrOfData+=1           
            
                                                
            if(bPrintOutResult):
                print("=Imageshape[{},{}] = [{}] ".format(i, j, aImageCrop.shape))
                    
                if( nChannels == 3):
                    plt.imshow(cv2.cvtColor(aImageCrop, cv2.COLOR_BGR2RGB))
                else:
                    plt.imshow(aImageCrop)
                            
                plt.show()    
                
                sDirectoryForOutputImages ='./temp/Crop'
                        
                sOutfileName='Crop-Row{}-Col{}.jpg'.format(i,j)
                cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), aImageCrop )                
                    
    return DataX, nNbrOfImages



def AugmentAImagesRegular1():
    
    
   
    sListForTestDataFileNames=[]
    oFileOpen  = open(m_sFileForInputImages, 'r') 
    for aline in oFileOpen:
        aImagePath, aImageName = os.path.split(aline)
        sImageName = str(aImageName)            
        if sImageName.find('.tif') > 0 or sImageName.find('jpg') > 0 or sImageName.find('.png') > 0  or sImageName.find('.bmp') > 0:                
            sLine = (str)(aline)
            sLine = sLine.strip('\r\n')
            sLine = sLine.lstrip()
            sLine = sLine.rstrip()
            sListForTestDataFileNames.append(sLine)       
    oFileOpen.close()
  
   
 
    nCount=0
    for aFileName in sListForTestDataFileNames[:]:
        aImagePath, aImageName = os.path.split(aFileName)

        sImagePath = str(aImagePath)
        sImagePath = sImagePath.strip('\r\n')
        sImagePath = sImagePath.lstrip()
        sImagePath = sImagePath.rstrip()

        sImageName = str(aImageName)
        sImageName = sImageName.strip('\r\n')
        sImageName = sImageName.lstrip()
        sImageName = sImageName.rstrip()    
    
            
        if sImageName.find('.tif') > 0 or sImageName.find('jpg') > 0 or sImageName.find('.png') > 0  or sImageName.find('.bmp') > 0:    
    
            aImage      = cv2.imread(os.path.join(aImagePath, aImageName), cv2.IMREAD_COLOR) 
            nRows = aImage.shape[0]
            nCols = aImage.shape[1]
            nImageBlue  = int( (int(aImage[1,1,0]) + int(aImage[nRows-2,1,0]) + int(aImage[1,nCols-2,0]) + int(aImage[nRows-2,nCols-2,0]))/4)
            nImageGreen = int( (int(aImage[1,1,1]) + int(aImage[nRows-2,1,1]) + int(aImage[1,nCols-2,1]) + int(aImage[nRows-2,nCols-2,1]))/4)
            nImageRed   = int( (int(aImage[1,1,2]) + int(aImage[nRows-2,1,2]) + int(aImage[1,nCols-2,2]) + int(aImage[nRows-2,nCols-2,2]))/4)
            
            print("=aImage [{}]".format(aImage.shape))
            print("=Average Edge pixel [{},{},{}]".format(nImageBlue,nImageGreen,nImageRed))
            

            ImageM = np.zeros( (nRows*3, nCols*3,3), np.uint8)
            ImageM[:] = (nImageBlue,nImageGreen, nImageRed)
                
            for m in range (nRows):
                for n in range (nCols):
                    ImageM[int(nRows)+m, int(nCols)+n] = aImage[m,n]
                
            print("=Large Image={}]".format(ImageM.shape))
            plt.imshow(cv2.cvtColor(ImageM, cv2.COLOR_BGR2RGB))
            plt.show()                 
            
            
            #for nScale in range (80,150,20):
            for nScale in range (m_nScaleStart, m_nScaleEnd, m_nScaleStep):
                
                #nRowsScale = int(nRows*i*0.01)
                #nColsScale = int(nCols*i*0.01)
                imageS = cv2.resize(ImageM, (int(ImageM.shape[0]*nScale*0.01),int(ImageM.shape[1]*nScale*0.01)))
                         
                '''
                if( nScale > 100):
                    ImageSF = imageS[int(ImageSR/2-nRows/2):int(ImageSR/2+nRows/2),int(imageSC/2-nCols/2):int(imageSC/2+nCols/2) ]

                    #imageS = cv2.warpAffine(aImage,np.array([[i*0.1,0,-nCols*(i)/2],[0,i*0.1,-nRows*(i)/2]],dtype=np.float32), (nRows, nCols))

                elif( i == 100):
                    ImageSF = aImage
                else:
                    ImageSF = np.zeros( aImage.shape, np.uint8)
                    ImageSF[:] = (nImageBlue,nImageGreen, nImageRed)
                    #ImageSF[int(ImageSR/2-nRows/2):int(ImageSR/2+nRows/2-1),int(imageSC/2-nCols/2):int(imageSC/2+nCols/2-1) ] = imageS                   
                    #ImageSF[int(ImageSR/2-nRows/2):, int(imageSC/2-nCols/2):,: ] = imageS    
                    
                    for m in range(imageS.shape[0]):
                        for n in range (imageS.shape[1]):
                            ImageSF[int(nRows/2-ImageSR/2)+m,int(nCols/2-imageSC/2)+n ] = imageS[m,n]
                    
                '''


                #print("=Scale [{}]  imageSF-shape[{}]".format(nScale*0.01, ImageS.shape))
                #plt.imshow(cv2.cvtColor(ImageSF, cv2.COLOR_BGR2RGB))
                #plt.show()                 
                       
         
                #for i in range(0,2): #flip x, Y, or X & Y 
                for i in range(m_nFlipStart, m_nFlipEnd, m_nFlipStep): #flip x, Y, or X & Y 
                               
                    if( i == 0):                
                        aImageF = imageS
                        print("===original===")
                    elif( i == 1 ):
                        aImageF = cv2.flip(imageS, 0) #Horizonal flip
                        print("===Flip Horizontal===")
                    elif( i == 2 ):                    # do not use it because same as Horizontal flip  with 180 rotation
                        aImageF = cv2.flip(imageS, 1) #vertical flip
                        print("===Flip Vertical===")
                    elif( i == 3 ):                     # do not use it because same as original image with 180 rotation
                        aImageF = cv2.flip(imageS, -1) #Both flip
                        print("===Flip Horizontal and Vertical===")
                
                
                
                    #plt.imshow(cv2.cvtColor(aImageR, cv2.COLOR_BGR2RGB))
                    #plt.show() 
                    
                    #for j in range (0,360,30):
                    for j in range (m_nRotationStart, m_nRotationEnd, m_nRotationStep):
                        M = cv2.getRotationMatrix2D((int(aImageF.shape[0]/2), int(aImageF.shape[1]/2)),j,1)
                        imageR = cv2.warpAffine(aImageF,M,(aImageF.shape[0],aImageF.shape[1]))
                        
                        #imageR = cv2.warpAffine(aImageF,M,(nCols,nRows))
                        
                        '''
                        for m in range (nRows):
                            for n in range (nCols):
                                if( rotated[m,n,0] < 10 and rotated[m,n,1] <= 10 and rotated[m,n,2] <= 10):
                                    rotated[m,n] = (nImageBlue,nImageGreen, nImageRed)
                        '''
                        
                        ImageF = imageR[int(imageR.shape[0]/2-nRows/2):int(imageR.shape[0]/2+nRows/2),int(imageR.shape[1]/2-nCols/2):int(imageR.shape[1]/2+nCols/2) ]
                        
                        
                        small = np.amin(ImageF)
                        biggest = np.amax(ImageF)
                        
                        print("=Scale=[{}] Flip=[{}] Rotate [{}]  shape [{}] = [min={}, max={}]".format(nScale,i,j, ImageF.shape, small, biggest))
                        plt.imshow(cv2.cvtColor(ImageF, cv2.COLOR_BGR2RGB))
                        plt.show()                 
                    

                   
                        #break
            nCount+=1
            if( nCount > 0 ):
                break
        






def GetAugmentedImageAndLabelRegular( aImage, aImageLabel ):

    '''
    nNbrOfScales = 3
    nNbrOfFlips  = 2
    nNbrOfRotations = 12
    nNbrOfImages = nNbrOfScales*nNbrOfFlips*nNbrOfRotations
    '''
    
    nNbrOfImages = GetNumberOfAugmentedImagesRegular()

    
    nRows = aImage.shape[0]
    nCols = aImage.shape[1]    
    
    if( nImageChannels == 3):
        DataX = np.ndarray((nNbrOfImages, nRows, nCols, nImageChannels), dtype=np.uint8)
    else:
        DataX = np.ndarray((nNbrOfImages, nRows, nCols), dtype=np.uint8)
        
    DataXLabel = np.ndarray((nNbrOfImages, nRows, nCols), dtype=np.uint8) 


    nImageBlue  = int( (int(aImage[1,1,0]) + int(aImage[nRows-2,1,0]) + int(aImage[1,nCols-2,0]) + int(aImage[nRows-2,nCols-2,0]))/4)
    nImageGreen = int( (int(aImage[1,1,1]) + int(aImage[nRows-2,1,1]) + int(aImage[1,nCols-2,1]) + int(aImage[nRows-2,nCols-2,1]))/4)
    nImageRed   = int( (int(aImage[1,1,2]) + int(aImage[nRows-2,1,2]) + int(aImage[1,nCols-2,2]) + int(aImage[nRows-2,nCols-2,2]))/4)
            
    ImageM = np.zeros( (nRows*3, nCols*3,3), np.uint8)
    ImageM[:] = (nImageBlue,nImageGreen, nImageRed)
                
    ImageLabelM = np.zeros( (nRows*3, nCols*3), np.uint8)
    ImageLabelM[:] = 0

    
    for m in range (nRows):
        for n in range (nCols):
            ImageM[int(nRows)+m, int(nCols)+n] = aImage[m,n]
            ImageLabelM[int(nRows)+m, int(nCols)+n] = aImage[m,n]
                
    print("=Large Image={}]".format(ImageM.shape))
    plt.imshow(cv2.cvtColor(ImageM, cv2.COLOR_BGR2RGB))
    plt.show()                 
            
            
    nNbrOfData=0
    #for nScale in range (100,140,15):
    for nScale in range (m_nScaleStart, m_nScaleEnd, m_nScaleStep):
                
        imageS = cv2.resize(ImageM, (int(ImageM.shape[0]*nScale*0.01),int(ImageM.shape[1]*nScale*0.01)))
        imageLabelS = cv2.resize(ImageLabelM, (int(ImageLabelM.shape[0]*nScale*0.01),int(ImageLabelM.shape[1]*nScale*0.01)))
                                 
        #for i in range(0,2): #flip x, Y, or X & Y 
        for i in range(m_nFlipStart, m_nFlipEnd, m_nFlipStep): #flip x, Y, or X & Y 
                               
            if( i == 0):                
                aImageF = imageS
                aImageLabelF = imageLabelS
                print("===original===")
            elif( i == 1 ):
                aImageF = cv2.flip(imageS, 0) #Horizonal flip
                aImageLabelF = cv2.flip(imageLabelS, 0) #Horizonal flip                
                print("===Flip Horizontal===")
            elif( i == 2 ):                    # do not use it because same as Horizontal flip  with 180 rotation
                aImageF = cv2.flip(imageS, 1) #vertical flip
                aImageLabelF = cv2.flip(imageLabelS, 1) #vertical flip
                print("===Flip Vertical===")
            elif( i == 3 ):                     # do not use it because same as original image with 180 rotation
                aImageF = cv2.flip(imageS, -1) #Both flip
                aImageLabelF = cv2.flip(imageLabelS, -1) #Both flip
                print("===Flip Horizontal and Vertical===")
                
                                    
            #for j in range (0,360,30):
            for j in range (m_nRotationStart, m_nRotationEnd, m_nRotationStep):
                M = cv2.getRotationMatrix2D((int(aImageF.shape[0]/2), int(aImageF.shape[1]/2)),j,1)
                imageR = cv2.warpAffine(aImageF,M,(aImageF.shape[0],aImageF.shape[1]))
                imageLabelR = cv2.warpAffine(aImageLabelF,M,(aImageLabelF.shape[0],aImageLabelF.shape[1]))
                                                
                ImageF = imageR[int(imageR.shape[0]/2-nRows/2):int(imageR.shape[0]/2+nRows/2),int(imageR.shape[1]/2-nCols/2):int(imageR.shape[1]/2+nCols/2) ]
                ImageLabelF = imageLabelR[int(imageLabelR.shape[0]/2-nRows/2):int(imageLabelR.shape[0]/2+nRows/2),int(imageLabelR.shape[1]/2-nCols/2):int(imageLabelR.shape[1]/2+nCols/2) ]
                      
                DataX[nNbrOfData] = np.array([ImageF])            
                DataXLabel[nNbrOfData] = np.array([ImageLabelF])                         
                        
                print("=Scale=[{}] Flip=[{}] Rotate [{}]  Imageshape [{}], ImageLabelShape [{}]".format(nScale,i,j, ImageF.shape, ImageLabelF.shape))
                plt.imshow(cv2.cvtColor(ImageF, cv2.COLOR_BGR2RGB))
                plt.show()                 
                    
    return DataX, DataXLabel, nNbrOfData




#This create black boundaries after rotation
def AugmentAImagesRegular():
    
    
   
    sListForTestDataFileNames=[]
    oFileOpen  = open(m_sFileForInputImages, 'r') 
    for aline in oFileOpen:
        aImagePath, aImageName = os.path.split(aline)
        sImageName = str(aImageName)            
        if sImageName.find('.tif') > 0 or sImageName.find('jpg') > 0 or sImageName.find('.png') > 0  or sImageName.find('.bmp') > 0:                
            sLine = (str)(aline)
            sLine = sLine.strip('\r\n')
            sLine = sLine.lstrip()
            sLine = sLine.rstrip()
            sListForTestDataFileNames.append(sLine)       
    oFileOpen.close()
  
   
 
    nCount=0
    for aFileName in sListForTestDataFileNames[:]:
        aImagePath, aImageName = os.path.split(aFileName)

        sImagePath = str(aImagePath)
        sImagePath = sImagePath.strip('\r\n')
        sImagePath = sImagePath.lstrip()
        sImagePath = sImagePath.rstrip()

        sImageName = str(aImageName)
        sImageName = sImageName.strip('\r\n')
        sImageName = sImageName.lstrip()
        sImageName = sImageName.rstrip()    
    
            
        if sImageName.find('.tif') > 0 or sImageName.find('jpg') > 0 or sImageName.find('.png') > 0  or sImageName.find('.bmp') > 0:    
    
            aImage      = cv2.imread(os.path.join(aImagePath, aImageName), cv2.IMREAD_COLOR) 
            nRows = aImage.shape[0]
            nCols = aImage.shape[1]
            nImageBlue  = int( (int(aImage[1,1,0]) + int(aImage[nRows-2,1,0]) + int(aImage[1,nCols-2,0]) + int(aImage[nRows-2,nCols-2,0]))/4)
            nImageGreen = int( (int(aImage[1,1,1]) + int(aImage[nRows-2,1,1]) + int(aImage[1,nCols-2,1]) + int(aImage[nRows-2,nCols-2,1]))/4)
            nImageRed   = int( (int(aImage[1,1,2]) + int(aImage[nRows-2,1,2]) + int(aImage[1,nCols-2,2]) + int(aImage[nRows-2,nCols-2,2]))/4)
            
            print("=aImage [{}]".format(aImage.shape))
            print("=Average Edge pixel [{},{},{}]".format(nImageBlue,nImageGreen,nImageRed))
            
                        
            #for nScale in range (80,150,20):
            for nScale in range (m_nScaleStart, m_nScaleEnd, m_nScaleStep):
                
                aImageS = cv2.resize(aImage, (int(nRows*nScale*0.01),int(nCols*nScale*0.01)))
                                
                if( nScale >= 100):
                    ImageS = aImageS[int(aImageS.shape[0]/2-nRows/2):int(aImageS.shape[0]/2+nRows/2),int(aImageS.shape[1]/2-nCols/2):int(aImageS.shape[1]/2+nCols/2) ]
                elif( nScale == 100):
                    ImageS = aImage
                else:
                    ImageS = np.zeros( aImage.shape, np.uint8)
                    ImageS[:] = (nImageBlue,nImageGreen, nImageRed)
                    
                    for m in range(aImageS.shape[0]):
                        for n in range (aImageS.shape[1]):
                            ImageS[int(nRows/2-aImageS.shape[0]/2)+m,int(nCols/2-aImageS.shape[1]/2)+n ] = aImageS[m,n]
                    
                    
                #print("=Scale [{}]  imageS-shape[{}]  imageSF-shape[{}]".format(i*0.1, imageS.shape, ImageSF.shape))
                #plt.imshow(cv2.cvtColor(ImageSF, cv2.COLOR_BGR2RGB))
                #plt.show()                 
                       
         
                #for i in range(0,2): #flip x, Y, or X & Y 
                for i in range(m_nFlipStart, m_nFlipEnd, m_nFlipStep): #flip x, Y, or X & Y 
                               
                    if( i == 0):                
                        aImageF = ImageS
                        print("===original===")
                    elif( i == 1 ):
                        aImageF = cv2.flip(ImageS, 0) #Horizonal flip
                        print("===Flip Horizontal===")
                    elif( i == 2 ):                    # do not use it because same as Horizontal flip  with 180 rotation
                        aImageF = cv2.flip(ImageS, 1) #vertical flip
                        print("===Flip Vertical===")
                    elif( i == 3 ):                     # do not use it because same as original image with 180 rotation
                        aImageF = cv2.flip(ImageS, -1) #Both flip
                        print("===Flip Horizontal and Vertical===")
                
                    #plt.imshow(cv2.cvtColor(aImageR, cv2.COLOR_BGR2RGB))
                    #plt.show() 
                    
                    #for j in range (0,360,30):
                    for j in range (m_nRotationStart, m_nRotationEnd, m_nRotationStep):
                        M = cv2.getRotationMatrix2D((int(aImageF.shape[0]/2),int(aImageF.shape[1]/2)),j,1)
                        ImageR = cv2.warpAffine(aImageF,M,(aImageF.shape[0],aImageF.shape[1]))
                        
                        for m in range (ImageR.shape[1]):
                            for n in range (ImageR.shape[1]):
                                if( ImageR[m,n,0] < 10 and ImageR[m,n,1] <= 10 and ImageR[m,n,2] <= 10):
                                    ImageR[m,n] = (nImageBlue,nImageGreen, nImageRed)
                        
                        
                        small = np.amin(ImageR)
                        biggest = np.amax(ImageR)
                        
                        print("=Scale=[{}] Flip=[{}] Rotate [{}]  shape [{}] = [min={}, max={}]".format(nScale,i,j, ImageR.shape, small, biggest))
                        plt.imshow(cv2.cvtColor(ImageR, cv2.COLOR_BGR2RGB))
                        plt.show()                 
                    
 
                   
                        #break
            nCount+=1
            if( nCount > 0 ):
                break
        



if __name__ == "__main__":
    
    nMode = 100 #For augumenting a image and the corresponding label image together
    #nMode = 200 #For augumenting a image
        
    m_nIndexScale, m_nIndexFlip, m_nIndexRotation, m_nIndexTranslation = SetAugmentation( m_bScaleUsage, m_bFlipUsage, m_bRotationUsage, m_bTranslationUsage )
    
    if( nMode == 100):
        AugmentTwoImagesRegularVariations() 
    else:
        #SetAugmentation( bScale=True, bFlip=True, bRotation=True, bTranslation=True )
        AugmentImagesRegularVariations()
        
    #AugmentAImagesRegular1()
    #AugmentAImagesRegular()


