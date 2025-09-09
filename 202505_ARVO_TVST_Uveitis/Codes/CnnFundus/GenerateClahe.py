# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 11:12:40 2018

@author: jongkim
"""


import os
import  cv2
import matplotlib.pyplot as plt
import numpy as np
import shutil

#from MySkeleton import Skeletionization


#sDirectoryForInputImages  = 'R:\MyIVFA\All Full Frame Leakage\All Images\ATest2'
#sDirectoryForOutputImages = 'R:\MyIVFA\All Full Frame Leakage\All Images\ATest2\Results2'


#for ceb-na\ceb-retinapartition
#sDirectoryForInputImages  = r'R:\GoogleDrive\New-Leakage-Images-20190523\FA2-Images'
#sDirectoryForOutputImages = r'R:\GoogleDrive\New-Leakage-Images-20190523\FA2-Images-Clahe'

#sDirectoryForInputImages  = r'R:\GoogleDrive\FAsLeAnne2020\Annotated100\ImagesForAll'
#sDirectoryForOutputImages = r'R:\GoogleDrive\FAsLeAnne2020\Annotated100\ImagesForAllClahe'


#sDirectoryForInputImages  = r"R:\GoogleDrive\FAsLeAnne2020\Annotated100\ImagesForAll"
#sDirectoryForOutputImages = r"R:\GoogleDrive\FAsLeAnne2020\Annotated100\ImagesForAllClahe"

#sDirectoryForInputImages  = r'N:\\lhcdevfiler\\retina2\\uveitis\\GoogleDrive\\FAsLeAnne2020\\Annotated100\\ImagesForAll"
#sDirectoryForOutputImages = r"N\\lhcdevfiler\\retina2\\uveitis\\GoogleDrive\\FAsLeAnne2020\\Annotated100\\ImagesForAllClahe"

#sDirectoryForInputImages  = r"N:\uveitis\GoogleDrive\FAsLeAnne2020\Annotated100\ImagesForAll"
#sDirectoryForOutputImages = r"N:\uveitis\GoogleDrive\FAsLeAnne2020\Annotated100\ImagesForAllClahe"

m_sDirectoryForInputImages  =   r"Y:\uveitis\Dataset\FAsLeAnne2020\AnnotatedAll\ImagesForAll"
m_sDirectoryForOutputImages =   r"Y:\uveitis\Dataset\FAsLeAnne2020\AnnotatedAll\ImagesAfterClahe\ImagesForAllClahe"
m_sDirectoryForOutputImagesHE = r"Y:\uveitis\Dataset\FAsLeAnne2020\AnnotatedAll\ImagesAfterClahe\ImagesForAllClaheHE"

m_nTileSizeForClahe = 10


# for Mohmet Yakin's data
m_sDirectoryForInputImages  =   r"Y:\uveitis\Dataset\FA-MehmetYakin2020\ImagesTif"
m_sDirectoryForOutputImages =   r"Y:\uveitis\Dataset\FA-MehmetYakin2020\ImagesTifClahe"
m_sDirectoryForOutputImagesOtsuBoundary =   r"Y:\uveitis\Dataset\FA-MehmetYakin2020\ImagesClaheOtsuBoundary"
m_sDirectoryForOutputImagesOtsuBinary   =   r"Y:\uveitis\Dataset\FA-MehmetYakin2020\ImagesClaheOtsuBianry"


# for Mohmet Yakin's data
m_sDirectoryForInputImages  =   r"Y:\uveitis\Dataset\FA-MehmetYakin2020\ImagesTif"
m_sDirectoryForOutputImages =   r"Y:\uveitis\Dataset\FA-MehmetYakin2020\ImagesTifClahe"
m_sDirectoryForOutputImagesOtsuBoundary =   r"Y:\uveitis\Dataset\FA-MehmetYakin2020\ImagesClaheOtsuBoundary"
m_sDirectoryForOutputImagesOtsuBinary   =   r"Y:\uveitis\Dataset\FA-MehmetYakin2020\ImagesClaheOtsuBianry"


# for Shilpa Kodati's data :Use Clahe to generate Binary Image for retinal area only
m_sDirectoryForInputImages  =   r"/retina/ARED/Uveitis/Fundus/Images/All"
m_sDirectoryForOutputImages =   r"/retina/ARED/Uveitis/Fundus/Images/AllClahe"
m_sDirectoryForOutputImagesOtsuBoundary =   r"/retina/ARED/Uveitis/Fundus/Images/AllClaheOtsuBoundary"
m_sDirectoryForOutputImagesOtsuBinary   =   r"/retina/ARED/Uveitis/Fundus/Images/AllClaheClaheOtsuBianry"



# for Shilpa Kodati's data :Use Clahe to generate Binary Image for retinal area only and Generate ROI images
m_sDirectoryForInputImages                       =   r"/retina/ARED/Uveitis/Fundus/Images/All"
m_sDirectoryForOutputImages                      =   r"/retina/ARED/Uveitis/Fundus/Images/AllClahe"
m_sDirectoryForOutputImagesOtsuBinary            =   r"/retina/ARED/Uveitis/Fundus/Images/AllClaheClaheOtsuBianry"
m_sDirectoryForOutputImagesOtsuBoundary          =   r"/retina/ARED/Uveitis/Fundus/Images/AllClaheOtsuBoundary"
m_sDirectoryForOutputImagesContourBoundaryAndROI =   r"/retina/ARED/Uveitis/Fundus/Images/AllContourBoundaryAndROI"
m_sDirectoryForOutputImagesROISquareOnly         =   r"/retina/ARED/Uveitis/Fundus/Images/ROISquare"
m_sDirectoryForOutputImagesROIOnly               =   r"/retina/ARED/Uveitis/Fundus/Images/ROI"


m_sDirectoryForInputImages                       =   r"/retina/ARED/Uveitis/Fundus/Images/All"
m_sDirectoryForOutputImages                      =   r"/retina/ARED/Uveitis/Fundus/Images/AllClahe2"
m_sDirectoryForOutputImagesOtsuBinary            =   r"/retina/ARED/Uveitis/Fundus/Images/AllClaheClaheOtsuBianry2"
m_sDirectoryForOutputImagesOtsuBoundary          =   r"/retina/ARED/Uveitis/Fundus/Images/AllClaheOtsuBoundary2"
m_sDirectoryForOutputImagesContourBoundaryAndROI =   r"/retina/ARED/Uveitis/Fundus/Images/AllContourBoundaryAndROI2"
m_sDirectoryForOutputImagesROISquareOnly         =   r"/retina/ARED/Uveitis/Fundus/Images/ROISquare2"
m_sDirectoryForOutputImagesROIOnly               =   r"/retina/ARED/Uveitis/Fundus/Images/ROI2"

#m_nTileSizeForClahe = 9999 #TileSize = min(row, column) * 0.90
m_nTileSizeForClahe = 9900 #TileSize = min(row, column) * 0.99

m_nInvertImage = 0


#sDirectoryForInputImages = 'R:\FA Images From Nida 20190314'
#sDirectoryForOutputImages = 'R:\FA Images From Nida 20190314\Results'

#def LoadTestDataForSegmentation( sDirectory, nImageRows, nImageColumns, nImageChannels, sTestDataName ):

def LoadDirectorysAndEnhanceImagesUsingOpenCVAndSaveUsingHE( sDirectoryForInputImages, sDirectoryForOutputImages ):
   
    if (os.path.exists(sDirectoryForOutputImages)):
        shutil.rmtree(sDirectoryForOutputImages)
        
    os.mkdir(sDirectoryForOutputImages)  


    nNbrOfData=0
    #lImagesInClass = os.listdir(sDirectoryForInputImages)
    oImages = os.listdir(sDirectoryForInputImages)

    for aImageName in oImages:
        sImageName = str(aImageName)
        #sImageName2 = str(sImageName.split('.')[0])
            
        if sImageName.endswith('.tif') or sImageName.endswith('jpg') or sImageName.endswith('.png') or sImageName.endswith('.bmp') :                                      
                           
            InputImage = cv2.imread(os.path.join(sDirectoryForInputImages, aImageName), cv2.IMREAD_COLOR)  
                                            
            aImageClone = InputImage.copy()  
            
            aImageB = InputImage[:,:,0]
            aImageG = InputImage[:,:,1]
            aImageR = InputImage[:,:,2]
            
            #if nNbrOfData == 0 :               
            #    aImageGOut=ExtractBloodVesselImageGreyJongwoo(aImageG, sDirectoryForOutputImages, sImageName, True)

    
            aImageHistEqu = cv2.equalizeHist(aImageG.copy())
    
            DisplayAndSaveImages(sDirectoryForOutputImages, sImageName, aImageHistEqu,  '==HE Image ==','')                
            
            nNbrOfData +=1
                 
        elif sImageName.find("thumbs.db") >= 0 :
            print('Removed wrong file {0}: {1}'.format(nNbrOfData, sImageName))               
        else :
            print('Removed wrong file {0}: {1}'.format(nNbrOfData, sImageName))  
                 
                            
    print('Loading Test Data done: {0} Data'.format(nNbrOfData))    
        
    
    
    
    

def LoadDirectorysAndEnhanceImagesUsingOpenCVAndSaveUsingClahe( sDirectoryForInputImages, sDirectoryForOutputImages ):
   
    if (os.path.exists(sDirectoryForOutputImages)):
        shutil.rmtree(sDirectoryForOutputImages)
        
    os.mkdir(sDirectoryForOutputImages)  


    nNbrOfData=0
    #lImagesInClass = os.listdir(sDirectoryForInputImages)
    oImages = os.listdir(sDirectoryForInputImages)

    for aImageName in oImages:
        sImageName = str(aImageName)
        #sImageName2 = str(sImageName.split('.')[0])
            
        if sImageName.endswith('.tif') or sImageName.endswith('jpg') or sImageName.endswith('.png') or sImageName.endswith('.bmp') :                                      
                           
            InputImage = cv2.imread(os.path.join(sDirectoryForInputImages, aImageName), cv2.IMREAD_COLOR)  
                                            
            aImageClone = InputImage.copy()  
            
            aImageB = InputImage[:,:,0]
            aImageG = InputImage[:,:,1]
            aImageR = InputImage[:,:,2]
            
            #if nNbrOfData == 0 :               
            #    aImageGOut=ExtractBloodVesselImageGreyJongwoo(aImageG, sDirectoryForOutputImages, sImageName, True)

            aImageGOut=ClaheOperator(aImageG, sDirectoryForOutputImages, sImageName, False)
            
            nNbrOfData +=1
                 
        elif sImageName.find("thumbs.db") >= 0 :
            print('Removed wrong file {0}: {1}'.format(nNbrOfData, sImageName))               
        else :
            print('Removed wrong file {0}: {1}'.format(nNbrOfData, sImageName))  
                 
                            
    print('Loading Test Data done: {0} Data'.format(nNbrOfData))    
    



   

def LoadDirectorysAndEnhanceImagesUsingOpenCVAndSaveUsingClaheMorphologyBinarizationUsingOtsu( sDirectoryForInputImages, sDirectoryForOutputImages, sDirectoryForOutputImagesOtsuBinary, sDirectoryForOutputImagesOtsuBoundary, nInvertImage  ):
   
    if (os.path.exists(sDirectoryForOutputImages)):
        shutil.rmtree(sDirectoryForOutputImages)
        
    os.mkdir(sDirectoryForOutputImages)  


    if (os.path.exists(sDirectoryForOutputImagesOtsuBinary)):
        shutil.rmtree(sDirectoryForOutputImagesOtsuBinary)
        
    os.mkdir(sDirectoryForOutputImagesOtsuBinary)  
    
    
    if (os.path.exists(sDirectoryForOutputImagesOtsuBoundary)):
        shutil.rmtree(sDirectoryForOutputImagesOtsuBoundary)
        
    os.mkdir(sDirectoryForOutputImagesOtsuBoundary)      
    


    nNbrOfData=0
    #lImagesInClass = os.listdir(sDirectoryForInputImages)
    oImages = os.listdir(sDirectoryForInputImages)

    for aImageName in oImages:
        sImageName = str(aImageName)
        #sImageName2 = str(sImageName.split('.')[0])
            
        if sImageName.endswith('.tif') or sImageName.endswith('jpg') or sImageName.endswith('.png') or sImageName.endswith('.bmp') :                                      
                           
            InputImage = cv2.imread(os.path.join(sDirectoryForInputImages, aImageName), cv2.IMREAD_COLOR)  
                                            
            aImageClone = InputImage.copy()  
            
            aImageB = InputImage[:,:,0]
            aImageG = InputImage[:,:,1]
            aImageR = InputImage[:,:,2]
            
            #if nNbrOfData == 0 :               
            #    aImageGOut=ExtractBloodVesselImageGreyJongwoo(aImageG, sDirectoryForOutputImages, sImageName, True)

            aImageClaheOut=ClaheOperatorOnly(aImageG, sDirectoryForOutputImages, sImageName, False)
                        
            # Otsu's thresholding after Gaussian filtering
            blur = cv2.GaussianBlur(aImageClaheOut,(5,5),0)
            threshold, aImageOtsuOut = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)    
        
            element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
            
            nIteration = 100
            nIteration_erode = nIteration
            nIteration_dilate = nIteration
            
            for i in range (nIteration_erode):
                aImageOtsuOut = cv2.erode(aImageOtsuOut, element)

            for i in range (nIteration_dilate):
                aImageOtsuOut = cv2.dilate(aImageOtsuOut, element)            
                
            
            if( nInvertImage == 1 ):
                aImageOtsuOut = np.invert(aImageOtsuOut)            
            #BinaryMask, ContourDraw = findLargestContourColor( aImageOtsuOut.copy(), aImageClone.copy() )
            
            if (nNbrOfData < 10  ):
                DisplayAndSaveImages(sDirectoryForOutputImages, sImageName, aImageClaheOut,  '==Binary Mask Image ==','',True )                   
                DisplayAndSaveImages(sDirectoryForOutputImagesOtsuBinary, sImageName, aImageOtsuOut,  '==Binary Mask Image ==','',True )                   
            else:
                DisplayAndSaveImages(sDirectoryForOutputImages, sImageName, aImageClaheOut,  '==Binary Mask Image ==','', False )                   
                DisplayAndSaveImages(sDirectoryForOutputImagesOtsuBinary, sImageName, aImageOtsuOut,  '==Binary Mask Image ==','', False  )                   
           
            nNbrOfData +=1
            
            if ( nNbrOfData%10 == 0 ):
                print('[{}/{}] {} done'.format( nNbrOfData, len(oImages),  sImageName))
                 
        elif sImageName.find("thumbs.db") >= 0 :
            print('Removed wrong file {0}: {1}'.format(nNbrOfData, sImageName))               
        else :
            print('Removed wrong file {0}: {1}'.format(nNbrOfData, sImageName))  
                 
                            
    print('Loading Test Data done: {0} Data'.format(nNbrOfData)) 
    
  

   

def LoadDirectorysAndEnhanceImagesUsingOpenCVAndSaveUsingClaheMorphologyBinarizationUsingOtsuAndCropImages( sDirectoryForInputImages, sDirectoryForOutputImages, sDirectoryForOutputImagesOtsuBinary, sDirectoryForOutputImagesOtsuBoundaryAndROI, sDirectoryForOutputImagesROIOnly, sDirectoryForOutputImagesROISquareOnly, nInvertImage  ):
   
    #Directory for Output images after CLAHE anly    
    if (os.path.exists(sDirectoryForOutputImages)):
        shutil.rmtree(sDirectoryForOutputImages)
        
    os.mkdir(sDirectoryForOutputImages)  

    #Directory for Output images after Binaryization after Otsu thresholding anly    
    if (os.path.exists(sDirectoryForOutputImagesOtsuBinary)):
        shutil.rmtree(sDirectoryForOutputImagesOtsuBinary)
        
    os.mkdir(sDirectoryForOutputImagesOtsuBinary)  
    

    #Directory for Output images showing Otsu Binaryization boundary and ROI boundary        
    if (os.path.exists(sDirectoryForOutputImagesOtsuBoundaryAndROI)):
        shutil.rmtree(sDirectoryForOutputImagesOtsuBoundaryAndROI)
        
    os.mkdir(sDirectoryForOutputImagesOtsuBoundaryAndROI)      
    
    
    #Directory for ROI images only
    if (os.path.exists(sDirectoryForOutputImagesROIOnly)):
        shutil.rmtree(sDirectoryForOutputImagesROIOnly)
        
    os.mkdir(sDirectoryForOutputImagesROIOnly)      
        
    #Directory for ROI Square images only
    if (os.path.exists(sDirectoryForOutputImagesROISquareOnly)):
        shutil.rmtree(sDirectoryForOutputImagesROISquareOnly)
        
    os.mkdir(sDirectoryForOutputImagesROISquareOnly)          

    nNbrOfData=0
    #lImagesInClass = os.listdir(sDirectoryForInputImages)
    oImages = os.listdir(sDirectoryForInputImages)
    
    print('Total Images: {}'.format( len(oImages) ) )

    for aImageName in oImages:
        sImageName = str(aImageName)
        #sImageName2 = str(sImageName.split('.')[0])
            
        if sImageName.endswith('.tif') or sImageName.endswith('jpg') or sImageName.endswith('.png') or sImageName.endswith('.bmp') :                                      
                           
            InputImage = cv2.imread(os.path.join(sDirectoryForInputImages, aImageName), cv2.IMREAD_COLOR)  
                                            
            aImageClone = InputImage.copy()  
            
            aImageB = InputImage[:,:,0]
            aImageG = InputImage[:,:,1]
            aImageR = InputImage[:,:,2]
            
            #if nNbrOfData == 0 :               
            #    aImageGOut=ExtractBloodVesselImageGreyJongwoo(aImageG, sDirectoryForOutputImages, sImageName, True)

            aImageClaheOut=ClaheOperatorOnly(aImageG, sDirectoryForOutputImages, sImageName, False)
                        
            # Otsu's thresholding after Gaussian filtering
            blur = cv2.GaussianBlur(aImageClaheOut,(5,5),0)
            threshold, aImageOtsuOut = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)    
        
            element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
            
            nIteration = 50
            nIteration_erode = nIteration
            nIteration_dilate = nIteration
            
            for i in range (nIteration_erode):
                aImageOtsuOut = cv2.erode(aImageOtsuOut, element)

            for i in range (nIteration_dilate):
                aImageOtsuOut = cv2.dilate(aImageOtsuOut, element)            
                
            
            if( nInvertImage == 1 ):
                aImageOtsuOut = np.invert(aImageOtsuOut)            
            #BinaryMask, ContourDraw = findLargestContourColor( aImageOtsuOut.copy(), aImageClone.copy() )
            
            
            
            #output_imgLabel_Vote_largestblob, output_mask = PickTheLargestBlob( aImageOtsuOut, 0 )            
            #Temp, RetinaBlob = PickTheLargestBlob( InputImage.copy(), aImageOtsuOut, 127 )
            finalBinaryMask, finalImageWMaskContour = PickTheLargestBlobInImage( InputImage.copy(), aImageOtsuOut, 127 )
            
            #finalImage, output_imgLabel_Vote_largestblob, output_imgLabel_Vote_largestblobColor, output_imgLabel_Vote_largestblobContour, output_imgLabel_Vote_largestblobContourColor = PickTheLargestBlobForBlobAndContour( InputImage, nThreshold = 127, nFillInsideBlob = True, FillColor =(255,255,255), ContourColor = (0,255,0) )
              
            #[x, y, w, h] = cv2.boundingRect(output_imgLabel_Vote_largestblob)
            [y, x, h, w] = cv2.boundingRect(finalBinaryMask)
                       
            #20220728 must be removed later
            InputImageWROI = InputImage.copy()
            cv2.rectangle(cv2.rectangle(InputImageWROI,(x,y),(x+w,y+h), (255,0,0), 2),(x,y),(x+w,y+h), (255,0,0), 2)
                        
            ROIImageS, ROIGTS, ri1, ri2, rj1, rj2, ROIImage, ROIGT, i1,i2,j1,j2, OverlapOutputImageROI, OverlapOutputImageROIAndMaskContour, sWarning = getROIImageAndGTAndROICoordinatesIJJW( InputImage.copy(), finalImageWMaskContour, finalBinaryMask, [[x,y,(x+w),(y+h)]], 1.0, False, True )

            #RetinaROI = RetinaBlob[ri1:ri2,rj1:rj2].copy()               
            #RetinaImage, RetinaBlob, ROIOpticDiscBlobColor, ROIOpticDiscBoundary, ROIOpticDiscBoundaryColor = PickTheLargestBlobForBlobAndContour( ROIGT, 0, True, (0,255,0), (0,255,255) )           

            
            if (nNbrOfData < 10  ):
                DisplayAndSaveImages(sDirectoryForOutputImages, sImageName, aImageClaheOut,  '== Clahe Image ==','',True )                   
                DisplayAndSaveImages(sDirectoryForOutputImagesOtsuBinary, sImageName, aImageOtsuOut,  '== Otsu Binary Image ==','',True )                   
                DisplayAndSaveImages(sDirectoryForOutputImagesOtsuBoundaryAndROI, sImageName, OverlapOutputImageROIAndMaskContour,  '== Image with ROI and Contour Boundary ==','',True )                   
                DisplayAndSaveImages(sDirectoryForOutputImagesROISquareOnly, sImageName, ROIImageS,  '== Cropped ROI Image Square==','',True )                   
                DisplayAndSaveImages(sDirectoryForOutputImagesROIOnly, sImageName, ROIImage,  '== Cropped ROI Image ==','',True )                   
            else:
                DisplayAndSaveImages(sDirectoryForOutputImages, sImageName, aImageClaheOut,  '== Clahe Image ==','', False )                   
                DisplayAndSaveImages(sDirectoryForOutputImagesOtsuBinary, sImageName, aImageOtsuOut,  '== Otsu Binary Image ==','', False  )                   
                DisplayAndSaveImages(sDirectoryForOutputImagesOtsuBoundaryAndROI, sImageName, OverlapOutputImageROIAndMaskContour,  '== Image with ROI and Contour Boundary ==','',False )                   
                DisplayAndSaveImages(sDirectoryForOutputImagesROISquareOnly, sImageName, ROIImageS,  '== Cropped ROI Image Square==','',False )                   
                DisplayAndSaveImages(sDirectoryForOutputImagesROIOnly, sImageName, ROIImage,  '== Cropped ROI Image ==','',False )                   
           
            nNbrOfData +=1
            
            if ( nNbrOfData%10 == 0 ):
                print('[{}/{}] {} done'.format( nNbrOfData, len(oImages),  sImageName))
                 
        elif sImageName.find("thumbs.db") >= 0 :
            print('Removed wrong file {0}: {1}'.format(nNbrOfData, sImageName))               
        else :
            print('Removed wrong file {0}: {1}'.format(nNbrOfData, sImageName))  
                 
                            
    print('Loading Test Data done: {0} Data'.format(nNbrOfData)) 
    
  


def PickTheLargestBlobInImage( InputImage, InputBinaryImageOtsu, nThreshold = 127 ):            

    InputImageOriginal = InputImage.copy()  
    finalImageWMaskContour = InputImage.copy()  
    
    ret, BinaryInputImage  = cv2.threshold(InputBinaryImageOtsu.astype(np.uint8), nThreshold, 255, cv2.THRESH_BINARY)    
           
    # Find the largest contour and extract it
    #im, contours, hierarchy = cv2.findContours(BinaryInputImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
    temp, contours, hierarchy = cv2.findContours(BinaryInputImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
    
    if( len(contours) >= 1 ):
        maxContour = 0
        for contour in contours:
            contourSize = cv2.contourArea(contour)
            if ( contourSize > maxContour ):
                maxContour = contourSize
                maxContourData = contour
        
        # Create a mask from the largest contour
        if( maxContour > 0 ):
            finalBinaryMask = np.zeros_like(BinaryInputImage)
            cv2.fillPoly(finalBinaryMask,[maxContourData],1)

            cv2.drawContours(finalImageWMaskContour, contour, -1, (0, 255, 0), 3)
        
            # Use mask to crop data from original image
            #finalImage = np.zeros_like(BinaryInputImage)
            #finalImage[:,:] = np.multiply(InputImageOriginal,finalBinaryMask)
            
        else:
            finalImage = InputImageOriginal.copy()
            finalBinaryMask  = (finalImage > nThreshold).astype(np.uint8)
            finalImageWMaskContour = finalImageWMaskContour
                               
    else:
        finalImage = InputImageOriginal.copy()
        finalBinaryMask = (finalImage > nThreshold).astype(np.uint8)
        finalImageWMaskContour = finalImageWMaskContour
        
    return finalBinaryMask, finalImageWMaskContour



def getROIImageAndGTAndROICoordinatesIJJW( image, imageWMaskContour, gt_mask, pred_box, fRoiScale=1.0, bResizeInput = False, bROIMaxOnly = True ):       
    
    sWarning=''
    
    OverlapOutputImage = image.copy()    
    OverlapOutputImageWContour = imageWMaskContour.copy()    

    nMinimumRange = 10
    
    fResizeRatioI = 1.0
    fResizeRatioJ = 1.0

    if( bResizeInput ):
        fResizeRatioI = float(image.shape[0])/float(gt_mask.shape[0])
        fResizeRatioJ = float(image.shape[1])/float(gt_mask.shape[1])
    
    
    boxes = np.concatenate([ pred_box])
    
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")

    for i in range(N):
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        
        oi1, oj1, oi2, oj2 = boxes[i]
        i1, j1, i2, j2 = boxes[i]
        i1 =int(fResizeRatioI*float(i1))
        if i1 > image.shape[0]:
            i1 = image.shape[0]-nMinimumRange
            
        if( i1 < 0 ):
            i1 = nMinimumRange
            
        i2 =int(fResizeRatioI*float(i2))
        if i2 > image.shape[0]:
            i2 = image.shape[0]-nMinimumRange
        
        j1 =int(fResizeRatioJ*float(j1))
        if j1 > image.shape[1]:
            j1 = image.shape[1]-nMinimumRange
            
        if( j1 < 0 ):
            j1 = nMinimumRange
            
        
        j2 =int(fResizeRatioJ*float(j2))
        if j2 > image.shape[1]:
            j2 = image.shape[1]-nMinimumRange        
        
                
        nRange= int(float(i2-i1)*fRoiScale/2.0)
        if( (i2-i1) < (j2-j1)):
            nRange = int(float(j2-j1)*fRoiScale/2.0)
            
        ncenterI = int(float(i2+i1)/2.0)
        ncenterJ = int(float(j2+j1)/2.0)

        ri1 = int(ncenterI-nRange)
        ri2 = int(ncenterI+nRange)
        rj1 = int(ncenterJ-nRange)
        rj2 = int(ncenterJ+nRange)
        
        di1=0
        if( ri1 < 0 ):
            di1= abs(ri1)            
            ri1 = 0
        
        di2=0
        if( ri2 > (image.shape[0]-1) ):
            di2 = abs(ri2 - (image.shape[0]-1))
            ri2 = image.shape[0]-1
        
        ri1 = ri1 - di2
        ri2 = ri2 + di1
        
        dj1=0
        if( rj1 < 0 ):
            dj1= abs(rj1)            
            rj1 = 0
                        
        dj2=0
        if( rj2 > (image.shape[1]-1) ):
            dj2 = abs(rj2 - (image.shape[1]-1))
            rj2 = image.shape[1]-1
            
        rj1 = rj1 - dj2
        rj2 = rj2 + dj1
        
        
        nMinimumRange = 10
        if( ri1 < 0 ):
            ri1 = nMinimumRange
            
        if( ri2 > int(image.shape[0]) ):
            ri2 = int(image.shape[0]) - nMinimumRange
            
        if( rj1 < 0 ):
            rj1= nMinimumRange
            
        if( rj2 > int(image.shape[1]) ):
            rj2 = int(image.shape[1]) - nMinimumRange            
        
        
        print("ROI[{}:{},{}:{}] *[{} or {}] = new ROI[{}:{},{}:{}] => Real ROI Range[R={}][{}:{},{}:{}]".format(oi1,oi2,oj1,oj2,fResizeRatioI,fResizeRatioJ,i1,i2,j1,j2,nRange,ri1,ri2,rj1,rj2))


        if( int(N) > 1 ):
            sWarning += "ROI[{}:{},{}:{}] *[{} or {}] = new ROI[{}:{},{}:{}] => Real ROI Range[R={}][{}:{},{}:{}] \n".format(oi1,oi2,oj1,oj2,fResizeRatioI,fResizeRatioJ,i1,i2,j1,j2,nRange,ri1,ri2,rj1,rj2)
            
        #color  = Red   (255, 0, 0)  
        #OverlapOutputImage  = cv2.rectangle(OverlapOutputImage, (j1, i1), (j2, i2), (255, 0, 0), thickness=3, lineType=8, shift=0)
        #OverlapOutputImage  = cv2.rectangle(OverlapOutputImage, (rj1, ri1), (rj2, ri2), (255, 0, 0), thickness=3, lineType=8, shift=0)

        #color  = Blue   (0, 0, 255)  
        OverlapOutputImage  = cv2.rectangle(OverlapOutputImage, (j1, i1), (j2, i2), (0, 0, 255), thickness=3, lineType=8, shift=0)
        OverlapOutputImage  = cv2.rectangle(OverlapOutputImage, (rj1, ri1), (rj2, ri2), (0, 255, 255), thickness=3, lineType=8, shift=0)


        OverlapOutputImageWContour  = cv2.rectangle(OverlapOutputImageWContour, (j1, i1), (j2, i2), (0, 0, 255), thickness=3, lineType=8, shift=0)
        OverlapOutputImageWContour  = cv2.rectangle(OverlapOutputImageWContour, (rj1, ri1), (rj2, ri2), (0, 255, 255), thickness=3, lineType=8, shift=0)
     
        ROIImageS =   image[ri1:ri2,rj1:rj2].copy()
        ROIGTS    = gt_mask[ri1:ri2,rj1:rj2].copy()

        ROIImage =   image[i1:i2,j1:j2].copy()
        ROIGT    = gt_mask[i1:i2,j1:j2].copy()
        
        if( bROIMaxOnly and i == 0):
            ROIImageH = ROIImage.copy()
            ROIGTH    = ROIGT.copy()
            ri1H = ri1
            ri2H = ri2
            rj1H = rj1
            rj2H = rj2
            
            
    if( bROIMaxOnly and int(N) > 1):
        ROIImage = ROIImageH
        ROIGT    = ROIGTH
        ri1 = ri1H
        ri2 = ri2H 
        rj1 = rj1H 
        rj2 = rj2H
        
            
    return ROIImageS, ROIGTS, ri1, ri2, rj1, rj2, ROIImage, ROIGT, i1, i2, j1, j2,  OverlapOutputImage, OverlapOutputImageWContour, sWarning





   

def LoadDirectorysAndEnhanceImagesUsingOpenCVAndSaveUsingClaheMorphologyBinarization( sDirectoryForInputImages, sDirectoryForOutputImages, sDirectoryForOutputImagesOtsuBinary, sDirectoryForOutputImagesOtsuBoundary  ):
   
    if (os.path.exists(sDirectoryForOutputImages)):
        shutil.rmtree(sDirectoryForOutputImages)
        
    os.mkdir(sDirectoryForOutputImages)  


    if (os.path.exists(sDirectoryForOutputImagesOtsuBinary)):
        shutil.rmtree(sDirectoryForOutputImagesOtsuBinary)
        
    os.mkdir(sDirectoryForOutputImagesOtsuBinary)  
    
    
    if (os.path.exists(sDirectoryForOutputImagesOtsuBoundary)):
        shutil.rmtree(sDirectoryForOutputImagesOtsuBoundary)
        
    os.mkdir(sDirectoryForOutputImagesOtsuBoundary)      
    


    nNbrOfData=0
    #lImagesInClass = os.listdir(sDirectoryForInputImages)
    oImages = os.listdir(sDirectoryForInputImages)

    for aImageName in oImages:
        sImageName = str(aImageName)
        #sImageName2 = str(sImageName.split('.')[0])
            
        if sImageName.endswith('.tif') or sImageName.endswith('jpg') or sImageName.endswith('.png') or sImageName.endswith('.bmp') :                                      
                           
            InputImage = cv2.imread(os.path.join(sDirectoryForInputImages, aImageName), cv2.IMREAD_COLOR)  
                                            
            aImageClone = InputImage.copy()  
            
            aImageB = InputImage[:,:,0]
            aImageG = InputImage[:,:,1]
            aImageR = InputImage[:,:,2]
            
            #if nNbrOfData == 0 :               
            #    aImageGOut=ExtractBloodVesselImageGreyJongwoo(aImageG, sDirectoryForOutputImages, sImageName, True)

            aImageClaheOut=ClaheOperatorOnly(aImageG, sDirectoryForOutputImages, sImageName, False)
                        
            # Otsu's thresholding after Gaussian filtering
            blur = cv2.GaussianBlur(aImageClaheOut,(5,5),0)
            threshold, aImageOtsuOut = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)    
        
            element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
            
            for i in range (100):
                aImageOtsuOut = cv2.erode(aImageOtsuOut, element)

            for i in range (100):
                aImageOtsuOut = cv2.dilate(aImageOtsuOut, element)            
                
            
                        
            BinaryMask, ContourDraw = findLargestContourColor( aImageOtsuOut.copy(), aImageClone.copy() )
            
            if (nNbrOfData < 10  ):
                DisplayAndSaveImages(sDirectoryForOutputImages, sImageName, aImageClaheOut,  '==Binary Mask Image ==','',True )                   
                DisplayAndSaveImages(sDirectoryForOutputImagesOtsuBinary, sImageName, BinaryMask,  '==Binary Mask Image ==','',True )                   
                DisplayAndSaveImages(sDirectoryForOutputImagesOtsuBoundary, sImageName, ContourDraw,  '==Boundary Mask Image ==','', True  )       
            else:
                DisplayAndSaveImages(sDirectoryForOutputImages, sImageName, aImageGOut,  '==Binary Mask Image ==','', False )                   
                DisplayAndSaveImages(sDirectoryForOutputImagesOtsuBinary, sImageName, BinaryMask,  '==Binary Mask Image ==','', False  )                   
                DisplayAndSaveImages(sDirectoryForOutputImagesOtsuBoundary, sImageName, ContourDraw,  '==Boundary Mask Image ==','', False  )       
           
            nNbrOfData +=1
                 
        elif sImageName.find("thumbs.db") >= 0 :
            print('Removed wrong file {0}: {1}'.format(nNbrOfData, sImageName))               
        else :
            print('Removed wrong file {0}: {1}'.format(nNbrOfData, sImageName))  
                 
                            
    print('Loading Test Data done: {0} Data'.format(nNbrOfData)) 
    
    
    
    

def findLargestContourColor( InputBinaryImage, InputColorImage ):
    
    #convexMask = np.zeros(InputBinaryImage.shape, dtype=np.uint8)
    BinaryMask = np.zeros(InputBinaryImage.shape, dtype=np.uint8)
    ContourDraw = InputColorImage.copy()
    
    
    
    
    #contours, hierarchy = cv2.findContours(InputBinaryImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours,hierarchy = cv2.findContours(InputBinaryImage.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )
     
    # find the largest blob
    maxContour = 0
    for contour in contours:
        if( len(contour) > 4 ):
            contourSize = cv2.contourArea(contour)
            if contourSize > maxContour:
                maxContour = contourSize
                maxContourData = contour    


    #convert to convex hull
    #hull = cv2.convexHull(maxContourData)    

    cv2.fillPoly(BinaryMask,[maxContourData],255)
    cv2.drawContours(ContourDraw, [maxContourData], -1, (0, 0, 255), 5*2)    

    return BinaryMask, ContourDraw
        
    
    
    

def LoadDirectorysAndEnhanceImagesUsingOpenCVAndSaveUsingClaheHE( sDirectoryForInputImages, sDirectoryForOutputImages, sDirectoryForOutputImagesHE ):
   
    if (os.path.exists(sDirectoryForOutputImages)):
        shutil.rmtree(sDirectoryForOutputImages)
        
    os.mkdir(sDirectoryForOutputImages)  

    if (os.path.exists(sDirectoryForOutputImagesHE)):
        shutil.rmtree(sDirectoryForOutputImagesHE)
        
    os.mkdir(sDirectoryForOutputImagesHE)  
    
    
    nNbrOfData=0
    #lImagesInClass = os.listdir(sDirectoryForInputImages)
    oImages = os.listdir(sDirectoryForInputImages)

    for aImageName in oImages:
        sImageName = str(aImageName)
        #sImageName2 = str(sImageName.split('.')[0])
            
        if sImageName.endswith('.tif') or sImageName.endswith('jpg') or sImageName.endswith('.png') or sImageName.endswith('.bmp') :                                      
                           
            InputImage = cv2.imread(os.path.join(sDirectoryForInputImages, aImageName), cv2.IMREAD_COLOR)  
                                            
            aImageClone = InputImage.copy()  
            
            aImageB = InputImage[:,:,0]
            aImageG = InputImage[:,:,1]
            aImageR = InputImage[:,:,2]
            
            #if nNbrOfData == 0 :               
            #    aImageGOut=ExtractBloodVesselImageGreyJongwoo(aImageG, sDirectoryForOutputImages, sImageName, True)

            aImageClaheOut,aImageClaheHEOut =ClaheAndHEOperator(aImageG, sDirectoryForOutputImages, sDirectoryForOutputImagesHE,sImageName, False)
            
            nNbrOfData +=1
                 
        elif sImageName.find("thumbs.db") >= 0 :
            print('Removed wrong file {0}: {1}'.format(nNbrOfData, sImageName))               
        else :
            print('Removed wrong file {0}: {1}'.format(nNbrOfData, sImageName))  
                 
                            
    print('Loading Test Data done: {0} Data'.format(nNbrOfData))    
    


def LoadDirectorysAndEnhanceImagesUsingOpenCVAndSave( sDirectoryForInputImages, sDirectoryForOutputImages ):
   
    if (os.path.exists(sDirectoryForOutputImages)):
        shutil.rmtree(sDirectoryForOutputImages)
        
    os.mkdir(sDirectoryForOutputImages)  


    if (os.path.exists(sDirectoryForOutputImages)):
        shutil.rmtree(sDirectoryForOutputImages)
        
    os.mkdir(sDirectoryForOutputImages)  



    nNbrOfData=0
    #lImagesInClass = os.listdir(sDirectoryForInputImages)
    oImages = os.listdir(sDirectoryForInputImages)

    for aImageName in oImages:
        sImageName = str(aImageName)
        #sImageName2 = str(sImageName.split('.')[0])
            
        if sImageName.endswith('.tif') or sImageName.endswith('jpg') or sImageName.endswith('.png') or sImageName.endswith('.bmp') :                                      
                           
            InputImage = cv2.imread(os.path.join(sDirectoryForInputImages, aImageName), cv2.IMREAD_COLOR)  
                                            
            aImageClone = InputImage.copy()  
            
            aImageB = InputImage[:,:,0]
            aImageG = InputImage[:,:,1]
            aImageR = InputImage[:,:,2]
            
            #if nNbrOfData == 0 :               
            #    aImageGOut=ExtractBloodVesselImageGreyJongwoo(aImageG, sDirectoryForOutputImages, sImageName, True)

            aImageGOut=ExtractBloodVesselImageGreyJongwoo(aImageG, sDirectoryForOutputImages, sImageName, False)
            
            nNbrOfData +=1
                 
        elif sImageName.find("thumbs.db") >= 0 :
            print('Removed wrong file {0}: {1}'.format(nNbrOfData, sImageName))               
        else :
            print('Removed wrong file {0}: {1}'.format(nNbrOfData, sImageName))  
                 
                            
    print('Loading Test Data done: {0} Data'.format(nNbrOfData))  
    

#def LoadTestDataForSegmentation( sDirectory, nImageRows, nImageColumns, nImageChannels, sTestDataName ):
def LoadDirectorysResizeAndEnhanceImagesUsingOpenCVAndSave( sDirectoryForInputImages, sDirectoryForOutputImages ):
   
    if (os.path.exists(sDirectoryForOutputImages)):
        shutil.rmtree(sDirectoryForOutputImages)
        
    os.mkdir(sDirectoryForOutputImages)  

    nNbrOfData=0
    #lImagesInClass = os.listdir(sDirectoryForInputImages)
    oImages = os.listdir(sDirectoryForInputImages)

    for aImageName in oImages:
        sImageName = str(aImageName)
        #sImageName2 = str(sImageName.split('.')[0])
            
        if sImageName.endswith('.tif') or sImageName.endswith('jpg') or sImageName.endswith('.png') or sImageName.endswith('.bmp') :                                      
                           
            InputImage = cv2.imread(os.path.join(sDirectoryForInputImages, aImageName), cv2.IMREAD_COLOR)  
            
            if (int(InputImage.shape[0]) > 999 ):   #shape =(row, column)
                Row = 999                
                ratio= float(InputImage.shape[0])/float(999.0)              
                Column = int(InputImage.shape[1]/ratio)
                
                InputImage =cv2.resize(InputImage,(Column,Row))
                                
            aImageClone = InputImage.copy()  
            aImageRed = InputImage[:,:,2]
            
            aImageB = InputImage[:,:,0]
            aImageG = InputImage[:,:,1]
            aImageR = InputImage[:,:,2]
            
            #if nNbrOfData == 0 :               
            #    aImageGOut=ExtractBloodVesselImageGreyJongwoo(aImageG, sDirectoryForOutputImages, sImageName, True)

            aImageGOut=ExtractBloodVesselImageGreyJongwoo(aImageG, sDirectoryForOutputImages, sImageName, False)
            
            nNbrOfData +=1
            '''
            aImageRedBlur = cv2.GaussianBlur(aImageRed,(5,5),0)  #Gaussian
                        
            threshold,aImageRedBlurOtsu = cv2.threshold(aImageRedBlur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


            nMaxArea=0
            nMaxX=0
            nMaxY=0
            nMaxW=0
            nMaxH=0                
            im2, contours, hierarchy = cv2.findContours(aImageRedBlurOtsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)            
            for c in contours:
                #rect = cv2.boundingRect(c)
                x,y,w,h = cv2.boundingRect(c)
                
                if( w*h > nMaxArea ):
                    nMaxArea=w*h
                    nMaxX=x
                    nMaxY=y
                    nMaxW=w
                    nMaxH=h
                                
            cv2.rectangle(aImageClone,(nMaxX,nMaxY),(nMaxX+nMaxW,nMaxY+nMaxH),(0,0,255),4)           
                               
            OutImage = InputImage[nMaxY:(nMaxY+nMaxH), nMaxX:(nMaxX+nMaxW)]
            #cv2.imshow("cropped Image", OutImage)
            #cv2.waitKey(0)            
            #cv2.imwrite(os.path.join(sDirectoryForOutputImages,lImagesInClass,sImageName), OutImage )      
            cv2.imwrite(os.path.join(sDirectoryForOutputImages,sImageName), OutImage )      
        
            if (nNbrOfData % 10 == 0):
                print('Done: {0} {1} images'.format(nNbrOfData, sImageName))
                
            nNbrOfData += 1

            bPrintResult=False
            if( nNbrOfData < 3 and bPrintResult == True ):
                cv2.imshow("cropped area", aImageClone)
                aImageClone = cv2.resize(aImageClone,None,fx=0.01, fy=0.01, interpolation = cv2.INTER_LINEAR )
                cv2.waitKey(0)            

                print('nMaxX={0} nMaxX+nMaxW={1}'.format( nMaxX, nMaxX+nMaxW))
                print('nMaxY={0} nMaxY+nMaxH={1}'.format(nMaxY, nMaxY+nMaxH))            
                print('nXCenter={0}, nYCenter={1}'.format( (nMaxX+nMaxW/2), (nMaxY-nMaxY+nMaxH/2)))
                if cv2.waitKey(): 
                    cv2.destroyAllWindows()
            '''
                 
        elif sImageName.find("thumbs.db") >= 0 :
            print('Removed wrong file {0}: {1}'.format(nNbrOfData, sImageName))               
        else :
            print('Removed wrong file {0}: {1}'.format(nNbrOfData, sImageName))  
                 
                            
    print('Loading Test Data done: {0} Data'.format(nNbrOfData))        
    

 
def ClaheOperator( aInputImage, sDirectoryForOutputImages, sImageName, bShowResult):
    
    aInputImageClone = aInputImage.copy()
    
    if( bShowResult ):
        DisplayAndSaveImages(sDirectoryForOutputImages, sImageName, aInputImageClone,  '==Original Image==',  '-Org' )                

    #aImageBlur          = cv2.blur(aInputImageClone, ksize=(3, 3))
    #aImageMedianBlur    = cv2.medianBlur(aInputImageClone, 3)
    aImageMedianBlur    = aInputImageClone
    

    nImageRows = aInputImage.shape[0]
    nImageCols = aInputImage.shape[1]
    
    
    #defaule  nClipSize = 4
    nClipSize = 70

    if( nImageRows > nImageCols):
        nSize = (int)(nImageCols/(nClipSize*nClipSize))
    else:
        nSize = (int)(nImageRows/(nClipSize*nClipSize))
   
    if( nSize%2 == 0):
        nSize = nSize-1
    
    nSize = m_nTileSizeForClahe   

    if( m_nTileSizeForClahe == 9999 ):
        if( nImageRows < nImageCols ):
            nSize = int( float(nImageRows)*0.90 )   
        else:
            nSize = int( float(nImageCols)*0.90 )   
    elif( m_nTileSizeForClahe == 9000 ):
        if( nImageRows < nImageCols ):
            nSize = int( float(nImageRows)*0.90 )   
        else:
            nSize = int( float(nImageCols)*0.90 )               
    elif( m_nTileSizeForClahe == 9500 ):
        if( nImageRows < nImageCols ):
            nSize = int( float(nImageRows)*0.95 )   
        else:
            nSize = int( float(nImageCols)*0.95 )               
    elif( m_nTileSizeForClahe == 9900 ):
        if( nImageRows < nImageCols ):
            nSize = int( float(nImageRows)*0.99 )   
        else:
            nSize = int( float(nImageCols)*0.99 )                       
        
    Clahe = cv2.createCLAHE(clipLimit=2.55, tileGridSize=(nSize, nSize))
    aImageClahe = Clahe.apply(aImageMedianBlur)
    
    #aImageHistEqu = cv2.equalizeHist(aImageMedianBlur[:,:,0], aImageHistEqu[:,:,0])
    
    aImageHistEqu = cv2.equalizeHist(aImageMedianBlur)
    
    
    
    if( True ):   # if( bShowResult ):
        DisplayAndSaveImages(sDirectoryForOutputImages, sImageName, aImageClahe,  '==Clahe Image ==',''  )                

    #if( False ):   # if( bShowResult ):
    #    DisplayAndSaveImages(sDirectoryForOutputImagesHE, sImageName, aImageHistEqu,  '==HE Image ==','')                

    return aImageClahe




 
def ClaheOperatorOnly( aInputImage, sDirectoryForOutputImages, sImageName, bShowResult):
    
    aInputImageClone = aInputImage.copy()
    
    if( bShowResult and False ):
        DisplayAndSaveImages(sDirectoryForOutputImages, sImageName, aInputImageClone,  '==Original Image==',  '-Org' )                

    #aImageBlur          = cv2.blur(aInputImageClone, ksize=(3, 3))
    #aImageMedianBlur    = cv2.medianBlur(aInputImageClone, 3)
    aImageMedianBlur    = aInputImageClone
    

    nImageRows = aInputImage.shape[0]
    nImageCols = aInputImage.shape[1]
    
    
    #defaule  nClipSize = 4
    nClipSize = 70

    if( nImageRows > nImageCols):
        nSize = (int)(nImageCols/(nClipSize*nClipSize))
    else:
        nSize = (int)(nImageRows/(nClipSize*nClipSize))
   
    if( nSize%2 == 0):
        nSize = nSize-1
    
    nSize = m_nTileSizeForClahe   

    if( m_nTileSizeForClahe == 9999 ):
        if( nImageRows < nImageCols ):
            nSize = int( float(nImageRows)*0.90 )   
        else:
            nSize = int( float(nImageCols)*0.90 )   
        
    Clahe = cv2.createCLAHE(clipLimit=2.55, tileGridSize=(nSize, nSize))
    aImageClahe = Clahe.apply(aImageMedianBlur)
    
    #aImageHistEqu = cv2.equalizeHist(aImageMedianBlur[:,:,0], aImageHistEqu[:,:,0])
    
    aImageHistEqu = cv2.equalizeHist(aImageMedianBlur)
    
    
    
    if( False ):   # if( bShowResult ):
        DisplayAndSaveImages(sDirectoryForOutputImages, sImageName, aImageClahe,  '==Clahe Image ==',''  )                

    #if( False ):   # if( bShowResult ):
    #    DisplayAndSaveImages(sDirectoryForOutputImagesHE, sImageName, aImageHistEqu,  '==HE Image ==','')                

    return aImageClahe


def ClaheOperatorColor3D( aInputImage, nClaheTileSize ):
    
    aImage    = aInputImage.copy()
    aImageB   = aImage[:,:,0]
    aImageG   = aImage[:,:,1]
    aImageR   = aImage[:,:,2]
     
    nSize = nClaheTileSize   
    
    nImageRows = aImage.shape[0]
    nImageCols = aImage.shape[1]
    
    if( m_nTileSizeForClahe == 9999 ):
        if( nImageRows < nImageCols ):
            nSize = int( float(nImageRows)*0.90 )   
        else:
            nSize = int( float(nImageCols)*0.90 )   
    elif( m_nTileSizeForClahe == 9000 ):
        if( nImageRows < nImageCols ):
            nSize = int( float(nImageRows)*0.90 )   
        else:
            nSize = int( float(nImageCols)*0.90 )               
    elif( m_nTileSizeForClahe == 9500 ):
        if( nImageRows < nImageCols ):
            nSize = int( float(nImageRows)*0.95 )   
        else:
            nSize = int( float(nImageCols)*0.95 )               
    elif( m_nTileSizeForClahe == 9900 ):
        if( nImageRows < nImageCols ):
            nSize = int( float(nImageRows)*0.99 )   
        else:
            nSize = int( float(nImageCols)*0.99 )                      
        
        
    Clahe = cv2.createCLAHE(clipLimit=2.55, tileGridSize=(nSize, nSize))
    aImageClaheB = Clahe.apply(aImageB)
    aImageClaheG = Clahe.apply(aImageG)
    aImageClaheR = Clahe.apply(aImageR)
    
    #aImageHistEqu = cv2.equalizeHist(aImageMedianBlur[:,:,0], aImageHistEqu[:,:,0])
    
    aImageClaheColor = np.zeros((aInputImage.shape[0], aInputImage.shape[1],3), np.uint8)
    aImageClaheColor[:,:,0] = aImageClaheB
    aImageClaheColor[:,:,1] = aImageClaheG 
    aImageClaheColor[:,:,2] = aImageClaheR     
                

    return aImageClaheColor   
    

    
 
def ClaheAndHEOperator( aInputImage, sDirectoryForOutputImages, sDirectoryForOutputImagesHE, sImageName, bShowResult):
    
    aInputImageClone = aInputImage.copy()
    
    if( bShowResult ):
        DisplayAndSaveImages(sDirectoryForOutputImages, sImageName, aInputImageClone,  '==Original Image==',  '-Org' )                

    #aImageBlur          = cv2.blur(aInputImageClone, ksize=(3, 3))
    #aImageMedianBlur    = cv2.medianBlur(aInputImageClone, 3)
    aImageMedianBlur    = aInputImageClone
    

    nImageRows = aInputImage.shape[0]
    nImageCols = aInputImage.shape[1]
    
    #defaule  nClipSize = 4
    nClipSize = 24
    if( nImageRows > nImageCols):
        nSize = (int)(nImageCols/(nClipSize*nClipSize))
    else:
        nSize = (int)(nImageRows/(nClipSize*nClipSize))
   
    
    if( nSize%2 == 0):
        nSize = nSize-1
                
    
        
    Clahe = cv2.createCLAHE(clipLimit=2.55, tileGridSize=(nSize, nSize))
    #Clahe = cv2.createCLAHE(clipLimit=2.55, tileGridSize=(nImageRows-1, nImageCols-1))
    aImageClahe = Clahe.apply(aImageMedianBlur)
    
    #aImageHistEqu = cv2.equalizeHist(aImageMedianBlur[:,:,0], aImageHistEqu[:,:,0])
    
    aImageHistEqu = cv2.equalizeHist(aImageClahe)
    
    
    
    if( True ):   # if( bShowResult ):
        DisplayAndSaveImages(sDirectoryForOutputImages, sImageName, aImageClahe,  '==Clahe Image ==',''  )                

    if( True ):   # if( bShowResult ):
        DisplayAndSaveImages(sDirectoryForOutputImagesHE, sImageName, aImageHistEqu,  '==HE Image ==','')                

    return aImageClahe, aImageHistEqu


    
 
def ExtractBloodVesselImageGreyJongwoo( aInputImage, sDirectoryForOutputImages, sImageName, bShowResult):
    
    aInputImageClone = aInputImage.copy()
    
    if( bShowResult ):
        DisplayAndSaveImages(sDirectoryForOutputImages, sImageName, aInputImageClone,  '==Original Image==',  '-Org' )                

    #aImageBlur          = cv2.blur(aInputImageClone, ksize=(3, 3))
    #aImageMedianBlur    = cv2.medianBlur(aInputImageClone, 3)
    aImageMedianBlur    = aInputImageClone
    

    nImageRows = aInputImage.shape[0]
    nImageCols = aInputImage.shape[1]
    
    #defaule  nClipSize = 4
    nClipSize = 2
    if( nImageRows > nImageCols):
        nSize = (int)(nImageCols/(nClipSize*nClipSize))
    else:
        nSize = (int)(nImageRows/(nClipSize*nClipSize))
   
    if( nSize%2 == 0):
        nSize = nSize-1
                
        
    Clahe = cv2.createCLAHE(clipLimit=2.55, tileGridSize=(nSize, nSize))
    aImageClahe = Clahe.apply(aImageMedianBlur)
    
    #aImageHistEqu = cv2.equalizeHist(aImageMedianBlur[:,:,0], aImageHistEqu[:,:,0])
    
    aImageHistEqu = cv2.equalizeHist(aImageMedianBlur)
    
    
    
    if( True ):   # if( bShowResult ):
        DisplayAndSaveImages(sDirectoryForOutputImages, sImageName, aImageClahe,  '==Clahe Image ==',''  )                

    if( True ):   # if( bShowResult ):
        DisplayAndSaveImages(sDirectoryForOutputImagesHistogram, sImageName, aImageHistEqu,  '==HE Image ==','')                

    return aInputImageClone


    #Make Color overlap images.
    
    
 
def ExtractBloodVesselImageGreyJongwoo20200709( aInputImage, sDirectoryForOutputImages, sImageName, bShowResult):
    
    aInputImageClone = aInputImage.copy()
    
    if( bShowResult ):
        DisplayAndSaveImages(sDirectoryForOutputImages, sImageName, aInputImageClone,  '==Original Image==',  '-Org' )                

    #aImageBlur          = cv2.blur(aInputImageClone, ksize=(3, 3))
    #aImageMedianBlur    = cv2.medianBlur(aInputImageClone, 3)
    aImageMedianBlur    = aInputImageClone
    

    if( False ):                      
        plt.title('==Blur Image==')
        plt.imshow(aImageBlur,cmap='gray')                    
        plt.show()    

        HistAImageBlur, bin_deges = np.histogram(aImageMedianBlur, density=False )
        plt.hist(HistAImageBlur, density=False, bins=25)
        plt.show()    
           
        
    if( bShowResult ):      
        DisplayAndSaveImages(sDirectoryForOutputImages, sImageName, aImageMedianBlur,  '==MedianBlur Image==',  '-Blur' )                

        
    Threshold, aImageMedianBlurOtsu = cv2.threshold(aImageMedianBlur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    if( False ):      
       DisplayAndSaveImages(sDirectoryForOutputImages, sImageName, aImageMedianBlurOtsu,  '==MedianBlurOtsu Image [Thr:{}]=='.format(Threshold),  '-Otsu')                
        
       

    nImageRows = aInputImage.shape[0]
    nImageCols = aInputImage.shape[1]
    
    
    
    if( nImageRows > nImageCols):
        nSize = (int)(nImageCols/(4*4))
        nAdaptThrSize = (int)(nImageCols/(4))
    else:
        nSize = (int)(nImageRows/(4*4))
        nAdaptThrSize = (int)(nImageRows/(4))
   
    if( nSize%2 == 0):
        nSize = nSize-1
                
    if( nAdaptThrSize%2 == 0):
        nAdaptThrSize = nAdaptThrSize+1
        
    Clahe = cv2.createCLAHE(clipLimit=2.55, tileGridSize=(nSize, nSize))
    aImageClahe = Clahe.apply(aImageMedianBlur)
    
    #aImageHistEqu = cv2.equalizeHist(aImageMedianBlur[:,:,0], aImageHistEqu[:,:,0])
    
    aImageHistEqu = cv2.equalizeHist(aImageMedianBlur)
    
    
    
    if( True ):   # if( bShowResult ):
        DisplayAndSaveImages(sDirectoryForOutputImages, sImageName, aImageClahe,  '==Clahe Image ==',  '-Clahe')                

    if( True ):   # if( bShowResult ):
        DisplayAndSaveImages(sDirectoryForOutputImagesHistogram, sImageName, aImageHistEqu,  '==HE Image ==',  '-He')                

    return aInputImageClone


    #Make Color overlap images.
    
def Skeletonization2F(InputImage255):            
 
            
    ######################################################################
    # Skeletonization using OpenCV-Python
    # from http://opencvpython.blogspot.com/2012/05/skeletonization-using-opencv-python.html
    #####################################################################
         
    bShow=False
    #JW img = cv2.imread('sofsk.png',0)
    img = InputImage255.copy()
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
    skelLabel1 = np.zeros(img.shape,np.uint8)
    skelLabel255 = np.zeros(img.shape,np.uint8)
         
    #Jw ret,img = cv2.threshold(img,127,255,0)
    ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)    
        
    #print("Non Zero in img: {}".format(cv2.countNonZero(img)))
    #print("Max: {}    Min: {}".format(np.max(img), np.min(img)))
        
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
         
    nCount=0
    while( not done):
        nCount+=1
        eroded = cv2.erode(img, element)
        temp1 = cv2.dilate(eroded, element)
        temp2 = cv2.subtract(img,temp1)
        skel = cv2.bitwise_or(skel,temp2)        
        img = eroded.copy()
                 
        #skelc = np.maximum( (skelLabel1 >= 0).astype(int)*1 , (skel > 0).astype(int)*(nCount/255)      
        #skelLabel1 = skelc.copy()  
        #print('Sleletonization [{}] Nozeros: skel={}, skelc={}, max skelc={}'.format(nCount, cv2.countNonZero(skel), cv2.countNonZero(skelc),np.max(skelc)))                 
        
        for i in range(0, skelLabel1.shape[0]):
            for j in range(0, skelLabel1.shape[1]):
                if( temp2[i,j] > 0):
                    skelLabel1[i,j] = nCount

        if( bShow ):
            print('Sleletonization [{}] Nozeros: skel={}, skelLabel1={}, max skelLabel1={}'.format(nCount, cv2.countNonZero(skel), cv2.countNonZero(skelLabel1),np.max(skelLabel1)))                 

        zeros = size - cv2.countNonZero(img)

        if zeros==size:
            done = True
            nCount -=1

    if( bShow ):            
        Histogram = np.zeros(255)
        for i in range(0, skelLabel1.shape[0]):
            for j in range(0, skelLabel1.shape[1]):
                nIndex = skelLabel1[i,j]
                if( nIndex > 0 ):
                    Histogram[nIndex]+=1    
        
        for i in range(0,nCount+1,1):
            print("Histogram [{}] ={}".format(i, Histogram[i] ))

    '''
    import matplotlib.pyplot as plt  
    Hist,Bins = np.histogram( Histogram, bins = range(nCount+1) )
    plt.bar(Bins[:-1], Hist, width = 1)
    #plt.xlim(1, max(Bins))
    plt.title("skelLabel1 Histogram")
    plt.show() 

    print(Hist)
    print(X1)
    plt.plot(Hist,X1[1:] )
    plt.title("skelLabel1 Histogram")
    plt.show()        
    '''
    '''
    import matplotlib.pyplot as plt  
    Hist,X1 = np.histogram( skelLabel1, bins = 100, normed = True )
    plt.plot(X1[1:], Hist)
    plt.show()     
    '''
    
    #skelLabel255 = skelLabel1*255/nCount 
    
    for i in range(0, skelLabel1.shape[0]):
        for j in range(0, skelLabel1.shape[1]):
            skelLabel255[i,j] = int(float(skelLabel1[i,j])*255.0/float(nCount) + 0.5 )
    
    
    print('Sleletonization Nozeros: max skelLabel1[{}]={}, max skelLabel255[{}]={}'.format( cv2.countNonZero(skelLabel1),np.max(skelLabel1),cv2.countNonZero(skelLabel255),np.max(skelLabel255)))                 
    
    '''
    import matplotlib.pyplot as plt  
    Hist,X1 = np.histogram( skelLabel255, bins = 100, normed = True )
    plt.plot(X1[1:], Hist)
    plt.show()        
    '''    
    #print("Non Zero in skel: {}".format(cv2.countNonZero(skel)))
    #print("Max: {}    Min: {}".format(np.max(skel), np.min(skel)))
        
 
    skeleton255 = np.array(skel, dtype=np.uint8)       
    #skeleton255 = skeleton.copy()
    #skeleton255 = skeleton255 *255 
    #skeleton255 = np.array(skeleton255, dtype=np.uint8)
    
    

    
    return skeleton255, skelLabel1, skelLabel255
    



    
def SkeletonizationLabel(InputImage255):            
 
            
    ######################################################################
    # Skeletonization using OpenCV-Python
    # from http://opencvpython.blogspot.com/2012/05/skeletonization-using-opencv-python.html
    #####################################################################
    bShow = False
    #JW img = cv2.imread('sofsk.png',0)
    img = InputImage255.copy()
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
    skelLabel1 = np.zeros(img.shape,np.uint8)
    skelLabel255 = np.zeros(img.shape,np.uint8)
         
    #Jw ret,img = cv2.threshold(img,127,255,0)
    ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)    
        
    #print("Non Zero in img: {}".format(cv2.countNonZero(img)))
    #print("Max: {}    Min: {}".format(np.max(img), np.min(img)))
        
    skelLabel1= img/255
    print ("skelLabel1 max={}".format(np.max(skelLabel1)))
        
    
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
         
    nCount=0
    while( not done):
        nCount+=1        
        eroded = cv2.erode(img, element)
        img = eroded.copy()
         
        #skelc = np.maximum( (skelLabel1 > 0).astype(int)*1 , (eroded > 0).astype(int)*nCount)        
        #skelLabel1 = skelc.copy()
        

        for i in range(0, skelLabel1.shape[0]):
            for j in range(0, skelLabel1.shape[1]):
                if( eroded[i,j] > 0):
                    skelLabel1[i,j] = nCount
           
        if( bShow ):
            print('SleletonizationLabel [{}] Nozeros: eroded={}, skelLabel1={}, max skelLabel1={}'.format(nCount, cv2.countNonZero(eroded), cv2.countNonZero(skelLabel1),np.max(skelLabel1)))                 




        zeros = size - cv2.countNonZero(img)

        if zeros==size:
            done = True
        
 
    if( bShow ):            
        Histogram = np.zeros(255)
        for i in range(0, skelLabel1.shape[0]):
            for j in range(0, skelLabel1.shape[1]):
                nIndex = int(skelLabel1[i,j])
                if( nIndex > 0 ):
                    Histogram[nIndex]+=1    
        
        for i in range(0,nCount+1,1):
            print("Histogram [{}] ={}".format(i, Histogram[i] ))        
        
        
        
        
        
    skelLabel255 = skelLabel1*255/nCount  
        
    #print("Non Zero in skel: {}".format(cv2.countNonZero(skel)))
    #print("Max: {}    Min: {}".format(np.max(skel), np.min(skel)))
        
 
    skeleton255 = np.array(skel, dtype=np.uint8)       
    #skeleton255 = skeleton.copy()
    #skeleton255 = skeleton255 *255 
    #skeleton255 = np.array(skeleton255, dtype=np.uint8)

    return skeleton255, skelLabel1, skelLabel255


    
def ExtractBloodLeakage( Skeleton, BloodVessel, OriginalImages ):            
 
            
    ######################################################################
    # Skeletonization using OpenCV-Python
    # from http://opencvpython.blogspot.com/2012/05/skeletonization-using-opencv-python.html
    #####################################################################
         
    #JW img = cv2.imread('sofsk.png',0)
    #img = Skeleton.copy()
    RealBoodVessel = np.zeros(Skeleton.shape,np.uint8)
    
    #RealBoodLeakage = np.zeros(img.shape,np.uint8)
    RealBoodLeakageBinary = BloodVessel.copy()
    RealBoodLeakageGrey = OriginalImages.copy()
    
    RealBoodVessel2 = np.zeros(Skeleton.shape,np.uint8)
    RealBoodLeakageBinary2 = BloodVessel.copy()
    RealBoodLeakageGrey2 = OriginalImages.copy()    
         
    RealBoodVessel3 = np.zeros(Skeleton.shape,np.uint8)
    RealBoodLeakageBinary3 = BloodVessel.copy()
    RealBoodLeakageGrey3 = OriginalImages.copy()    
    
    nWindowSize=3
    nWindowSize2=nWindowSize*2   
    
    # Estimate real blood vessel inages
    #Jw ret,img = cv2.threshold(img,127,255,0)
    for i in range(nWindowSize2, Skeleton.shape[0]-nWindowSize2):
        for j in range(nWindowSize2, Skeleton.shape[1]-nWindowSize2):

            if( Skeleton[i,j] > 0 ):
                nCount=0
                for m in range(i-nWindowSize, i+nWindowSize+1):
                    for n in range(j-nWindowSize, j+nWindowSize+1):
                        if( BloodVessel[m,n] < 127  ):
                            nCount += 1
                if( nCount> 0 ):
                    for m in range(i-nWindowSize, i+nWindowSize+1):
                        for n in range(j-nWindowSize, j+nWindowSize+1):
                            RealBoodVessel[m,n]= BloodVessel[m,n]     
                
                bBloodVessel=False
                nTotal=0;
                nHorizontal=0;
                if( BloodVessel[i-nWindowSize2,j] < 127 and BloodVessel[i+nWindowSize2,j] < 127 ):
                    nHorizontal=1
                    nTotal +=1
                    
                nVertical=0
                if( BloodVessel[i, j-nWindowSize2] < 127 and BloodVessel[i, j+nWindowSize2] < 127 ):
                    nVertical=1
                    nTotal +=1
                    
                nLeftDiagonal=0
                if( BloodVessel[i-nWindowSize2,j-nWindowSize2] < 127 and BloodVessel[i+nWindowSize2,j+nWindowSize2] < 127 ):
                    nLeftDiagonal=1
                    nTotal +=1
                    
                nRightDiagonal=0
                if( BloodVessel[i+nWindowSize2,j-nWindowSize2] < 127 and BloodVessel[i-nWindowSize2,j+nWindowSize2] < 127 ):
                    nRightDiagonal=1
                    nTotal +=1
                    
                if( (nHorizontal+nVertical) == 1 and (nLeftDiagonal+nRightDiagonal) == 1  ):
                    bBloodVessel=True
                    
                nDiagonalTotal=0;    
                bDiagonalPrevious=False     
                bDiagonalThick=False
                bBloodVessel3=False
                for m in range(-nWindowSize2, nWindowSize2+1):
                    nDiagonalHorizontal=nDiagonalVertical=0                    
                    if( BloodVessel[i+m,j-nWindowSize] < 127 and BloodVessel[i-m,j+nWindowSize2] < 127 ):
                        nDiagonalHorizontal =1
                    if( BloodVessel[i-nWindowSize,j-m] < 127 and BloodVessel[i+nWindowSize2,j+m] < 127 ):
                        nDiagonalVertical =1
                        
                    if( (nDiagonalHorizontal+nDiagonalVertical) == 1 ):
                        nDiagonalTotal+=1
                        
                        if(bDiagonalPrevious == True ):
                            bDiagonalThick = True
                            
                        bDiagonalPrevious = True
                        
                    else:
                        bDiagonalPrevious = False
                                                                           
                if( nDiagonalTotal > 0 ):
                    
                    if(nDiagonalTotal == 1):
                        bBloodVessel3=True
                    if(nDiagonalTotal > 1 and nDiagonalTotal <= nWindowSize2 and bDiagonalThick == True):
                        bBloodVessel3=True
                    
                    
                    
                if( bBloodVessel ):
                    for m in range(i-nWindowSize2, i+nWindowSize2+1):
                        for n in range(j-nWindowSize2, j+nWindowSize2+1):
                            RealBoodVessel2[m,n]= BloodVessel[m,n]     

                if( bBloodVessel3 ):
                    for m in range(i-nWindowSize2, i+nWindowSize2+1):
                        for n in range(j-nWindowSize2, j+nWindowSize2+1):
                            RealBoodVessel3[m,n]= BloodVessel[m,n]     
                        
                    
    # Estimate real leakage and real leakage images               
    
    
    
    for i in range(0, RealBoodVessel.shape[0]):
        for j in range(0, RealBoodVessel.shape[1]):
            if( RealBoodVessel[i,j] > 0 ):
                RealBoodLeakageBinary[i,j]=0
                RealBoodLeakageGrey[i,j]=0
                
                
    for i in range(0, RealBoodVessel2.shape[0]):
        for j in range(0, RealBoodVessel2.shape[1]):
            if( RealBoodVessel2[i,j] > 0 ):
                RealBoodLeakageBinary2[i,j]=0
                RealBoodLeakageGrey2[i,j]=0                
                
    for i in range(0, RealBoodVessel3.shape[0]):
        for j in range(0, RealBoodVessel3.shape[1]):
            if( RealBoodVessel3[i,j] > 0 ):
                RealBoodLeakageBinary3[i,j]=0
                RealBoodLeakageGrey3[i,j]=0     
                
                
               
    RealBoodVessel = np.array(RealBoodVessel, dtype=np.uint8)                       
    RealBoodLeakageBinary = np.array(RealBoodLeakageBinary, dtype=np.uint8)       
    RealBoodLeakageGrey = np.array(RealBoodLeakageGrey, dtype=np.uint8)       

    RealBoodVessel2 = np.array(RealBoodVessel2, dtype=np.uint8)                       
    RealBoodLeakageBinary2 = np.array(RealBoodLeakageBinary2, dtype=np.uint8)       
    RealBoodLeakageGrey2 = np.array(RealBoodLeakageGrey2, dtype=np.uint8)       

    RealBoodVessel3 = np.array(RealBoodVessel3, dtype=np.uint8)                       
    RealBoodLeakageBinary3 = np.array(RealBoodLeakageBinary3, dtype=np.uint8)       
    RealBoodLeakageGrey3 = np.array(RealBoodLeakageGrey3, dtype=np.uint8)    
    
 
    return RealBoodVessel, RealBoodLeakageBinary, RealBoodLeakageGrey, RealBoodVessel2, RealBoodLeakageBinary2, RealBoodLeakageGrey,RealBoodVessel3, RealBoodLeakageBinary3, RealBoodLeakageGrey3       
    


    
def ExtractBloodLeakage1( Skeleton, BloodVessel, OriginalImages ):            
 
            
    ######################################################################
    # Skeletonization using OpenCV-Python
    # from http://opencvpython.blogspot.com/2012/05/skeletonization-using-opencv-python.html
    #####################################################################
         
    #JW img = cv2.imread('sofsk.png',0)
    #img = Skeleton.copy()
    RealBoodVessel = np.zeros(Skeleton.shape,np.uint8)
    
    #RealBoodLeakage = np.zeros(img.shape,np.uint8)
    RealBoodLeakageBinary = BloodVessel.copy()
    RealBoodLeakageGrey = OriginalImages.copy()
    
    RealBoodVessel2 = np.zeros(Skeleton.shape,np.uint8)
    RealBoodLeakageBinary2 = BloodVessel.copy()
    RealBoodLeakageGrey2 = OriginalImages.copy()    
         
    RealBoodVessel3 = np.zeros(Skeleton.shape,np.uint8)
    RealBoodLeakageBinary3 = BloodVessel.copy()
    RealBoodLeakageGrey3 = OriginalImages.copy()    
    
    nWindowSize=3
    nWindowSize2=nWindowSize*2   
    
    HistogramOfBloodVesselThickness = np.zeros[100]
       
    # Estimate real blood vessel inages
    #Jw ret,img = cv2.threshold(img,127,255,0)
    for i in range(nWindowSize2, Skeleton.shape[0]-nWindowSize2):
        for j in range(nWindowSize2, Skeleton.shape[1]-nWindowSize2):

            nMaxThickness=0;
            sMaxThicknessDirection=''
            nMinThickness = nWindowSize2
            sMinThicknessDirection=''
            
            if( Skeleton[i,j] > 0 ):
                
                '''
                nCount=0
                for m in range(i-nWindowSize, i+nWindowSize+1):
                    for n in range(j-nWindowSize, j+nWindowSize+1):
                        if( BloodVessel[m,n] < 127  ):
                            nCount += 1
                if( nCount> 0 ):
                    for m in range(i-nWindowSize, i+nWindowSize+1):
                        for n in range(j-nWindowSize, j+nWindowSize+1):
                            RealBoodVessel[m,n]= BloodVessel[m,n]     
                '''  
                                
                nCount=0                            
                for m in range(-nWindowSize, nWindowSize+1):
                    for n in range(-nWindowSize, nWindowSize+1):
                        if( BloodVessel[i+m,j+n] < 127  ):
                            nCount += 1
                if( nCount> 0 ):
                    for m in range(-nWindowSize, nWindowSize+1):
                        for n in range(-nWindowSize, nWindowSize+1):
                            RealBoodVessel[i+m,j+n]= BloodVessel[m,n]     
                                                                        
                bBloodVessel=False
                nTotal=0;
                nHorizontal=nHorizontalLeft=nHorizontalRight=nHorizontalThickness=0;
                
                #if( BloodVessel[i-nWindowSize2,j] < 127 and BloodVessel[i+nWindowSize2,j] < 127 ):
                #    nHorizontal=1
                #    nTotal +=1
                    
                for m in range(-1, -nWindowSize2-1, -1):
                    if( BloodVessel[i+m,j] >= 127 ):   
                        nHorizontalThickness+=1
                    else:                       
                        nHorizontalLeft =1
                        break
                        
                for m in range(1, nWindowSize2+1):
                    if( BloodVessel[i+m,j] >= 127 ):                       
                        nHorizontalThickness+=1
                    else:                       
                        nHorizontalRight = 1
                        break
                        
                if( nHorizontalLeft > 0 and nHorizontalRight > 0 ):
                    nHorizontal=1
                    nHorizontalThickness+=1                    
                    nTotal +=1
                else:
                    nHorizontalThickness=0                   
                    
                if( nHorizontalThickness >  nMaxThickness ):
                    nMaxThickness = nHorizontalThickness
                    sMaxThicknessDirection='Horixontal'
                    
                if( nHorizontalThickness <  nMinThickness ):
                    nMinThickness = nHorizontalThickness
                    sMinThicknessDirection='Horixontal'
                    
                    
                nVertical=nVerticalLeft=nVerticalRight=nVerticalThickness=0
                #if( BloodVessel[i, j-nWindowSize2] < 127 and BloodVessel[i, j+nWindowSize2] < 127 ):
                #    nVertical=1
                #    nTotal +=1
                    
                for n in range(-1,-nWindowSize2-1, -1):
                    if( BloodVessel[i,j+n] >= 127 ):   
                        nVerticalThickness+=1
                    else:
                        nVerticalLeft=1
                        break
                        
                for n in range(1, nWindowSize2+1,1):
                    if( BloodVessel[i,j+n] >= 127 ):                       
                        nVerticalThickness+=1
                    else:
                        nVerticalRight=1
                        break        
                    
                if( nVerticalLeft > 0 and nVerticalRight > 0 ):
                    nVertical=1
                    nVerticalThickness+=1                    
                    nTotal +=1
                else:
                    nVerticalThickness=0
                    
                if( nVerticalThickness >  nMaxThickness ):
                    nMaxThickness = nVerticalThickness
                    sMaxThicknessDirection='Vertical'
                    
                if( nVerticalThickness <  nMinThickness ):
                    nMinThickness = nVerticalThickness
                    sMinThicknessDirection='Vertical'

                    
                nLeftDiagonal=nLeftDiagonalLeft=nLeftDiagonalRight=nLeftDiagonalThickness=0
                #if( BloodVessel[i-nWindowSize2,j-nWindowSize2] < 127 and BloodVessel[i+nWindowSize2,j+nWindowSize2] < 127 ):
                #    nLeftDiagonal=1
                #    nTotal +=1
                    
                for m in range(-1,-nWindowSize2-1, -1):
                    if( BloodVessel[i+m,j+m] > 127 ):    
                        nLeftDiagonalThickness+=1
                    else:
                        nLeftDiagonalLeft=1
                        break
                        
                for m in range(1, nWindowSize2+1,1):
                    if( BloodVessel[i+m,j+m] > 127 ):                       
                        nLeftDiagonalThickness+=1
                    else:
                        nLeftDiagonalRight=1
                        break        
                    
                if( nLeftDiagonalLeft > 1 and nLeftDiagonalRight>1 ):
                    nLeftDiagonal=1
                    nLeftDiagonalThickness+=1
                    nTotal +=1  
                else:
                    nLeftDiagonalThickness=0
                   
                if( nLeftDiagonalThickness >  nMaxThickness ):
                    nMaxThickness = nLeftDiagonalThickness
                    sMaxThicknessDirection='LeftDiagonal'
                    
                if( nLeftDiagonalThickness <  nMinThickness ):
                    nMinThickness = nLeftDiagonalThickness
                    sMinThicknessDirection='LeftDiagonal'
                    
                    
                nRightDiagonal=nRightDiagonalLeft=nRightDiagonalRight=nRightDiagonalThickness=0
                #if( BloodVessel[i+nWindowSize2,j-nWindowSize2] < 127 and BloodVessel[i-nWindowSize2,j+nWindowSize2] < 127 ):
                #    nRightDiagonal=1
                #    nTotal +=1
                    
                for n in range(-1,-nWindowSize2, -1):
                    if( BloodVessel[i-n,j+n] > 127 ): 
                        nRightDiagonalThickness+=1
                    else:
                        nRightDiagonalLeft=1
                        break
                        
                for n in range(1,nWindowSize2+1,1):
                    if( BloodVessel[i-n,j+n] > 127 ):                       
                        nRightDiagonalThickness+=1
                    else:
                        nRightDiagonalRight=1
                        break        
                    
                if( nRightDiagonalLeft > 1 and nRightDiagonalRight > 1 ):
                    nRightDiagonal=1
                    nRightDiagonalThickness+=1
                    nTotal +=1                    
                    
                if( nRightDiagonalThickness >  nMaxThickness ):
                    nMaxThickness = nRightDiagonalThickness
                    sMaxThicknessDirection='RightDiagonal'
                    
                if( nRightDiagonalThickness <  nMinThickness ):
                    nMinThickness = nRightDiagonalThickness
                    sMinThicknessDirection='RightDiagonal'
                    
                                        
                if( (nHorizontal+nVertical) == 1 or (nLeftDiagonal+nRightDiagonal) == 1  ):
                    bBloodVessel=True
                    HistogramOfBloodVesselThickness[nMaxThickness]+=1
                    
                    
                if( bBloodVessel ):
                    for m in range(i-nWindowSize2, i+nWindowSize2+1):
                        for n in range(j-nWindowSize2, j+nWindowSize2+1):
                            RealBoodVessel2[m,n]= BloodVessel[m,n]   
                            
                            
                    
                    
                nDiagonalTotal=0;    
                bDiagonalPrevious=False     
                bDiagonalThick=False
                bBloodVessel3=False
                for m in range(-nWindowSize2, nWindowSize2+1):
                    nDiagonalHorizontal=nDiagonalVertical=0                    
                    if( BloodVessel[i+m,j-nWindowSize2] < 127 and BloodVessel[i-m,j+nWindowSize2] < 127 ):
                        nDiagonalHorizontal =1
                    if( BloodVessel[i-nWindowSize2,j-m] < 127 and BloodVessel[i+nWindowSize2,j+m] < 127 ):
                        nDiagonalVertical =1
                        
                    if( (nDiagonalHorizontal+nDiagonalVertical) == 1 ):
                        nDiagonalTotal+=1
                        
                        if(bDiagonalPrevious == True ):
                            bDiagonalThick = True
                            
                        bDiagonalPrevious = True
                        
                    else:
                        bDiagonalPrevious = False
                                                                           
                if( nDiagonalTotal > 0 ):
                    
                    if(nDiagonalTotal == 1):
                        bBloodVessel3=True
                    if(nDiagonalTotal > 1 and nDiagonalTotal <= nWindowSize2 and bDiagonalThick == True):
                        bBloodVessel3=True
                    
                    
                    

                if( bBloodVessel3 ):
                    for m in range(i-nWindowSize2, i+nWindowSize2+1):
                        for n in range(j-nWindowSize2, j+nWindowSize2+1):
                            RealBoodVessel3[m,n]= BloodVessel[m,n]     
                        
                    
    # Estimate real leakage and real leakage images               
    
    
    
    for i in range(0, RealBoodVessel.shape[0]):
        for j in range(0, RealBoodVessel.shape[1]):
            if( RealBoodVessel[i,j] > 0 ):
                RealBoodLeakageBinary[i,j]=0
                RealBoodLeakageGrey[i,j]=0
                
                
    for i in range(0, RealBoodVessel2.shape[0]):
        for j in range(0, RealBoodVessel2.shape[1]):
            if( RealBoodVessel2[i,j] > 0 ):
                RealBoodLeakageBinary2[i,j]=0
                RealBoodLeakageGrey2[i,j]=0                
                
    for i in range(0, RealBoodVessel3.shape[0]):
        for j in range(0, RealBoodVessel3.shape[1]):
            if( RealBoodVessel3[i,j] > 0 ):
                RealBoodLeakageBinary3[i,j]=0
                RealBoodLeakageGrey3[i,j]=0     
                
                
               
    RealBoodVessel = np.array(RealBoodVessel, dtype=np.uint8)                       
    RealBoodLeakageBinary = np.array(RealBoodLeakageBinary, dtype=np.uint8)       
    RealBoodLeakageGrey = np.array(RealBoodLeakageGrey, dtype=np.uint8)       

    RealBoodVessel2 = np.array(RealBoodVessel2, dtype=np.uint8)                       
    RealBoodLeakageBinary2 = np.array(RealBoodLeakageBinary2, dtype=np.uint8)       
    RealBoodLeakageGrey2 = np.array(RealBoodLeakageGrey2, dtype=np.uint8)       

    RealBoodVessel3 = np.array(RealBoodVessel3, dtype=np.uint8)                       
    RealBoodLeakageBinary3 = np.array(RealBoodLeakageBinary3, dtype=np.uint8)       
    RealBoodLeakageGrey3 = np.array(RealBoodLeakageGrey3, dtype=np.uint8)    
    
 
    return RealBoodVessel, RealBoodLeakageBinary, RealBoodLeakageGrey, RealBoodVessel2, RealBoodLeakageBinary2, RealBoodLeakageGrey,RealBoodVessel3, RealBoodLeakageBinary3, RealBoodLeakageGrey3       





def MakeColorImageForBloodLeakage( OriginalImageGrey, BloodLeakageBinary ):            

    OverlapImage = np.zeros((OriginalImageGrey.shape[0], OriginalImageGrey.shape[1],3),np.uint8)  
    
    #OverlapImage[:,:,1] = OriginalImageGrey[:,:]
    
    for i in range(0, OriginalImageGrey.shape[0]):
        for j in range(0, OriginalImageGrey.shape[1]):
            OverlapImage[i,j,0]= OriginalImageGrey[i,j]
            OverlapImage[i,j,1]= OriginalImageGrey[i,j]
            OverlapImage[i,j,2]= OriginalImageGrey[i,j]
            
            if(BloodLeakageBinary[i,j] > 127 ):
                OverlapImage[i,j,0]= 0
                OverlapImage[i,j,1]= 0
                OverlapImage[i,j,2]= 255
                   
    return OverlapImage
            

def RemoveOpticDisc( BloodLeakageBinary ):            

    BloodLeakageOriginal = BloodLeakageBinary.copy()  
    
    BloodLeakageCenter = BloodLeakageBinary.copy()
    
    for i in range(0, BloodLeakageCenter.shape[0]):
        for j in range(0, BloodLeakageCenter.shape[1]):
            if( i< (int)(BloodLeakageCenter.shape[0]/3) or i >(int)(BloodLeakageCenter.shape[0]*2/3) ):
                BloodLeakageCenter[i,j]=0
            if( j< (int)(BloodLeakageCenter.shape[1]/3) or j >(int)(BloodLeakageCenter.shape[1]*2/3) ):
                BloodLeakageCenter[i,j]=0
    
    ret,Ithres  = cv2.threshold(BloodLeakageCenter,127,255,0)
    # Find the largest contour and extract it
    im, contours, hierarchy = cv2.findContours(Ithres,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE )
    
    maxContour = 0
    for contour in contours:
        contourSize = cv2.contourArea(contour)
        if contourSize > maxContour:
            maxContour = contourSize
            maxContourData = contour
    
    # Create a mask from the largest contour
    mask = np.zeros_like(Ithres)
    cv2.fillPoly(mask,[maxContourData],1)
    
    # Use mask to crop data from original image
    finalImage = np.zeros_like(Ithres)
    finalImage[:,:] = np.multiply(BloodLeakageOriginal,mask)
    #cv2.imshow('final',finalImage)
    
    for i in range(0, BloodLeakageCenter.shape[0]):
        for j in range(0, BloodLeakageCenter.shape[1]):
            if( finalImage[i,j] >127 ):
                BloodLeakageOriginal[i,j]=0    
                   
    return BloodLeakageOriginal


def ConvertGreyImageToColorImage( aImage ):
                
    #ColorMap =[[255,255,255],[255,255,140],[[255,255,70],[255,255,0],[255,185,0],[255,127,0], [255,62,0],[255,22,0], [237,0,0],[197,0,0],[157,0,0],[117,0,0],[70,0,0]]
        
    #ColorMap =[[255,255,255],[255,255,140],[[255,255,70],[255,255,0],[255,185,0],[255,127,0], [255,62,0],[255,22,0], [237,0,0],[197,0,0],[157,0,0],[117,0,0],[70,0,0]]
    ColorMap =[[35,0,0],[70,0,0],[93,0,0],[117,0,0],[137,0,0],[157,0,0],[177,0,0],[197,0,0], [217,0,0], [237,0,0], \
               [255,11,0],[255,22,0], [255,42,0], [255,62,0],[255,92,0],[255,127,0],[255,150,0],[255,185,0], \
               [255,255,0], [255,255,35],[255,255,70],[255,255,105],[255,255,140],[255,255,180], [255,255,255]]

    aColorImage = np.zeros((aImage.shape[0], aImage.shape[1],3), np.uint8 )
        
    nMaxMap = np.size(ColorMap)/3 - 1
    nMaxValue = np.max(aImage)
    for i in range(aImage.shape[0]):
        for j in range(aImage.shape[1]):
            nIndex = int(float(aImage[i,j])*float(nMaxMap)/float(nMaxValue))
            aColorImage[i,j,0]=ColorMap[nIndex][2]
            aColorImage[i,j,1]=ColorMap[nIndex][1]
            aColorImage[i,j,2]=ColorMap[nIndex][0]                        
        
    return aColorImage       



def DisplayAndSaveImages20200709(sDirectoryForOutputImages, sImageName, aImage,  sPltTitle,  sOutExtension ):

    plt.title(sPltTitle)
    plt.imshow(aImage,cmap='gray')
    plt.show() 
        
    sOutfileName =sImageName 
   
    if sImageName.endswith('.tif')  :    
        sOutFileNameExt =  '-{}.tif'.format(sOutExtension)                            
        sOutfileName = sOutfileName.replace('.tif', sOutFileNameExt)
    elif sImageName.endswith('jpg')  :  
        sOutFileNameExt =  '-{}.jpg'.format(sOutExtension)                            
        sOutfileName = sOutfileName.replace('.jpg', sOutFileNameExt)                                    
    elif sImageName.endswith('.png') :  
        sOutFileNameExt =  '-{}.png'.format(sOutExtension)                            
        sOutfileName = sOutfileName.replace('.png', sOutFileNameExt)                                    
    elif sImageName.endswith('.bmp') :                                      
        sOutFileNameExt =  '-{}.bmp'.format(sOutExtension)                            
        sOutfileName = sOutfileName.replace('.bmp', sOutFileNameExt)
    else :                                      
        sOutFileNameExt =  '-{}.tif'.format(sOutExtension)                            
        sOutfileName = sOutfileName.replace('.tif', sOutFileNameExt)
            
    cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), aImage )               
 


def DisplayAndSaveImages(sDirectoryForOutputImages, sImageName, aImage,  sPltTitle,  sOutExtension='', bDisplay = True ):

    if( bDisplay == True ):
        plt.title(sPltTitle)
        plt.imshow(aImage,cmap='gray')
        plt.show() 
        
    sOutfileName =sImageName 
    
    if( len(sOutExtension) > 0 ):
   
        if sImageName.endswith('.tif')  :    
            sOutFileNameExt =  '-{}.tif'.format(sOutExtension)                            
            sOutfileName = sOutfileName.replace('.tif', sOutFileNameExt)
        elif sImageName.endswith('jpg')  :  
            sOutFileNameExt =  '-{}.jpg'.format(sOutExtension)                            
            sOutfileName = sOutfileName.replace('.jpg', sOutFileNameExt)                                    
        elif sImageName.endswith('.png') :  
            sOutFileNameExt =  '-{}.png'.format(sOutExtension)                            
            sOutfileName = sOutfileName.replace('.png', sOutFileNameExt)                                    
        elif sImageName.endswith('.bmp') :                                      
            sOutFileNameExt =  '-{}.bmp'.format(sOutExtension)                            
            sOutfileName = sOutfileName.replace('.bmp', sOutFileNameExt)
        else :                                      
            sOutFileNameExt =  '-{}.tif'.format(sOutExtension)                            
            sOutfileName = sOutfileName.replace('.tif', sOutFileNameExt)

                    
    cv2.imwrite(os.path.join(sDirectoryForOutputImages, sOutfileName), aImage )               
 
    
        

if __name__ == "__main__": 
    
    sMethod ='HE'
    sMethod ='ClaheAndHE'
    sMethod ='Clahe'
    sMethod ='ClaheBinarizationBoundary'
    sMethod ='ClaheBinarizationUsingOtsu'
    sMethod ='ClaheBinarizationUsingOtsuAndCropImage'
     
        
    if( sMethod == 'HE'):        
        LoadDirectorysAndEnhanceImagesUsingOpenCVAndSaveUsingHE( m_sDirectoryForInputImages, m_sDirectoryForOutputImagesHE )
    elif( sMethod == 'Clahe'):    
        LoadDirectorysAndEnhanceImagesUsingOpenCVAndSaveUsingClahe( m_sDirectoryForInputImages, m_sDirectoryForOutputImages )
    elif( sMethod == 'ClaheAndHE'):        
        LoadDirectorysAndEnhanceImagesUsingOpenCVAndSaveUsingClaheHE( m_sDirectoryForInputImages, m_sDirectoryForOutputImages, m_sDirectoryForOutputImagesHE )
    elif( sMethod == 'ClaheBinarizationUsingOtsu'):  
        m_nInvertImage = 1
        LoadDirectorysAndEnhanceImagesUsingOpenCVAndSaveUsingClaheMorphologyBinarizationUsingOtsu(  m_sDirectoryForInputImages, m_sDirectoryForOutputImages, m_sDirectoryForOutputImagesOtsuBinary, m_sDirectoryForOutputImagesOtsuBoundary, m_nInvertImage  )
    elif( sMethod == 'ClaheBinarizationBoundary'):    
        LoadDirectorysAndEnhanceImagesUsingOpenCVAndSaveUsingClaheMorphologyBinarization( m_sDirectoryForInputImages, m_sDirectoryForOutputImages, m_sDirectoryForOutputImagesOtsuBinary, m_sDirectoryForOutputImagesOtsuBoundary  )
    elif( sMethod == 'ClaheBinarizationUsingOtsuAndCropImage'):  
        m_nInvertImage = 1
        LoadDirectorysAndEnhanceImagesUsingOpenCVAndSaveUsingClaheMorphologyBinarizationUsingOtsuAndCropImages(  m_sDirectoryForInputImages, m_sDirectoryForOutputImages, m_sDirectoryForOutputImagesOtsuBinary, m_sDirectoryForOutputImagesContourBoundaryAndROI, m_sDirectoryForOutputImagesROIOnly, m_sDirectoryForOutputImagesROISquareOnly, m_nInvertImage  )
        
        
        
        