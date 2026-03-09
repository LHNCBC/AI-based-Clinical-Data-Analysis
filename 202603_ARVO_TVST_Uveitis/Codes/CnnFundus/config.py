# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 09:18:38 2019

@author: jongkim
"""

from keras import backend as K

class Config:
    def __init__(self):
        self.verbose = True

        self.network = 'UNet'
        self.bUseMultiClassClassifier = False            

        #These are for OpticDisc and Cup

        #These are for Uveitis
        #self.sMethod = "All"  #All abnormal lesions
        #self.sMethod = "Leakage"
        #self.sMethod = "BloodVessel"
        #self.sMethod = "BloodVessel"
        
        # setting for data augmentation        
        self.bUseAugmentation = True            #
        self.use_horizontal_flips = True
        self.use_vertical_flips = True
        self.rot_90 = True
        self.rois_to_estimate = 1
        self.nNbrOfAugmentations=1
    
    
        self.sMethod = "TrainTest"  #All abnormal lesions    
        self.sOperation = "traintest"  #All abnormal lesions    
        self.nNbrOfGPUs = 1        
        self.sCNNType = 'VGG16'
        self.nPyramidPoolModule = 0     # 20240320 add MAPNet after the backbone CNN (sCNNType)
        self.sOptimizer = 'sgd'
        self.nTransferLearning = 1
        self.nPreprocessInput = 0  
        self.nImageRows = 224 #48 #224 
        self.nImageColumns = 224 # 48 #224
        self.nImageChannels = 3 # 48 #224
        self.nNbrOfClasses = 2
        self.sClassAssignment = ''  #20200723
        self.sImageChannelType = 'BGR'      #'BGR', 'YCbCr' ,'HSV', 'HLS', 'LAB'
        
        self.nEpochs= 100 #1000 #200 #100
        self.nBatchSize= 20 #20 #10
        self.dDropoutRatio = 0.50
        self.dValidationSplit = 0.20
        self.dTrainValidDataGeneration = 0
        self.bBatchNormalization = 0 #Ture 
        
        self.sFilenameExtension=''
              
        self.nUseWeightedCategorialCrossEntropy = 0
        
        self.nAugmentationNumber = 0
        self.bAugmentationScale = True
        self.bAugmentationFlip = True
        self.bAugmentationRotation = True
        self.bAugmentationTranslation = False

        self.sFilePathForTrainImages  = ''
        self.sDirectoryForTrainImages = ''

        self.sFilePathForTestImages  = ''
        self.sDirectoryForTestImages = ''

        self.sFilePathForTrainImagesForTest  = ''

        self.sFilePathForTrainImagesAll  = 'X:/ARED/Jongwoo/OCT/TrainTestData/TrainDataAll.txt'


        self.nTestUsingFilePathForTrainImages = 0

        #selected
        self.sDirectoryForWeights = ''
        self.sDirectoryForWeights2 = ''
        self.sDirectoryForWeights3 = ''
        self.sVotingMethod = 'soft'
        self.sDirectoryForOutputRoot = ''
        self.sDirectoryForOutput = ''
        
        
        self.sDirectoryForTrainingResultsRoot = ''
        self.sDirectoryForTrainingResultsCurrent = ''
        self.sDirectoryForWeightsToRetrain = ''
        
        
        self.sLabelNames = ['CNV','DME','DRU','NOR']

        self.sWeight  = 'weights.h5'
        
        self.sHeatMapMethod = "" #'MULTIPLICATION_WITH_MASK', 'SUMMATION_WITH_MASK'
        self.fHeatMapThreshold = 0.25
        self.nHeatMapConvLayerLocation = -7
        self.nHeatMapNbrOfDenseLayers = 2
        
         
       
        self.smooth = 1.
        
        
        self.nImageLabelChannels = 1   #Channels of Ouput image 
        
        self.nBoundaryRangeToAdjust = 20 # To adjust boundary pixel values within the edge range that are too bright  with neighbor pixel values
        self.nUseClaheForInputImage = 0  # (0:DoNot Use CLAHE operator) (1:DoNot Use CLAHE operator) 
        self.bUseClaheForInputImage = False  #Use CLAHE operator 
        self.nClaheTileSize = 5

        self.bImageIntensityNormalization = False  #normalize image intensity by mean and standard deviation
        self.bImageIntensityNormalizationUseRGBChannel = True #normalize image intensity for each RGB channel or globally (R+G+B)/3
        
        self.nImageIntensityNormalization = 0  #normalize image intensity by mean and standard deviation
        self.nImageIntensityNormalizationUseRGBChannel = 0 #normalize image intensity for each RGB channel or globally (R+G+B)/3
        self.nImageIntensityNormalizationToMax1 = 0  #normalize image intensity by mean and standard deviation
        
        
        
        
        
        
        
        self.nNbrOfFilters = 64
        
        self.nImageRowsOriginal = 400
        self.nImageColumnsOriginal =400
        
        #Crop Images for training
        self.bUseCrop = True            
        self.nImageRowsToCrop = self.nImageRows
        self.nImageColumnsToCrop = self.nImageColumns
        self.nImageStrideToCrop = int(self.nImageRows/2) #=int(self.nImageRows/2)
        self.nImageStrideToCropToTest = self.nImageRows
        
        self.sTrainSetFold = ''  #to choose train set for n-fold
        
        
        #m_bUseDropoutAllLayers = False #True
        # =2 show the best performance (DoNot Change) (10: all layers=AHL, 5: 5 left half=HDL,  2: 2 left layers,  else: 2 left layers)
        self.nUseDropoutAllLayers = 2  #(10: all layers, 5: 5 left half,  2: 2 left layers,  else: 2 left layers)
        
        print('K.image_data_format(): ',K.image_data_format())
        
        '''
        if K.image_data_format() == 'channels_first':
            self.input_shape = (self.nImageChannels, self.nImageRows, self.nImageColumns)
        else:
            self.input_shape = (self.nImageRows, self.nImageColumns, self.nImageChannels)
        '''  
            
            
        
        
                
        
        '''
        # Use filename as input to segment blood leakage
        self.sDirectoryForGroundTruthImages =''
        self.sDirectoryForGroundTruthLabels=''
        
        self.sDirectoryForTrainImagesRoot = ''
        self.sDirectoryForTrainLabels = ''
        self.sDirectoryForTrainLabelsRoot = ''
        self.sDirectoryForSavingTrainCrop=''
        self.sFilePathForTrainingResult =''
        self.sWeight = 'weights.h5'
        self.fClassWeight = 1.0  
        self.sWeightforInitialization =''

        #Use for test and evaluation        
        self.sTestSetFold = ''  #to choose train set for n-fold
        self.sFilePathForTestImages  = ''  
        self.sDirectoryForTestImages = ''  #= self.sDirectoryForGroundTruthImages
        self.sDirectoryForTestLabels = ''  #=self.sDirectoryForGroundTruthLabels  
        self.sDirectoryForOverlapOutputImagesDisc = ''
        self.sDirectoryForOverlapOutputImagesCup = ''
        self.sDirectoryForOverlapOutputImagesLabelAll = ''
        
        
        #Use for testing
        self.sDirectoryForReferenceImages    = '' #directory for ground truth data
        self.sDirectoryForSavingTestResults   = '' #Main directory o save test results
        self.sDirectoryForOverlapOutputImages ='' # = cfg.sDirectoryForSavingTestResults + "LabelAllOverlap"
        self.sDirectoryForTestOutputImages    = ''   # = cfg.sDirectoryForSavingTestResults + "AllLabelsGrey"
        self.sDirectoryForOutputImages        = ''  # = cfg.sDirectoryForSavingTestResults + "AllLabels"
        
        self.sDirectoryForSavingTestResultsGrey    = ''  #self.sDirectoryForSavingTestResults + "Grey"
        self.sDirectoryForSavingTestResultsBinary  = ''  #self.sDirectoryForSavingTestResults + "Binary"
        self.sDirectoryForSavingTestResultsOverlap = ''  #self.sDirectoryForSavingTestResults + "Overlap"   
        self.sDirectoryForSavingTestResultsOverlapFull = ''  #self.sDirectoryForSavingTestResults + "Overlap"   
        self.sDirectoryForSavingTestResultsOverlapBinaryOnly = '' #self.sDirectoryForSavingTestResults + "OverlapBnaryOnly"
        self.sDirectoryForSavingTestResultsOverlapReferenceOnly = '' #self.sDirectoryForSavingTestResults + "Reference Only"
        self.sDirectoryForSavingTestResultsOverlapClahe = ''  #self.sDirectoryForSavingTestResults + "Overlap on Clahe image"           
        self.sDirectoryForSavingTestResultsOverlapClaheFull = ''  #self.sDirectoryForSavingTestResults + "Overlap on Clahe image fully"           
        self.sDirectoryForSavingTestResultsOverlapBinaryOnlyClahe = '' #self.sDirectoryForSavingTestResults + "OverlapBnaryOnly on Clahe image"
        self.sDirectoryForSavingTestResultsOverlapReferenceOnlyClahe = '' #self.sDirectoryForSavingTestResults + "OverlapBnaryOnly on Clahe image"
        
        self.sDirectoryForSavingTestResultsGreyMethod    = ''  #self.sDirectoryForSavingTestResults + "Grey"
        self.sDirectoryForSavingTestResultsBinaryMethod  = ''  #self.sDirectoryForSavingTestResults + "Binary"
        self.sDirectoryForSavingTestResultsOverlapMethod = ''  #self.sDirectoryForSavingTestResults + "Overlap"   
        self.sDirectoryForSavingTestResultsOverlapFullMethod = ''  #self.sDirectoryForSavingTestResults + "Overlap"   
        self.sDirectoryForSavingTestResultsOverlapBinaryOnlyMethod = '' #self.sDirectoryForSavingTestResults + "OverlapBnaryOnly"
        self.sDirectoryForSavingTestResultsOverlapReferenceOnlyMethod = '' #self.sDirectoryForSavingTestResults + "Reference Only"
        self.sDirectoryForSavingTestResultsOverlapClaheMethod = ''  #self.sDirectoryForSavingTestResults + "Overlap on Clahe image"           
        self.sDirectoryForSavingTestResultsOverlapClaheFullMethod = ''  #self.sDirectoryForSavingTestResults + "Overlap on Clahe image fully"           
        self.sDirectoryForSavingTestResultsOverlapBinaryOnlyClaheMethod = '' #self.sDirectoryForSavingTestResults + "OverlapBnaryOnly on Clahe image"
        self.sDirectoryForSavingTestResultsOverlapReferenceOnlyClaheMethod = '' #self.sDirectoryForSavingTestResults + "OverlapBnaryOnly on Clahe image"
        
        self.sDirectoryForSavingTestResultsCropTemp = ''  #self.sDirectoryForSavingTestResults + "Crop"
        self.sFileNameToSaveTestProcess=''
        self.bForEvaluationUseGroundTruthData = True     #(True= Generate output image and comparison results with ground-truth data)
        #self.bForEvaluationUseGroundTruthData = False     #(False= Generate output image only, no comparison results with ground-truth data)
        
        
        
        
        #self.sDirectoryForSaveTestResultsPrediction = '' #=self.sDirectoryForSavingTestResultsGrey
                  
        #Use for evaluation
       # self.sDirectoryForTestOutputImages    = ''   #must remove
       # self.sDirectoryForOverlapOutputImages = ''   #must remove
       ''' 
        
        
        
        self.nThresholdForBinarization = 127
        #self.nThresholdForBinarizationForTest = self.nThresholdForBinarization 
        self.nThresholdForBinarizationForTest = -1 #Use Otsu threhsold
        
        self.bUseSlurm = False
        

        #-----------------------------------------------
        # HeatMap Parameters
        #-----------------------------------------------
        # Eight different sHeatMapMethod
        #-----------------------------------------------
        #'ORIGINAL':
        #'POSITIVE':
        #'NEGATIVE':
        #'NEGATIVE_NORMALIZED_ON_NEG_MAX':
        #'NEGATIVE_NORMALIZED_ON_POS_MAX':
        #'SUMMATION':
        #'SUMMATION_WITH_MASK':
        #'MULTIPLICATION_WITH_MASK':
        #-----------------------------------------------      
        self.sHeatMapMethod = 'MULTIPLICATION_WITH_MASK'
        
        #Threshold to remove values with less than nPositiveThreshold
        self.nPositiveThreshold = 0.25
        
        #Convolutional Layer to use to draw the heatmap.
        self.nConvolutionalLayer  = -7
        
        self.sDirectoryForSavingHeatMapImage = ''
        self.sDirectoryForSavingHeatMapImageWithOriginalImage = ''
        self.sDirectoryForSavingHeatMapImageWithHeatMapROI = ''
        self.sDirectoryForSavingHeatMapImageWithCroppedROI = ''
        self.sDirectoryForSavingHeatMapImageWithCroppedROISquare = ''
        
        self.sClassLabelNames = []
        
        # 20230427 to print all test data output
        self.nPrintAllTestOutputs = 0
      