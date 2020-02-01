#### Feature Specific Reactivation ####
# assumes all required files are in the working directory
#
# Encoding model neural activity predicitions are generated/saved in section "Encoding Model Brain Activity Predictions for Recall and Recogntion"
# Predictions are used to generate reactivation rank values in section "Reactivation Rank Measure"
#
# ROIs divided into 'chunks' because doing all ROIs at once takes too long (see "Get Reactivation Results by ROI 'chunk'" section)
# 2-8 hours per ROI 'chunk'
#
# To generate/save the reactivation values simply run the whole script. If you just want to get data for one ROI chunk, modify 'useChunks'.
#
# Encoding model vertex activity predictions for recall (predActVisSub) and recognition (predActPrbSub) are saved in 
# "predActLvl<feature level>Roi<ROI chunk>.RData". The data is a list (names = subject) of 2-D arrays (row = image (alpha order), column = vertex)
#
# Reactivation ranks for recall (rankTriSubPrbROI) and recognition (rankTriSubPrbROI) are saved in 
# "reacRankLvl<feature level>Roi<ROI chunk>.RData". The data is a list (names = ROI) of 2-D arrays (row = subject, column = trial/image (alpha order))




#### Dependencies ####

library(nnlasso)
library(R.matlab)



#### Functions ####

## log transform features
logMod = function(mat) {
  ix = which(mat<0)
  mat = log(abs(mat) + 1)
  mat[ix] = mat[ix]*-1
  mat
}



#### Global Variables ####

smooth = T # use spatially smoothed brain data
subjectsEx = c(1002,1006,1007,1008,1009,1010,1011,1012,1014,1016,1019,1020,1022,1024,1025,1026,1027,
               1028,1030,1031,1032,1033,1034,1035,1036,1037,1038) # subject numbers
FeatLvlNames = c('L1','L2','L3','L4','L5','FC') # feature level names

# get groups of Freesurfer ROIs
# because it takes so long to get the encoding model predictions I did it in 'chunks' of ROIs
load(file='ROINamesDiv.RData')
ROILvlNames = c('LowOc','HighOc','Tmprl','Prtl','Front','Other') # names of 'chunks'

# load image names
# nBackImagesAll = all image names
# list (names = subjects) of vectors containing image names:
# nBackImages = encoding image for each trial
# recogImages = cued recall images for each image and time point (sorted alphabetically)
# recogProbes = recognition probe images for each image and time point (sorted alphabetically)
load(file=paste0("imageNames.RData"))



#### Get Reactivation Results by ROI 'chunk' #####
# there are 6 chucks:
# 1 = (LowOc) occipital early visual
# 2 = (HighOc) occipital late visual
# 3 = (Tmprl) temporal
# 4 = (Prtl) parietal
# 5 = (Front) frontal
# 6 = (Other) every other ROI
useChunks = 1

ROILvl = 1
for (ROILvl in useChunks) {
  print(ROILvlNames[ROILvl])
  tROINames = ROINamesDiv[[ROILvl]]
  
  # load ROI name of each vertex (roiNodeNames)
  # list (names = subjects) of vectors containing roi names
  load(file=paste0("roiNodeNames",ROILvlNames[ROILvl],".RData"))
  
  # load movie neural data (movie2VntrDatSep, movie3VntrDatSep)
  # list (names = subjects) of matrices (row = time point, column = brain vertex)
  # data z-scored for each vertex
  load(file=paste0("movie",ROILvlNames[ROILvl],".RData"))
  
  # load encoding neural data (nBackVntrDat)
  # list (names = subjects) of matrices (row = trial, column = brain vertex)
  # data z-scored for each vertex
  load(file=paste0("nback",ROILvlNames[ROILvl],".RData"))
  
  # load recall (recogVis) and recognition (recogPrb) neural data
  # list (names = subjects) of matrices (row = image (alphabetically ordered), column = brain vertex)
  load(paste0("recogVis",ROILvlNames[ROILvl],".RData"))
  load(paste0("recogPrb",ROILvlNames[ROILvl],".RData"))
  
  
  
  #### Encoding Model Brain Activity Predictions for Recall and Recogntion ####
  # generate/save encoding model predictions of recall/recognition vertex activity
  
  ## feature level CV and lambda variables ##
  featLvls = c(1,3,5,6)         # DNN feature levels to use. Extracted from VGG16.
                                # The 'levels' are the last convolution or fully connected layers 
                                # in each set of layers divided by the max pooling layers.
                                # level 1 = conv layer 2
                                # level 2 = conv layer 4
                                # level 3 = conv layer 7
                                # level 4 = conv layer 10
                                # level 5 = conv layer 13
                                # level 6 = fully connected layer 3
  folds = 3                     # CV folds
  foldSize = 90/folds           # CV fold size (over 90 images)
  lambdas = 10^seq(3,-3,by=-.5) # lambda values to use for the lasso regression
  
  
  ## generate encoding model predictions of recall/recognition vertex activity for the target 
  ## feature levels and save them
  FeatLvl = 1
  for (FeatLvl in featLvls) {
    print(ROILvlNames[ROILvl])
    print(FeatLvlNames[FeatLvl])
    
    # load DNN feature activation data for encoding images (rows = images ordered alphabetically) 
    # and movies (rows = timepoints hemodynamically smoothed). Columns = features.
    if (FeatLvl<6) {
      # DNN level (Conv)
      lvl = FeatLvl
      imFeatPics = readMat(paste0("DNNL",toString(lvl),"ImMat3X3.mat"))
      imFeatPics = imFeatPics[['DNNImMat']]
      imFeatMean = mean(imFeatPics)                              ### log mod ### 
      imFeatPics = logMod(imFeatPics)                            ### log mod ###
      imFeatPics = scale(imFeatPics)
      
      imFeatMix1 = readMat(paste0("DNNL",toString(lvl),"Mix1Mat3X3HR.mat"))
      imFeatMix1 = imFeatMix1[['imFeatMix1']]
      imFeatMix1 = imFeatMix1*(imFeatMean/mean(imFeatMix1))      ### log mod ###      
      imFeatMix1 = logMod(imFeatMix1)                            ### log mod ###
      imFeatMix1 = imFeatMix1[8:nrow(imFeatMix1),]
      imFeatMix1 = scale(imFeatMix1)
      
      imFeatMix2 = readMat(paste0("DNNL",toString(lvl),"Mix2Mat3X3HR.mat"))
      imFeatMix2 = imFeatMix2[['imFeatMix2']]
      imFeatMix2 = imFeatMix2*(imFeatMean/mean(imFeatMix2))      ### log mod ###   
      imFeatMix2 = logMod(imFeatMix2)                            ### log mod ###
      imFeatMix2 = imFeatMix2[8:nrow(imFeatMix2),]
      imFeatMix2 = scale(imFeatMix2)
    } else {
      # DNN level (FC)
      imFeatPics = readMat(paste0("DNNFC3ImMat.mat"))
      imFeatPics = imFeatPics[['DNNImMat']]
      imFeatMean = mean(imFeatPics)                              ### log mod ### 
      imFeatPics = logMod(imFeatPics)                            ### log mod ###
      imFeatPics = scale(imFeatPics)
      
      imFeatMix1 = readMat(paste0("DNNFC3Mix1MatHR.mat"))
      imFeatMix1 = imFeatMix1[['imFeatMix1']]
      imFeatMix1 = imFeatMix1*(imFeatMean/mean(imFeatMix1))      ### log mod ###      
      imFeatMix1 = logMod(imFeatMix1)                            ### log mod ###
      imFeatMix1 = imFeatMix1[8:nrow(imFeatMix1),]
      imFeatMix1 = scale(imFeatMix1)
      
      imFeatMix2 = readMat(paste0("DNNFC3Mix2MatHR.mat"))
      imFeatMix2 = imFeatMix2[['imFeatMix2']]
      imFeatMix2 = imFeatMix2*(imFeatMean/mean(imFeatMix2))      ### log mod ###   
      imFeatMix2 = logMod(imFeatMix2)                            ### log mod ###
      imFeatMix2 = imFeatMix2[8:nrow(imFeatMix2),]
      imFeatMix2 = scale(imFeatMix2)
    }
    
    # number of features for the current DNN lvl
    featN = ncol(imFeatPics)
    
    # initialize data structures
    predActVisSub = list()
    predActPrbSub = list()
    sseVoxSub = list()
    rankTriSubVis = array(NA,c(length(subjectsEx),nrow(recogVis[[1]])))
    rownames(rankTriSubVis) = subjectsEx
    rankTriSubPrb = array(NA,c(length(subjectsEx),nrow(recogPrb[[1]])))
    rownames(rankTriSubPrb) = subjectsEx
    
    ## generate encoding model predictions for each subject
    sub = 1002
    f = 5
    vox = 1
    for (sub in subjectsEx) {
      print(sub)
      
      # get brain vertex activity
      dat1 = movie2VntrDatSep[[toString(sub)]]
      dat1 = dat1[8:nrow(dat1),]
      dat2 = movie3VntrDatSep[[toString(sub)]]
      dat2 = dat2[8:nrow(dat2),]
      dat3 = nBackVntrDat[[toString(sub)]]
      datPrb = scale(recogPrb[[toString(sub)]])
      datVis = scale(recogVis[[toString(sub)]])
      
      # get trial-by-trial image order and features for the subject during encoding
      dat3Names = nBackImages[[toString(sub)]]
      tempIX = sapply(dat3Names,function (x) {which(nBackImagesAll==x)})
      imFeatTemp = imFeatPics[tempIX,]
      imgOrder = ceiling(tempIX/2)
      
      # get trial-by-trial cued images and associated features for the subject during recall
      datVisNames = recogImages[[toString(sub)]]
      datVisNames = datVisNames[(1:90)*16]
      tempIX = sapply(datVisNames,function (x) {which(nBackImagesAll==x)})
      imFeatTempVis = imFeatPics[tempIX,]
      
      # get trial-by-trial probe images and associated features for the subject during recognition
      datPrbNames = recogProbes[[toString(sub)]]
      datPrbNames = datPrbNames[(1:90)*16]
      tempIX = sapply(datPrbNames,function (x) {which(nBackImagesAll==x)})
      imFeatTempPrb = imFeatPics[tempIX,]
      
      # concatenate movie and encoding data for training
      dat = rbind(dat1,dat2,dat3) # brain vertex activity
      imFeatTempAll = rbind(imFeatMix1,imFeatMix2,imFeatTemp) # DNN feature activity
      trainPred = imFeatTempAll
      testPredPrb = imFeatTempPrb
      testPredVis = imFeatTempVis
      nbackRowStart = nrow(trainPred)-nrow(imFeatTemp)

      # initialize data structures
      sseVoxSub[[toString(sub)]] = 1:ncol(dat1) # prediction SSE per vertex
      predActVis = array(NA,c(nrow(datPrb),ncol(datPrb))) # vertex activity prediction recall
      predActPrb = array(NA,c(nrow(datPrb),ncol(datPrb))) # vertex activity prediction recog

      
      ## generate encoding model predictions for each brain vertex
      for (vox in 1:ncol(dat)) {
        if (vox%%100==0) {
          print(vox)
        }
        
        datVox = dat[,vox]
        
        # only use top 100 DNN features that correlate with vertex activity (too slow otherwise)
        corIX = sort.int(cor(matrix(datVox,nrow=length(datVox),ncol=1),trainPred),index.return=T,decreasing=T)$ix[1:100]
        
        # initialize data structures
        sseFold = 1:folds
        
        ## cross validation
        cvImRand = sample(1:90) # randomly select test images for each fold
        for (f in 1:folds) {
          # get test and train data indexes for the fold
          fIms = cvImRand[(foldSize*(f-1)+1):(foldSize*(f))]
          ImgsTest = imgOrder%in%fIms
          fIXTest = nbackRowStart+which(ImgsTest)
          fIXTrain = c(1:nbackRowStart,nbackRowStart+which(!ImgsTest))
          
          # train non-negative lasso regression to predict brain activity from DNN features of the image/video.
          # uses multiple lambdas
          g1 = nnlasso(trainPred[fIXTrain,corIX],datVox[fIXTrain],family="normal",lambda=NA,
                       intercept=TRUE,normalize=TRUE,tau=1,tol=1e-3,maxiter=10000,nstep=5,min.lambda=1e-4,
                       eps=1e-6,path=T,SE=F)

          # select best lambda (using held out encoding data)
          sseTemp = 1:length(g1$lambdas)
          for (l in 1:length(g1$lambdas)) {
            prednn = rep(g1$beta0[l],length(fIXTest))+rowSums(matrix(g1$coef[l,],nrow=length(fIXTest),ncol=length(corIX),byrow=T)*trainPred[fIXTest,corIX])
            sseTemp[l] = sum((datVox[fIXTest]-prednn)^2)/sum((datVox[fIXTest]-mean(datVox[fIXTest]))^2)
          }
          lmdaIX = which.min(sseTemp)
          sseFold[f] = min(sseTemp)
          
          # get prediction for recall brain vertex activity
          predActVis[fIms,vox] = rep(g1$beta0[lmdaIX],foldSize) + 
            rowSums(matrix(g1$coef[lmdaIX,],nrow=foldSize,ncol=length(corIX),byrow=T)*testPredVis[fIms,corIX])
          
          # get prediction for recognition brain vertex activity
          predActPrb[fIms,vox] = rep(g1$beta0[lmdaIX],foldSize) + 
            rowSums(matrix(g1$coef[lmdaIX,],nrow=foldSize,ncol=length(corIX),byrow=T)*testPredPrb[fIms,corIX])
        }
        sseVoxSub[[toString(sub)]][vox] = mean(sseFold)
      }
      # recall and recognition brain activity predictions for the subject
      predActVisSub[[toString(sub)]] = predActVis
      predActPrbSub[[toString(sub)]] = predActPrb
    
      voxIX = 1:ncol(dat)
      
      # generate/print trial-average reactivation (rank) results during recall for the current ROI:feature-level:subject
      predCor = cor(t(predActVisSub[[toString(sub)]][,voxIX]),t(datVis[,voxIX]))
      for (trial in 1:nrow(datVis)) {
        rankTriSubVis[toString(sub),trial] = which(sort.int(predCor[trial,],decreasing=T,index.return=T)$ix==trial)
      }
      print(45.5-mean(rankTriSubVis[toString(sub),]))
      
      # generate/print trial-average reactivation (rank) results during recog for the current ROI:feature-level:subject
      predCor = cor(t(predActPrbSub[[toString(sub)]][,voxIX]),t(datPrb[,voxIX]))
      for (trial in 1:nrow(datPrb)) {
        rankTriSubPrb[toString(sub),trial] = which(sort.int(predCor[trial,],decreasing=T,index.return=T)$ix==trial)
      }
      print(45.5-mean(rankTriSubPrb[toString(sub),]))
    }
    
    # save the predicted brain activity for the current ROI:feature-level
    save(predActPrbSub,predActVisSub,sseVoxSub,
         file=paste0("predActLvl",FeatLvlNames[FeatLvl],"Roi",ROILvlNames[ROILvl],".RData"))
  }
  
  
  
  #### Reactivation Rank Measure ####
  # generates reactivation rank as described in the paper
  # uses the saved recall/recognition encoding model brain activity predictions
  
  ## feature level  ##
  featLvls = c(1,3,5,6) # DNN feature levels to use. Extracted from VGG16.
                        # The 'levels' are the last convolution or fully connected layers 
                        # in each set of layers divided by the max pooling layers.
                        # level 1 = conv layer 2
                        # level 2 = conv layer 4
                        # level 3 = conv layer 7
                        # level 4 = conv layer 10
                        # level 5 = conv layer 13
                        # level 6 = fully connected layer 3
  
  # load recall (vis) and recognition (prb) neural data
  # list (names = subjects) of matrices (row = image (alphabetically ordered), column = brain vertex)
  load(paste0("recogVis",ROILvlNames[ROILvl],".RData"))
  load(paste0("recogPrb",ROILvlNames[ROILvl],".RData"))
  
  # get FreeSurfer ROI names in the current larger ROI
  uniqueROIs = sort(unique(ROINodeNames[[1]]))
  
  # chance rank
  rankChance = mean(1:90)
  
  ## generate/save reactivation rank for each feature level
  FeatLvl = 1
  for (FeatLvl in featLvls) {
    print(ROILvlNames[ROILvl])
    print(FeatLvlNames[FeatLvl])
    
    # load activity prediction
    datVars = load(file=paste0("predActLvl",FeatLvlNames[FeatLvl],"Roi",ROILvlNames[ROILvl],".RData"))
    
    # initialize data structures
    rankTriSubPrbROI = list()
    rankTriSubVisROI = list()
    corDifPrbVsOthrSubVisROI = list()
    TValROI = rep(NA,length(uniqueROIs))
    names(TValROI) = uniqueROIs
    
    ## get reactivation ranks for each FreeSurfer ROI
    roi = uniqueROIs[4]
    sub = 1002
    countROI = 0
    for (roi in uniqueROIs) {
      countROI = countROI + 1
      print(roi)

      # initialize data structures
      rankTriSubPrb = array(NA,c(length(subjectsEx),nrow(recogPrb[[1]])))
      rownames(rankTriSubPrb) = subjectsEx
      rankTriSubVis = rankTriSubPrb
      
      ## get reactivation ranks for each subject
      sub = subjectsEx[1]
      count = 0
      for (sub in subjectsEx) {
        count = count + 1
        print(sub)

        # get column vector index for current ROI
        ROIIx = which(ROINodeNames[[toString(sub)]]==roi)
        
        # get recall (vis) and recognition (prb) brain data for subject
        datVis = scale(recogVis[[toString(sub)]])
        datPrb = scale(recogPrb[[toString(sub)]])
        
        # get encoding model predictions
        predDatVis = eval(parse(text=paste0(datVars[2],"[[toString(sub)]]")))
        predDatPrb = eval(parse(text=paste0(datVars[1],"[[toString(sub)]]")))

        # get number of vertices and images
        voxN = ncol(datPrb)
        imN = nrow(datPrb)
        
        # limit vertices to those in the target ROI
        voxIX = ROIIx
        
        # generate/print trial-average reactivation (rank) results during recall for the current ROI:feature-level:subject
        predCor = cor(t(datVis[,voxIX]),t(predDatVis[,voxIX]))
        for (trial in 1:nrow(datPrb)) {
          rankTriSubVis[toString(sub),trial] = which(sort.int(predCor[trial,],decreasing=T,index.return=T)$ix==trial)
        }
        
        # generate/print trial-average reactivation (rank) results during recognition for the current ROI:feature-level:subject
        predCor = cor(t(datPrb[,voxIX]),t(predDatPrb[,voxIX]))
        for (trial in 1:nrow(datPrb)) {
          rankTriSubPrb[toString(sub),trial] = which(sort.int(predCor[trial,],decreasing=T,index.return=T)$ix==trial)
        }
      }
      rankTriSubVisROI[[toString(roi)]] = rankChance - rankTriSubVis
      rankTriSubPrbROI[[toString(roi)]] = rankChance - rankTriSubPrb
    }
    
    # save rank data
    save(rankTriSubPrbROI,rankTriSubVisROI,
         file=paste0("reacRankLvl",FeatLvlNames[FeatLvl],"Roi",ROILvlNames[ROILvl],".RData"))
  }
}


