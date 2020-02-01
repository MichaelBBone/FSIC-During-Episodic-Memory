#### Encoding-Decoding and FSIC (brain activity simulation) ####
# assumes all required files are in the working directory
#
# instructions:
# - run code in "Dependances" and "Functions" (download any missing packages)
#
# - if you want to generate simulated decoding accuracy, run the "Simulation" section (can adjust variables under "Parameters")
# - takes about 2-3 min per simulated subject (400-600 min for 200 subs)
# - results will be saved in "simRankTest1.RData" by default
#
# - if you want to recreate the results in Figure 3 (mean accuracy and FSIC stats), first run "Results: Data" section
# - be sure to load "simRankIdenticalMem200Sub.RData" in the "data to use" section, which contains the simulated decoding accuracy assuming 
#   identical trial-by-trial memory loss across feature levels
# - you can also load other data such as "simRankIndependantMem200Sub.RData", which contains the simulated decoding accuracy assuming 
#   independent trial-by-trial memory loss across feature levels, or your own simulated data
#
# - if you want to generate decoding accuracy stats (Figure 3a) run the "Results: Mean Rank" section (after running the Results: Data" section)
# - takes about 2 hours with 200 subjects and 1000 bootstrap iterations
# - you can load the data used in the paper (and skip the 2 hour wait) by loading "simRankIdenticalMem200Sub_MeanStats.RData"
#   under "load pre-calculated values"
#
# - if you want to generate FSIC stats (Figure 3b) run the "Results: FSIC" section (after running the Results: Data" section)
# - takes about 2.5 hours with 200 subjects and 1000 bootstrap iterations
# - you can load the data used in the paper (and skip the 2.5 hour wait) by loading "simRankIdenticalMem200Sub_FSICStats.RData"
#   under "load pre-calculated values"



#### Dependances ####

library(abind)
library(nnlasso)
library(R.matlab)
library(lme4)
library(ggplot2)



#### Functions ####

# log-transform matrix data #
logMod = function(mat) {
  ix = which(mat<0)
  mat = log(abs(mat) + 1)
  mat[ix] = mat[ix]*-1
  mat
}



#### Simulation ####
# takes 2-3 min per simulated subject (400-600 min for 200 subs)

## parameters ##
sameMemAccAllLvl = T # set to T (F) for identical (independent) feature memory retention across feature levels per trial
NSim = 200 # number of simulated subjects
SNRSubLvls = c(.15,.25,.35) # three SNR values (selected randomly per subject)
memVivLvls = c(.05,.6) # feature memory retention proportion range (random value selected in range for each trial)
##


simRankPrb = array(NA,c(NSim,8,4))
simRankVis = array(NA,c(NSim,8,4))
simRankPrbTri = array(NA,c(NSim,8,4,90))
simRankVisTri = array(NA,c(NSim,8,4,90))
simCorPrb = array(NA,c(NSim,8,4))
simCorVis = array(NA,c(NSim,8,4))
memViv = array(NA,c(NSim,4,90))
simImg = array(NA,c(NSim,90))
sim = 1
for (sim in 1:NSim) {
  
  ## create sim training data ##
  simVoxN = 100
  simVoxFrac = .2
  simVoxIX = list()
  imIX = (1:90)*2-sample(c(0,1),90,replace=T)
  simImg[sim,] = imIX
  
  SNRPrb = SNRSubLvls[sample(1:3,1)]
  
  if (sameMemAccAllLvl) {
    memViv[sim,1,] = runif(90,memVivLvls[1],memVivLvls[2])
    memViv[sim,2,] = memViv[sim,1,]
    memViv[sim,3,] = memViv[sim,1,]
    memViv[sim,4,] = memViv[sim,1,]
  } else {
    memViv[sim,1,] = runif(90,memVivLvls[1],memVivLvls[2])
    memViv[sim,2,] = runif(90,memVivLvls[1],memVivLvls[2])
    memViv[sim,3,] = runif(90,memVivLvls[1],memVivLvls[2])
    memViv[sim,4,] = runif(90,memVivLvls[1],memVivLvls[2])
  }
  
  # get CNN feature data for voxel activity simulation
  lvl = 'L1'
  count = 0
  for (lvl in c('L1','L3','L5','FC3')) {
    count = count + 1
    
    if (lvl=='FC3') {
      imFeat = readMat(paste0("DNNFC3ImMat.mat"))
      imFeat = imFeat[['DNNImMat']]
      imFeatMean = mean(imFeat)                              ### log mod ### 
      imFeat = logMod(imFeat)                            ### log mod ###
      imFeat = scale(imFeat)
      simVoxIX[[count]] = sample(1:ncol(imFeat))[1:simVoxN]
      simDatPics = cbind(simDatPics,imFeat[,simVoxIX[[count]]])
      
      imFeat = readMat(paste0("DNNFC3Mix1MatHR.mat"))
      imFeat = imFeat[['imFeatMix1']]
      imFeat = imFeat*(imFeatMean/mean(imFeat))      ### log mod ###      
      imFeat = logMod(imFeat)                            ### log mod ###
      imFeat = imFeat[8:nrow(imFeat),]
      imFeat = scale(imFeat)
      simDatMix1 = cbind(simDatMix1,imFeat[,simVoxIX[[count]]])
      
      imFeat = readMat(paste0("DNNFC3Mix2MatHR.mat"))
      imFeat = imFeat[['imFeatMix2']]
      imFeat = imFeat*(imFeatMean/mean(imFeat))      ### log mod ###   
      imFeat = logMod(imFeat)                            ### log mod ###
      imFeat = imFeat[8:nrow(imFeat),]
      imFeat = scale(imFeat)
      simDatMix2 = cbind(simDatMix2,imFeat[,simVoxIX[[count]]])
    } else {
      imFeat = readMat(paste0("DNN",lvl,"ImMat3X3.mat"))
      imFeat = imFeat[['DNNImMat']]
      imFeatMean = mean(imFeat)                              ### log mod ### 
      imFeat = logMod(imFeat)                            ### log mod ###
      imFeat = scale(imFeat)
      simVoxIX[[count]] = sample(1:ncol(imFeat))[1:simVoxN]
      if (count==1) {
        simDatPics = imFeat[,simVoxIX[[count]]]
      } else {
        simDatPics = cbind(simDatPics,imFeat[,simVoxIX[[count]]])
      }
      
      imFeat = readMat(paste0("DNN",lvl,"Mix1Mat3X3HR.mat"))
      imFeat = imFeat[['imFeatMix1']]
      imFeat = imFeat*(imFeatMean/mean(imFeat))      ### log mod ###      
      imFeat = logMod(imFeat)                            ### log mod ###
      imFeat = imFeat[8:nrow(imFeat),]
      imFeat = scale(imFeat)
      if (count==1) {
        simDatMix1 = imFeat[,simVoxIX[[count]]]
      } else {
        simDatMix1 = cbind(simDatMix1,imFeat[,simVoxIX[[count]]])
      }
      
      imFeat = readMat(paste0("DNN",lvl,"Mix2Mat3X3HR.mat"))
      imFeat = imFeat[['imFeatMix2']]
      imFeat = imFeat*(imFeatMean/mean(imFeat))      ### log mod ###   
      imFeat = logMod(imFeat)                            ### log mod ###
      imFeat = imFeat[8:nrow(imFeat),]
      imFeat = scale(imFeat)
      if (count==1) {
        simDatMix2 = imFeat[,simVoxIX[[count]]]
      } else {
        simDatMix2 = cbind(simDatMix2,imFeat[,simVoxIX[[count]]])
      }
    }
  }
  simDatPics = simDatPics[imIX,]
  
  # create feature memory loss matrix
  for (lvl in 1:4) {
    if (lvl==1) {
      lvlCol = length(simVoxIX[[lvl]])
      lvlRow = nrow(simDatPics)
      MemMat = matrix(t(sapply(memViv[sim,lvl,],function (x) {sample(c(0,1),lvlCol*lvlRow,replace=T,prob=c(1-x,x))})),nrow=lvlRow,ncol=lvlCol)
    } else {
      lvlCol = length(simVoxIX[[lvl]])
      lvlRow = nrow(simDatPics)
      MemMat = cbind(MemMat,matrix(t(sapply(memViv[sim,lvl,],function (x) {sample(c(0,1),lvlCol*lvlRow,replace=T,prob=c(1-x,x))})),nrow=lvlRow,ncol=lvlCol))
    }
  }
  
  # add noise and feature memory loss to testing (retrival) sim data
  simDatPrbNoise = simDatPics*SNRPrb + matrix(rnorm(length(simDatPics),0,1),nrow=nrow(simDatPics),ncol=ncol(simDatPics))*(1-SNRPrb)
  simDatPrbNoise = cbind(simDatPrbNoise,simDatPics*SNRPrb + matrix(rnorm(length(simDatPics),0,1),nrow=nrow(simDatPics),ncol=ncol(simDatPics))*(1-SNRPrb))

  simDatVivNoise = simDatPics*MemMat*SNRPrb + matrix(rnorm(length(simDatPics),0,1),nrow=nrow(simDatPics),ncol=ncol(simDatPics))*(1-SNRPrb)
  simDatVivNoise = cbind(simDatVivNoise,simDatPics*MemMat*SNRPrb + matrix(rnorm(length(simDatPics),0,1),nrow=nrow(simDatPics),ncol=ncol(simDatPics))*(1-SNRPrb))

  # add noise to training sim data (movie and encode)
  simDatPicsNoise = simDatPics*SNRPrb + matrix(rnorm(length(simDatPics),0,1),nrow=nrow(simDatPics),ncol=ncol(simDatPics))*(1-SNRPrb)
  simDatPicsNoise = cbind(simDatPicsNoise,simDatPics*SNRPrb + matrix(rnorm(length(simDatPics),0,1),nrow=nrow(simDatPics),ncol=ncol(simDatPics))*(1-SNRPrb))
  
  simDatMix1Noise = simDatMix1*SNRPrb + matrix(rnorm(length(simDatMix1),0,1),nrow=nrow(simDatMix1),ncol=ncol(simDatMix1))*(1-SNRPrb)
  simDatMix1Noise = cbind(simDatMix1Noise,simDatMix1*SNRPrb + matrix(rnorm(length(simDatMix1),0,1),nrow=nrow(simDatMix1),ncol=ncol(simDatMix1))*(1-SNRPrb))
  
  simDatMix2Noise = simDatMix2*SNRPrb + matrix(rnorm(length(simDatMix2),0,1),nrow=nrow(simDatMix2),ncol=ncol(simDatMix2))*(1-SNRPrb)
  simDatMix2Noise = cbind(simDatMix2Noise,simDatMix2*SNRPrb + matrix(rnorm(length(simDatMix2),0,1),nrow=nrow(simDatMix2),ncol=ncol(simDatMix2))*(1-SNRPrb))
  
  
  ## train encoding model and predict (simulated) neural activity for each feature level ##
  lvlFeat = 3
  for (lvlFeat in 1:4) {
    print(paste0('subject: ',sim))
    print(paste0('feature level: ',lvlFeat))
    
    # get feature data
    lvlList = c('1','3','5')
    if (lvlFeat<4) {
      imFeatPics = readMat(paste0("DNNL",lvlList[lvlFeat],"ImMat3X3.mat"))
      imFeatPics = imFeatPics[['DNNImMat']]
      imFeatMean = mean(imFeatPics)                              ### log mod ### 
      imFeatPics = logMod(imFeatPics)                            ### log mod ###
      imFeatPics = scale(imFeatPics)
      
      imFeatMix1 = readMat(paste0("DNNL",lvlList[lvlFeat],"Mix1Mat3X3HR.mat"))
      imFeatMix1 = imFeatMix1[['imFeatMix1']]
      imFeatMix1 = imFeatMix1*(imFeatMean/mean(imFeatMix1))      ### log mod ###      
      imFeatMix1 = logMod(imFeatMix1)                            ### log mod ###
      imFeatMix1 = imFeatMix1[8:nrow(imFeatMix1),]
      imFeatMix1 = scale(imFeatMix1)
      
      imFeatMix2 = readMat(paste0("DNNL",lvlList[lvlFeat],"Mix2Mat3X3HR.mat"))
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
    featN = ncol(imFeatPics)
    
    
    # train encoding model and predict (simulated) neural activity
    # variable initialization
    folds = 3
    foldSize = 90/folds
    
    sub = 1002
    f = 1
    vox = 1
    rankTriSubPrb = array(NA,c(1,90))
    rankTriSubVis = array(NA,c(1,90))
    predActPrbSub = list()
    predActVisSub = list()
    sseVoxSub = list()
    iterVoxSub = list()
    optLambda = list()
    
    dat1 = simDatMix1Noise
    dat2 = simDatMix2Noise
    dat3 = simDatPicsNoise
    datPrb = simDatPrbNoise
    datVis = simDatVivNoise
    
    sseVoxSub[[toString(sub)]] = 1:ncol(dat1)
    iterVoxSub[[toString(sub)]] = 1:ncol(dat1)
    optLambda[[toString(sub)]] = 1:ncol(dat1)
    
    imFeatTemp = imFeatPics[imIX,]
    imgOrder = imIX
    imFeatTempPrb = imFeatPics[imIX,]
    imFeatTempVis = imFeatPics[imIX,]
    
    dat = rbind(dat1,dat2,dat3)
    imFeatTempAll = rbind(imFeatMix1,imFeatMix2,imFeatTemp)
    
    predActPrb = array(NA,c(nrow(datPrb),ncol(datPrb)))
    predActVis = array(NA,c(nrow(datPrb),ncol(datPrb)))
    
    trainPred = imFeatTempAll
    testPredPrb = imFeatTempPrb
    testPredVis = imFeatTempVis
    
    nbackRowStart = nrow(trainPred)-nrow(imFeatTemp)
    
    # predict activity for each voxel
    for (vox in 1:ncol(dat)) {
      if (vox%%100==0) {
        print(vox)
      }
      
      datVox = dat[,vox]
      
      corIX = sort.int(cor(matrix(datVox,nrow=length(datVox),ncol=1),trainPred),index.return=T,decreasing=T)$ix[1:100]
      
      # predict activity using 3-fold cross validation
      cvImRand = sample(1:90)
      lmdaFold = 1:folds
      sseFold = 1:folds
      iterFold = 1:folds
      for (f in 1:folds) {
        fIms = cvImRand[(foldSize*(f-1)+1):(foldSize*(f))]
        ImgsTest = imgOrder%in%fIms
        fIXTest = nbackRowStart+which(ImgsTest)
        fIXTrain = c(1:nbackRowStart,nbackRowStart+which(!ImgsTest))
        
        # train model
        g1 = nnlasso(trainPred[fIXTrain,corIX],datVox[fIXTrain],family="normal",lambda=NA,
                     intercept=TRUE,normalize=TRUE,tau=1,tol=1e-3,maxiter=10000,nstep=5,min.lambda=1e-4,
                     eps=1e-6,path=T,SE=F)
        sseTemp = 1:length(g1$lambdas)
        for (l in 1:length(g1$lambdas)) {
          prednn = rep(g1$beta0[l],length(fIXTest))+rowSums(matrix(g1$coef[l,],nrow=length(fIXTest),ncol=length(corIX),byrow=T)*trainPred[fIXTest,corIX])
          sseTemp[l] = sum((datVox[fIXTest]-prednn)^2)/sum((datVox[fIXTest]-mean(datVox[fIXTest]))^2)
        }
        lmdaIX = which.min(sseTemp)
        lmdaFold[f] = g1$lambdas[lmdaIX]
        sseFold[f] = min(sseTemp)
        iterFold[f] = sum(g1$lambda.iter)
        
        # use trained model to predict voxel activity during retrieval
        predActPrb[fIms,vox] = rep(g1$beta0[lmdaIX],foldSize) + 
          rowSums(matrix(g1$coef[lmdaIX,],nrow=foldSize,ncol=length(corIX),byrow=T)*testPredPrb[fIms,corIX])
        
        predActVis[fIms,vox] = rep(g1$beta0[lmdaIX],foldSize) + 
          rowSums(matrix(g1$coef[lmdaIX,],nrow=foldSize,ncol=length(corIX),byrow=T)*testPredVis[fIms,corIX])
      }
      optLambda[[toString(sub)]][vox] = mean(lmdaFold)
      sseVoxSub[[toString(sub)]][vox] = mean(sseFold)
      iterVoxSub[[toString(sub)]][vox] = mean(iterFold)
      
    }
    predActPrbSub[[toString(sub)]] = predActPrb
    predActVisSub[[toString(sub)]] = predActVis
    
    # voxel indexes for each simulated ROI (8 total)
    ROIStart = rep(NA,length(simVoxIX))
    for (x in 1:length(simVoxIX)) {
      if (x==1) {
        ROIStart[x] = 0
      } else {
        ROIStart[x] = ROIStart[x-1]+length(simVoxIX[[x-1]])
      }
    }
    ROIStart[5:8] = ROIStart + ROIStart[4] + length(simVoxIX[[4]])
    voxIX = list()
    voxIX[[1]] = 1:length(simVoxIX[[1]])
    voxIX[[2]] = (max(voxIX[[1]])+1):(max(voxIX[[1]])+length(simVoxIX[[2]]))
    voxIX[[3]] = (max(voxIX[[2]])+1):(max(voxIX[[2]])+length(simVoxIX[[3]]))
    voxIX[[4]] = (max(voxIX[[3]])+1):(max(voxIX[[3]])+length(simVoxIX[[4]]))
    voxIX[[5]] = (max(voxIX[[4]])+1):(max(voxIX[[4]])+length(simVoxIX[[1]]))
    voxIX[[6]] = (max(voxIX[[5]])+1):(max(voxIX[[5]])+length(simVoxIX[[2]]))
    voxIX[[7]] = (max(voxIX[[6]])+1):(max(voxIX[[6]])+length(simVoxIX[[3]]))
    voxIX[[8]] = (max(voxIX[[7]])+1):(max(voxIX[[7]])+length(simVoxIX[[4]]))
    
    # get prediction rank of target image for each ROI:trial
    # Prb results simulates perception (i.e. no memory loss)
    print(paste0('Average Rank Accuracy per ROI:'))
    rankAv = 45.5
    print('Perception')
    for (roi in 1:8) {
      predCor = cor(t(predActPrbSub[[toString(sub)]][,voxIX[[roi]]]),t(datPrb[,voxIX[[roi]]]))
      for (trial in 1:length(imIX)) {
        rankTriSubPrb[1,trial] = which(sort.int(predCor[trial,],decreasing=T,index.return=T)$ix==trial)
      }
      print(rankAv-mean(rankTriSubPrb[1,]))
      simRankPrb[sim,roi,lvlFeat] = rankAv-mean(rankTriSubPrb[1,])
      simRankPrbTri[sim,roi,lvlFeat,] = rankAv-rankTriSubPrb[1,]
      simCorPrb[sim,roi,lvlFeat] = mean(diag(cor(predActPrbSub[[toString(sub)]][,voxIX[[roi]]],datPrb[,voxIX[[roi]]])))
    }
    
    # Vis results simulates recall (i.e. memory loss)
    print('Recall')
    for (roi in 1:8) {
      predCor = cor(t(predActVisSub[[toString(sub)]][,voxIX[[roi]]]),t(datVis[,voxIX[[roi]]]))
      for (trial in 1:length(imIX)) {
        rankTriSubVis[1,trial] = which(sort.int(predCor[trial,],decreasing=T,index.return=T)$ix==trial)
      }
      print(rankAv-mean(rankTriSubVis[1,]))
      simRankVis[sim,roi,lvlFeat] = rankAv-mean(rankTriSubVis[1,])
      simRankVisTri[sim,roi,lvlFeat,] = rankAv-rankTriSubVis[1,]
      simCorVis[sim,roi,lvlFeat] = mean(diag(cor(predActVisSub[[toString(sub)]][,voxIX[[roi]]],datVis[,voxIX[[roi]]])))
    }
  }
}

# save rank accuracy results for all simulated subjects, feature levels, and trials
save(simRankPrb,simRankVis,simRankPrbTri,simRankVisTri,simCorPrb,simCorVis,simImg,file="simRankTest1.RData")



#### Results: Data ####
# load the desired reactivation (rank format) results 
# 'simRankIdenticalMem200Sub.RData' and 'simRankIndependantMem200Sub.RData' is the simulated data from the paper assuming
# identical and independent trial-by-trial reactivation fidelity across feature levels, respectively

## data to use ##
load(file="simRankIdenticalMem200Sub.RData")
#load(file="simRankIndependantMem200Sub.RData")
##

# put data in useful format
subN = dim(simRankVisTri)[1] # 200 simulated subjects
trialN = dim(simRankVisTri)[4] # 90 recall trials
dat = matrix()
sub = 1
trial = 1
count = 0
for (sub in 1:subN) {
  for (trial in 1:trialN) {
    count = count + 1
    if (count == 1) {
      temp = c(sub,trial,simImg[sub,trial],NA,NA,NA,NA,c(simRankVisTri[sub,,,trial]))
      dat = matrix(temp,ncol=length(temp))
    } else {
      dat = rbind(dat,c(sub,trial,simImg[sub,trial],NA,NA,NA,NA,c(simRankVisTri[sub,,,trial])))
    }
  }
}
dat = as.data.frame(dat)
tempNames = c()
lvlN = dim(simRankVisTri)[3]
roiN = dim(simRankVisTri)[2]
for (l in 1:lvlN) {
  for (r in 1:roiN) {
    tempNames = c(tempNames,paste0("l",toString(l),"r",toString(r)))
  }
}
colnames(dat) = c('sub','trial','image','vividL1','vividL2','vividL3','vividL4',tempNames)
dat$sub = as.factor(dat$sub)
dat$image = as.factor(dat$image)
dat$vividL1 = 3-floor(dat$vividL1)
dat$vividL2 = 3-floor(dat$vividL2)
dat$vividL3 = 3-floor(dat$vividL3)
dat$vividL4 = 3-floor(dat$vividL4)



#### Results: Mean Rank ####
## Mean Decoding Accuracy (generate Figure 3a using "simRankIdenticalMem200Sub.RData") ##
# takes about 2 hours with 200 subjects and 1000 bootstrap iterations

BootN = 1000 # bootstrap iterations


ROItemp = list()
ROItemp[[1]] = c("l1r1","l1r2","l1r3","l1r4")
ROItemp[[2]] = c("l2r1","l2r2","l2r3","l2r4")
ROItemp[[3]] = c("l3r1","l3r2","l3r3","l3r4")
ROItemp[[4]] = c("l4r1","l4r2","l4r3","l4r4")
bootRand = F
MeanVals = array(NA,c(length(ROItemp[[1]]),4))
rownames(MeanVals) = ROItemp[[1]]
TVals = MeanVals
PVals = MeanVals
seVals = MeanVals
UBVals = MeanVals
LBVals = MeanVals
for (lvl in 1:4) {
  print(lvl)
  count = 0
  for (roiV in ROItemp[[lvl]]) {
    count = count + 1
    
    fmla = formula(paste0(roiV,' ~ 1 + (1|sub) + (1|image)'))
    nlm = lmer(fmla,data=dat,control=lmerControl(optimizer="Nelder_Mead"))
    
    MeanVals[count,lvl] = summary(nlm)$coefficients[1]
    seVals[count,lvl] = summary(nlm)$coefficients[2]
    TVals[count,lvl] = summary(nlm)$coefficients[3]
    bootTemp = bootMer(nlm,function (fit) {c(fixef(fit)[1])},nsim=BootN)$t
    bootTemp = sort(bootTemp)
    PVals[count,lvl] = which.min(abs(bootTemp))/length(bootTemp)
    UBVals[count,lvl] = bootTemp[.95*BootN]
    LBVals[count,lvl] = bootTemp[.05*BootN]
  }
}
save(MeanVals,TVals,PVals,UBVals,LBVals,file="simRankIdenticalMem200Sub_MeanStats_Test.RData")

# load pre-calculated values
load(file="simRankIdenticalMem200Sub_MeanStats.RData")

# plot results (Figure 2a using "simRankIdenticalMem200Sub_FSICStats.RData")
simRankDat = data.frame('val'=c(MeanVals),'p'=c(PVals),'ub'=c(UBVals),'lb'=c(LBVals),'ROI'=as.factor(rep(c('1','2','3','4'),4)),
                        'lvl'=as.factor(rep(c('1','2','3','4'),each=4)))
limits = aes(ymax = ub, ymin = lb)
dodge = position_dodge(width=0.9)
print(ggplot(simRankDat[which(!simRankDat$ROI=='All'),], aes(x=ROI, y=val, fill=lvl)) + 
        theme(panel.background = element_rect(fill = 'white', colour = 'grey'),text = element_text(size=20),
              axis.text.x=element_text(colour="black"),axis.text.y=element_text(colour="black")) +
        geom_bar(position=dodge, stat="identity") + 
        geom_errorbar(limits, position=dodge, width=0.20, size=.5) + 
        xlab("ROI") +
        ylab("rank") +
        ggtitle('Sim Mean Rank'))



#### Results: FSIC ####
## Figure 3b generated using "simRankIdenticalMem200Sub.RData" ##
# takes about 2.5 hours with 200 subjects and 1000 bootstrap iterations

BootN = 1000 # bootstrap iterations


ROItemp = c("l1r1","l1r2","l1r3","l1r4")
ROItemp1 = c("l2r1","l2r2","l2r3","l2r4")
ROItemp2 = c("l3r1","l3r2","l3r3","l3r4")
ROItemp3 = c("l4r1","l4r2","l4r3","l4r4")
LvlNameROIs = c("l1r5","l2r6","l3r7","l4r8")
datBoot = dat
CorVals = array(NA,c(length(ROItemp),4,length(LvlNameROIs)))
rownames(CorVals) = ROItemp
TVals = CorVals
PVals = CorVals
seVals = CorVals
UBVals = CorVals
LBVals = CorVals

LvlNameROI = "l1r5"
count2 = 0
for (roiF in c(LvlNameROIs)) {
  count2 = count2 + 1
  print(roiF)
  
  count = 0
  for (roiV in ROItemp) {
    count = count + 1
    
    fmla = formula(paste0('scale(',roiF,') ~ scale(',ROItemp[count],') + scale(',ROItemp1[count],') + scale(',ROItemp2[count],') + scale(',ROItemp3[count],') + (1|sub) + (1|image)'))
    nlm = lmer(fmla,data=datBoot,control=lmerControl(optimizer="Nelder_Mead"))

    CorVals[count,1,count2] = summary(nlm)$coefficients[2]
    CorVals[count,2,count2] = summary(nlm)$coefficients[3]
    CorVals[count,3,count2] = summary(nlm)$coefficients[4]
    CorVals[count,4,count2] = summary(nlm)$coefficients[5]
    seVals[count,1,count2] = summary(nlm)$coefficients[7]
    seVals[count,2,count2] = summary(nlm)$coefficients[8]
    seVals[count,3,count2] = summary(nlm)$coefficients[9]
    seVals[count,4,count2] = summary(nlm)$coefficients[10]
    TVals[count,1,count2] = summary(nlm)$coefficients[12]
    TVals[count,2,count2] = summary(nlm)$coefficients[13]
    TVals[count,3,count2] = summary(nlm)$coefficients[14]
    TVals[count,4,count2] = summary(nlm)$coefficients[15]
    bootTemp = bootMer(nlm,function (fit) {c(fixef(fit)[2],fixef(fit)[3],fixef(fit)[4],fixef(fit)[5])},nsim=BootN)$t
    bootVarsN = ncol(bootTemp)
    for (v in 1:bootVarsN) {
      varBoot = sort(bootTemp[,v])
      PVals[count,v,count2] = which.min(abs(varBoot))/length(varBoot)
      UBVals[count,v,count2] = varBoot[.95*BootN]
      LBVals[count,v,count2] = varBoot[.05*BootN]
    }
  }
}
save(CorVals,TVals,PVals,UBVals,LBVals,file="simRankIdenticalMem200Sub_FSICStats_Test.RData")


# load pre-calculated values
load(file="simRankIdenticalMem200Sub_FSICStats.RData")
#load(file="simRankIndependantMem200Sub_FSICStats.RData")

# reorganize data for plotting
temp = apply(CorVals,c(1,2,3),mean)
TValsDat = c()
for (i in 1:4) {
  TValsDat = c(TValsDat,temp[,i,i])
}
temp = apply(PVals,c(1,2,3),mean)
pDat = c()
for (i in 1:4) {
  pDat = c(pDat,temp[,i,i])
}
temp = apply(UBVals,c(1,2,3),mean)
ubDat = c()
for (i in 1:4) {
  ubDat = c(ubDat,temp[,i,i])
}
temp = apply(LBVals,c(1,2,3),mean)
lbDat = c()
for (i in 1:4) {
  lbDat = c(lbDat,temp[,i,i])
}
TVals
array(p.adjust(PVals,method ="fdr"),dim(PVals))


# plot results (Figure 3b using "simRankIdenticalMem200Sub_FSICStats.RData")
simRankDat = data.frame('TVal'=TValsDat,'p' = pDat,'ub' = ubDat,'lb' = lbDat,'ROI'=as.factor(rep(c('1','2','3','4'),4)),
                        'lvl'=as.factor(rep(c('1','2','3','4'),each=4)))
limits = aes(ymax = ub, ymin = lb)
dodge = position_dodge(width=0.9)
print(ggplot(simRankDat[which(!simRankDat$ROI=='All'),], aes(x=ROI, y=TVal, fill=lvl)) + 
        theme(panel.background = element_rect(fill = 'white', colour = 'grey'),text = element_text(size=20),
              axis.text.x=element_text(colour="black"),axis.text.y=element_text(colour="black")) +
        geom_bar(position=dodge, stat="identity") + 
        geom_errorbar(limits, position=dodge, width=0.20, size=.5) + 
        xlab("ROI") +
        ylab("coefficient") +
        ggtitle('FSIC'))

