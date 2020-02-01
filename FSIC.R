#### FSIC Stats ####
# assumes all required files are in the working directory
#
# instructions:
# - run code in "Dependances" (download any missing packages)
# - run code in "Get Data"
#
# - if you want to recreate the data in Figure 4b (FSIC stats), first run "Results: Data" section followed by "Results: FSIC"
# - saves csv files of the t values in Figure 4b (t value of -10 indicates a seed roi). Only significant t values 
#   included (FDR corrected over ROIs, excluding the seed ROIs)



#### Dependances ####

library(lme4)
library(ggplot2)



#### Get Data ####
# load the desired reactivation (rank format) results:
#
# 'ReacRankRecall.RData' contains the recall reactivation data used in the paper.
# trials are along the rows (sorted alphabetically by cue label, not in temporal order).
# columns:
# sub = subject id
# label = descriptive label for image pair
# image = the cued encoded image
# probe = the recollection probe image (if image = probe then recognition condition is 'old')
# vivid = vividness rating (1-4, NA means no response - NA trials (21 out of 2430) were not included in most analyses)
# acc = 1 if old/new recognition task response is correct, 0 otherwise
# confidence = confidence rating (1-4), wasn't used in current study
# old = true if recognition condition is 'old', false otherwise
# oldAcc = same as acc but with all 'new' trials NA (useful when averaging accuracy by condition)
# newAcc = same as acc but with all 'old' trials NA
# roi<roi number (FreeSurfer rois)>lvl<feature level (1-4, i.e. low, mid, high, semantic)>vis = reactivation rank measure
#            for the specified roi and feature level during recall
# seed<seed level (1-4, i.e. low, mid, high, semantic)>lvl<feature level (1-4, i.e. low, mid, high, semantic)>vis = 
#            reactivation rank measure for the specified roi (see Figure 4a) and feature level during recall
# seed<seed level ('low' or 'high')>lvl<feature level ('low' or 'high')>vis = 
#            reactivation rank measure for the specified roi (see Figure 5a. 'low' = average of 1 and 2 above,
#            high = average of 3 and 4) and feature level ('low' = average of 1 and 2,
#            high = average of 3 and 4) during recall

## data to use ##
load(file="ReacRankRecall.RData") # reactivation
load(file="ROINamesNum.RData") # roi numerical names (only left hemisphere because it was used to represent bilateral ROIs)
load(file="ROINamesFreeSurf.RData") # mapping between numerical and descriptive names
load(file="seedWeights.RData") # seed weights
##



#### Results: FSIC ####
## generate FSIC stats in Figure 4b using "ReacRankRecall.RData" ##
# takes about 3 hours

dat = ReacRankRecall[which(!is.na(ReacRankRecall$vivid)),] # reactivation data.frame (remove vivid NAs)

BootN = 1000 # bootstrap iterations

CorVals = array(NA,c(length(ROINamesNum),4,4),dimnames=list('roi'=ROINamesNum,'lvl'=1:4,'seed'=1:4)) # mean rank
TVals = CorVals # t value of mean
PVals = CorVals # p value of mean
UBVals = CorVals # 90% CI upper bound of mean
LBVals = CorVals # 90% CI lower bound of mean
for (seed in 1:4) {
  print(seed)
  seedName = paste0('seed',seed,'lvl',seed,'vis')
  for (roi in ROINamesNum) {
    print(roi)
    trgtName1 = paste0('roi',roi,'lvl',1,'vis')
    trgtName2 = paste0('roi',roi,'lvl',2,'vis')
    trgtName3 = paste0('roi',roi,'lvl',3,'vis')
    trgtName4 = paste0('roi',roi,'lvl',4,'vis')
    
    fmla = formula(paste0('scale(',seedName,') ~ scale(',trgtName1,') + scale(',trgtName2,') + scale(',trgtName3,') + scale(',trgtName4,') + (1|sub) + (1|image)'))
    nlm = lmer(fmla,data=dat,control=lmerControl(optimizer="Nelder_Mead"))
    
    CorVals[roi,1,seed] = summary(nlm)$coefficients[2]
    CorVals[roi,2,seed] = summary(nlm)$coefficients[3]
    CorVals[roi,3,seed] = summary(nlm)$coefficients[4]
    CorVals[roi,4,seed] = summary(nlm)$coefficients[5]
    TVals[roi,1,seed] = summary(nlm)$coefficients[12]
    TVals[roi,2,seed] = summary(nlm)$coefficients[13]
    TVals[roi,3,seed] = summary(nlm)$coefficients[14]
    TVals[roi,4,seed] = summary(nlm)$coefficients[15]
    bootTemp = bootMer(nlm,function (fit) {c(fixef(fit)[2],fixef(fit)[3],fixef(fit)[4],fixef(fit)[5])},nsim=BootN)$t
    bootVarsN = ncol(bootTemp)
    for (v in 1:bootVarsN) {
      varBoot = sort(bootTemp[,v])
      PVals[roi,v,seed] = which.min(abs(varBoot))/BootN
      UBVals[roi,v,seed] = varBoot[.95*BootN]
      LBVals[roi,v,seed] = varBoot[.05*BootN]
    }
  }
}
save(CorVals,TVals,PVals,UBVals,LBVals,file="reacFSICStats_Test.RData")


# load pre-calculated values
load(file="reacFSICStats.RData")

# generate csv files of the t values in Figure 4b (t value of -10 indicates a seed roi)
# only significant t values included (FDR corrected over ROIs, excluding the seed ROIs)
tempNames = row.names(PVals)
seed=4
lvl=4
for (seed in 1:4) {
  seedIX = seedWeights[,seed]>0
  TVals[seedIX,,seed] = -10
  PValsAdj = PVals[,,seed]
  for (lvl in 1:4) {
    PValsAdj[-seedIX,lvl] = p.adjust(PVals[-seedIX,lvl,seed],method ="fdr")
  }
  PValsAdj[seedIX,] = 0
  
  for (lvl in 1:4) {
    sigIX = which(PValsAdj[,lvl]<.05)
    LowVisFeatRein = data.frame('ROI'=tempNames[sigIX],'ROIName'=ROINamesFreeSurf[tempNames[sigIX]],'FCor'=TVals[sigIX,lvl,seed])
    write.csv(LowVisFeatRein,paste0('FSICSeed',seed,'Feat',lvl,'.csv'),row.names=F)
  }
}



#### Alternative Approach: FSIC using LM instead of LMER ####
## generate FSIC stats without using LMER - results are very similar ##
# controls for subject and image by adding two IVs composed of the subject and image DV means
# takes about 4 hours

dat = ReacRankRecall # reactivation data.frame (remove vivid NAs)
subN = length(unique(dat$sub))
imgN = length(unique(dat$label))

BootN = 1000 # bootstrap iterations

CorVals = array(NA,c(length(ROINamesNum),4,4),dimnames=list('roi'=ROINamesNum,'lvl'=1:4,'seed'=1:4)) # mean rank
TVals = CorVals # t value of mean
PVals = CorVals # p value of mean
UBVals = CorVals # 90% CI upper bound of mean
LBVals = CorVals # 90% CI lower bound of mean
seed = 1
roi = ROINamesNum[1]
for (seed in 1:4) {
  print(seed)
  seedName = paste0('seed',seed,'lvl',seed,'vis')
  for (roi in ROINamesNum) {
    print(roi)
    trgtName1 = paste0('roi',roi,'lvl',1,'vis')
    trgtName2 = paste0('roi',roi,'lvl',2,'vis')
    trgtName3 = paste0('roi',roi,'lvl',3,'vis')
    trgtName4 = paste0('roi',roi,'lvl',4,'vis')
    
    dat2 = dat[which(!is.na(dat$vivid)),]
    dat2$subMeanTemp = NA
    dat2$imgMeanTemp = NA
    
    subMeans = aggregate(formula(paste0(seedName,'~sub')),dat2,FUN=mean)
    rownames(subMeans) = subMeans$sub
    for (sub in subMeans$sub) {
      dat2[which(dat2$sub==sub),'subMeanTemp'] = subMeans[sub,seedName]
    }
    imgMeans = aggregate(formula(paste0(seedName,'~image')),dat2,FUN=mean)
    rownames(imgMeans) = imgMeans$image
    for (img in imgMeans$image) {
      dat2[which(dat2$image==img),'imgMeanTemp'] = imgMeans[img,seedName]
    }
    
    fmla = formula(paste0('scale(',seedName,') ~ scale(',trgtName1,') + scale(',trgtName2,') + scale(',trgtName3,') + scale(',trgtName4,') + scale(subMeanTemp) + scale(imgMeanTemp)'))
    nlm = lm(fmla,data=dat2)
    CorVals[roi,1,seed] = summary(nlm)$coefficients[2]
    CorVals[roi,2,seed] = summary(nlm)$coefficients[3]
    CorVals[roi,3,seed] = summary(nlm)$coefficients[4]
    CorVals[roi,4,seed] = summary(nlm)$coefficients[5]
    TVals[roi,1,seed] = summary(nlm)$coefficients[16]
    TVals[roi,2,seed] = summary(nlm)$coefficients[17]
    TVals[roi,3,seed] = summary(nlm)$coefficients[18]
    TVals[roi,4,seed] = summary(nlm)$coefficients[19]
    
    bootTemp1 = 1:BootN
    bootTemp2 = 1:BootN
    bootTemp3 = 1:BootN
    bootTemp4 = 1:BootN
    for (n in 1:BootN) {
      subsBoot = sample(1:subN,replace=T) # sample subjects
      ix = rep(1:imgN,subN) + rep((subsBoot-1)*imgN,each=imgN)
      dat2 = dat[ix,]
      dat2 = dat2[which(!is.na(dat2$vivid)),]
      dat2$subMeanTemp = NA
      dat2$imgMeanTemp = NA
      
      subMeans = aggregate(formula(paste0(seedName,'~sub')),dat2,FUN=mean)
      rownames(subMeans) = subMeans$sub
      for (sub in subMeans$sub) {
        dat2[which(dat2$sub==sub),'subMeanTemp'] = subMeans[sub,seedName]
      }
      imgMeans = aggregate(formula(paste0(seedName,'~image')),dat2,FUN=mean)
      rownames(imgMeans) = imgMeans$image
      for (img in imgMeans$image) {
        dat2[which(dat2$image==img),'imgMeanTemp'] = imgMeans[img,seedName]
      }
      
      fmla = formula(paste0('scale(',seedName,') ~ scale(',trgtName1,') + scale(',trgtName2,') + scale(',trgtName3,') + scale(',trgtName4,') + scale(subMeanTemp) + scale(imgMeanTemp)'))
      nlm = lm(fmla,data=dat2)
      bootTemp1[n] = summary(nlm)$coefficients[2]
      bootTemp2[n] = summary(nlm)$coefficients[3]
      bootTemp3[n] = summary(nlm)$coefficients[4]
      bootTemp4[n] = summary(nlm)$coefficients[5]
    }
    bootTemp = sort(bootTemp1)
    PVals[roi,1,seed] = which.min(abs(bootTemp))/BootN
    UBVals[roi,1,seed] = bootTemp[.95*BootN]
    LBVals[roi,1,seed] = bootTemp[.05*BootN]
    bootTemp = sort(bootTemp2)
    PVals[roi,2,seed] = which.min(abs(bootTemp))/BootN
    UBVals[roi,2,seed] = bootTemp[.95*BootN]
    LBVals[roi,2,seed] = bootTemp[.05*BootN]
    bootTemp = sort(bootTemp3)
    PVals[roi,3,seed] = which.min(abs(bootTemp))/BootN
    UBVals[roi,3,seed] = bootTemp[.95*BootN]
    LBVals[roi,3,seed] = bootTemp[.05*BootN]
    bootTemp = sort(bootTemp4)
    PVals[roi,4,seed] = which.min(abs(bootTemp))/BootN
    UBVals[roi,4,seed] = bootTemp[.95*BootN]
    LBVals[roi,4,seed] = bootTemp[.05*BootN]
  }
}
save(CorVals,TVals,PVals,UBVals,LBVals,file="reacFSICStats_LM_Test.RData")


# load pre-calculated values
load(file="reacFSICStats_LM_Test.RData")

# generate csv files of the t values in Figure 4b (t value of -10 indicates a seed roi)
# only significant t values included (FDR corrected over ROIs, excluding the seed ROIs)
tempNames = row.names(PVals)
seed=4
lvl=4
for (seed in 1:4) {
  seedIX = seedWeights[,seed]>0
  TVals[seedIX,,seed] = -10
  PValsAdj = PVals[,,seed]
  for (lvl in 1:4) {
    PValsAdj[-seedIX,lvl] = p.adjust(PVals[-seedIX,lvl,seed],method ="fdr")
  }
  PValsAdj[seedIX,] = 0
  
  for (lvl in 1:4) {
    sigIX = which(PValsAdj[,lvl]<.05)
    LowVisFeatRein = data.frame('ROI'=tempNames[sigIX],'ROIName'=ROINamesFreeSurf[tempNames[sigIX]],'FCor'=TVals[sigIX,lvl,seed])
    write.csv(LowVisFeatRein,paste0('FSICSeed',seed,'Feat',lvl,'_LM.csv'),row.names=F)
  }
}
