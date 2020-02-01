#### Reactivation Correlated with Vividness and Accuracy Stats ####
# assumes all required files are in the working directory
#
# instructions:
# - run code in "Dependances" (download any missing packages)
# - run code in "Get Data"
#
# - if you want to recreate the data in Figure 5, first run "Results: Data" section followed by:
#   - "Results: Reactivation COR Vivid (Within-Subject)" for Figure 5b
#   - "Results: Reactivation COR Acc (Between-Subject)" for Figure 5c
#   - "Results: Reactivation COR Acc (Within-Subject) (Low vs High Lure Acc Subs)" for Figure 5d



#### Dependances ####

library(abind)
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
load(file="ROINamesNum.RData") # roi numerical names
load(file="ROINamesFreeSurf.RData") # mapping between numerical and descriptive names
load(file="seedWeights.RData") # seed weights
##



#### Results: Reactivation COR Vivid (Within-Subject) ####
## generate stats in Figure 5b using "ReacRankRecall.RData" ##
# takes under 1 min

dat = ReacRankRecall[which(!is.na(ReacRankRecall$vivid)),] # reactivation data.frame (remove vivid NAs)

BootN = 1000
tempNames = c('low','hgh')
CorVals = array(NA,c(length(tempNames),2),dimnames=list('ROI'=tempNames,'Feat'=1:2))
TVals = CorVals
PVals = CorVals
UBVals = CorVals
LBVals = CorVals

fmla = formula(paste0('scale(vivid) ~ scale(seedlowlvllowvis) + scale(seedlowlvlhighvis) + scale(seedhighlvllowvis) + scale(seedhighlvlhighvis) + (1|sub) + (1|image)'))
nlm = lmer(fmla,data=dat,control=lmerControl(optimizer="Nelder_Mead"))
summary(nlm)
coefN = 5
CorVals['low',1] = summary(nlm)$coefficients[(coefN*0)+2]
CorVals['low',2] = summary(nlm)$coefficients[(coefN*0)+3]
CorVals['hgh',1] = summary(nlm)$coefficients[(coefN*0)+4]
CorVals['hgh',2] = summary(nlm)$coefficients[(coefN*0)+5]
TVals['low',1] = summary(nlm)$coefficients[(coefN*2)+2]
TVals['low',2] = summary(nlm)$coefficients[(coefN*2)+3]
TVals['hgh',1] = summary(nlm)$coefficients[(coefN*2)+4]
TVals['hgh',2] = summary(nlm)$coefficients[(coefN*2)+5]
bootTemp = bootMer(nlm,function (fit) {fixef(fit)[c(2,3,4,5)]},nsim=BootN)$t
PVals['low',1] = which.min(abs(sort(bootTemp[,1])))/BootN
PVals['low',2] = which.min(abs(sort(bootTemp[,2])))/BootN
PVals['hgh',1] = which.min(abs(sort(bootTemp[,3])))/BootN
PVals['hgh',2] = which.min(abs(sort(bootTemp[,4])))/BootN
UBVals['low',1] = sort(bootTemp[,1])[.975*BootN]
UBVals['low',2] = sort(bootTemp[,2])[.975*BootN]
UBVals['hgh',1] = sort(bootTemp[,3])[.975*BootN]
UBVals['hgh',2] = sort(bootTemp[,4])[.975*BootN]
LBVals['low',1] = sort(bootTemp[,1])[.025*BootN]
LBVals['low',2] = sort(bootTemp[,2])[.025*BootN]
LBVals['hgh',1] = sort(bootTemp[,3])[.025*BootN]
LBVals['hgh',2] = sort(bootTemp[,4])[.025*BootN]


# two-tailed p vals
temp = sapply(c(PVals),function (x) {if (x>.5) x = 1+1/BootN-x
                                     x})*2
# FDR corrected
p.adjust(temp,method='fdr')

# is the difference between low- and high-level features sig?
which.min(abs(sort(bootTemp[,1]-bootTemp[,4])))/BootN


# Figure 5b
datTemp = data.frame('coef'=c(CorVals),'lb'=c(LBVals),'ub'=c(UBVals),
                     'roi'=rep(c('1','2'),2),'feat'=rep(c('1','2'),each=2))
limits = aes(ymax = ub, ymin = lb)
dodge = position_dodge(width=0.9)
print(ggplot(datTemp, aes(x=feat, y=coef, fill=roi)) + 
        theme(panel.background = element_rect(fill = 'white', colour = 'grey'),text = element_text(size=20),
              axis.text.x=element_text(colour="black"),axis.text.y=element_text(colour="black")) +
        geom_bar(position=dodge, stat="identity") + 
        geom_errorbar(limits, position=dodge, width=0.20, size=.5) + 
        #coord_cartesian(ylim = c(-.01,.17)) +
        xlab("rein lvl") +
        ylab("coefficient"))



#### Results: Reactivation COR Acc (Within-Subject) ####
## generate stats in Figure 5b using "ReacRankRecall.RData" ##
# takes under 1 min

dat = ReacRankRecall[which(!is.na(ReacRankRecall$vivid)),] # reactivation data.frame (remove vivid NAs)

BootN = 1000
tempNames = c('low','hgh')
CorVals = array(NA,c(length(tempNames),2),dimnames=list('ROI'=tempNames,'Feat'=1:2))
TVals = CorVals
PVals = CorVals
UBVals = CorVals
LBVals = CorVals

fmla = formula(paste0('scale(acc) ~ scale(seedlowlvllowvis) + scale(seedlowlvlhighvis) + scale(seedhighlvllowvis) + scale(seedhighlvlhighvis) + (1|sub) + (1|image)'))
nlm = lmer(fmla,data=dat,control=lmerControl(optimizer="Nelder_Mead"))
summary(nlm)
coefN = 5
CorVals['low',1] = summary(nlm)$coefficients[(coefN*0)+2]
CorVals['low',2] = summary(nlm)$coefficients[(coefN*0)+3]
CorVals['hgh',1] = summary(nlm)$coefficients[(coefN*0)+4]
CorVals['hgh',2] = summary(nlm)$coefficients[(coefN*0)+5]
TVals['low',1] = summary(nlm)$coefficients[(coefN*2)+2]
TVals['low',2] = summary(nlm)$coefficients[(coefN*2)+3]
TVals['hgh',1] = summary(nlm)$coefficients[(coefN*2)+4]
TVals['hgh',2] = summary(nlm)$coefficients[(coefN*2)+5]
bootTemp = bootMer(nlm,function (fit) {fixef(fit)[c(2,3,4,5)]},nsim=BootN)$t
PVals['low',1] = which.min(abs(sort(bootTemp[,1])))/BootN
PVals['low',2] = which.min(abs(sort(bootTemp[,2])))/BootN
PVals['hgh',1] = which.min(abs(sort(bootTemp[,3])))/BootN
PVals['hgh',2] = which.min(abs(sort(bootTemp[,4])))/BootN
UBVals['low',1] = sort(bootTemp[,1])[.975*BootN]
UBVals['low',2] = sort(bootTemp[,2])[.975*BootN]
UBVals['hgh',1] = sort(bootTemp[,3])[.975*BootN]
UBVals['hgh',2] = sort(bootTemp[,4])[.975*BootN]
LBVals['low',1] = sort(bootTemp[,1])[.025*BootN]
LBVals['low',2] = sort(bootTemp[,2])[.025*BootN]
LBVals['hgh',1] = sort(bootTemp[,3])[.025*BootN]
LBVals['hgh',2] = sort(bootTemp[,4])[.025*BootN]


# two-tailed p vals
temp = sapply(c(PVals),function (x) {if (x>.5) x = 1+1/BootN-x
                                     x})*2
# FDR corrected
p.adjust(temp,method='fdr')

# is the difference between low- and high-level features sig?
which.min(abs(sort(bootTemp[,1]-bootTemp[,4])))/BootN


# Figure not included in paper
datTemp = data.frame('coef'=c(CorVals),'lb'=c(LBVals),'ub'=c(UBVals),
                     'roi'=rep(c('1','2'),2),'feat'=rep(c('1','2'),each=2))
limits = aes(ymax = ub, ymin = lb)
dodge = position_dodge(width=0.9)
print(ggplot(datTemp, aes(x=feat, y=coef, fill=roi)) + 
        theme(panel.background = element_rect(fill = 'white', colour = 'grey'),text = element_text(size=20),
              axis.text.x=element_text(colour="black"),axis.text.y=element_text(colour="black")) +
        geom_bar(position=dodge, stat="identity") + 
        geom_errorbar(limits, position=dodge, width=0.20, size=.5) + 
        #coord_cartesian(ylim = c(-.01,.17)) +
        xlab("rein lvl") +
        ylab("coefficient"))



#### Results: Reactivation COR Acc (Between-Subject) ####
## generate stats in Figure 5b using "ReacRankRecall.RData" ##
# takes under 1 min

dat = aggregate(ReacRankRecall[,-c(1:4)], list(ReacRankRecall[,'sub']), function (x) {mean(x, na.rm=TRUE)}) # sub average

subsAcndAcc = dat[sort.int(dat$acc,index.return=T)$ix,'Group.1']
subsAcndOldAcc = dat[sort.int(dat$oldAcc,index.return=T)$ix,'Group.1']
subsAcndNewAcc = dat[sort.int(dat$newAcc,index.return=T)$ix,'Group.1']

BootN = 1000
tempNames = c('low','hgh')
CorVals = array(NA,c(length(tempNames),2),dimnames=list('ROI'=tempNames,'Feat'=1:2))
TVals = CorVals
PVals = CorVals
UBVals = CorVals
LBVals = CorVals

fmla = formula(paste0('scale(acc) ~ scale(seedlowlvllowvis) + scale(seedlowlvlhighvis) + scale(seedhighlvllowvis) + scale(seedhighlvlhighvis)'))
nlm = lm(fmla,data=dat)
summary(nlm)
coefN = 5
CorVals['low',1] = summary(nlm)$coefficients[(coefN*0)+2]
CorVals['low',2] = summary(nlm)$coefficients[(coefN*0)+3]
CorVals['hgh',1] = summary(nlm)$coefficients[(coefN*0)+4]
CorVals['hgh',2] = summary(nlm)$coefficients[(coefN*0)+5]
TVals['low',1] = summary(nlm)$coefficients[(coefN*2)+2]
TVals['low',2] = summary(nlm)$coefficients[(coefN*2)+3]
TVals['hgh',1] = summary(nlm)$coefficients[(coefN*2)+4]
TVals['hgh',2] = summary(nlm)$coefficients[(coefN*2)+5]

bootTemp = matrix(NA,nrow=BootN,ncol=4)
for (n in 1:BootN) {
  ix = sample(1:27,replace=T)
  nlm = lm(fmla,data=dat[ix,])
  coefN = 5
  bootTemp[n,1] = summary(nlm)$coefficients[(coefN*0)+2]
  bootTemp[n,2] = summary(nlm)$coefficients[(coefN*0)+3]
  bootTemp[n,3] = summary(nlm)$coefficients[(coefN*0)+4]
  bootTemp[n,4] = summary(nlm)$coefficients[(coefN*0)+5]
}
PVals['low',1] = which.min(abs(sort(bootTemp[,1])))/BootN
PVals['low',2] = which.min(abs(sort(bootTemp[,2])))/BootN
PVals['hgh',1] = which.min(abs(sort(bootTemp[,3])))/BootN
PVals['hgh',2] = which.min(abs(sort(bootTemp[,4])))/BootN
UBVals['low',1] = sort(bootTemp[,1])[.975*BootN]
UBVals['low',2] = sort(bootTemp[,2])[.975*BootN]
UBVals['hgh',1] = sort(bootTemp[,3])[.975*BootN]
UBVals['hgh',2] = sort(bootTemp[,4])[.975*BootN]
LBVals['low',1] = sort(bootTemp[,1])[.025*BootN]
LBVals['low',2] = sort(bootTemp[,2])[.025*BootN]
LBVals['hgh',1] = sort(bootTemp[,3])[.025*BootN]
LBVals['hgh',2] = sort(bootTemp[,4])[.025*BootN]


# two-tailed p vals (FDR corrected)
temp = sapply(c(PVals),function (x) {if (x>.5) x = 1+1/BootN-x
                                     x})*2
# FDR corrected
p.adjust(c(1,1,1,1,temp),method='fdr')

# is the difference between low- and high-level features sig?
which.min(abs(sort(bootTemp[,1]-bootTemp[,4])))/BootN


# Figure 5c
datTemp = data.frame('coef'=c(CorVals),'lb'=c(LBVals),'ub'=c(UBVals),
                     'roi'=rep(c('1','2'),2),'feat'=rep(c('1','2'),each=2))
limits = aes(ymax = ub, ymin = lb)
dodge = position_dodge(width=0.9)
print(ggplot(datTemp, aes(x=feat, y=coef, fill=roi)) + 
        theme(panel.background = element_rect(fill = 'white', colour = 'grey'),text = element_text(size=20),
              axis.text.x=element_text(colour="black"),axis.text.y=element_text(colour="black")) +
        geom_bar(position=dodge, stat="identity") + 
        geom_errorbar(limits, position=dodge, width=0.20, size=.5) + 
        #coord_cartesian(ylim = c(-.01,.17)) +
        xlab("rein lvl") +
        ylab("correlation coefficient"))



#### Results: Reactivation COR Acc (Within-Subject) (Low vs High Lure Acc Subs) ####
## generate stats in Figure 5b using "ReacRankRecall.RData" ##
# takes under 1 min

# sort subjects by accuracy
dat = aggregate(ReacRankRecall[,-c(1:4)], list(ReacRankRecall[,'sub']), function (x) {mean(x, na.rm=TRUE)}) # sub average
subsAcndAcc = dat[sort.int(dat$acc,index.return=T)$ix,'Group.1']
subsAcndOldAcc = dat[sort.int(dat$oldAcc,index.return=T)$ix,'Group.1']
subsAcndNewAcc = dat[sort.int(dat$newAcc,index.return=T)$ix,'Group.1']

# get data
dat = ReacRankRecall[which(!is.na(ReacRankRecall$vivid)),] # reactivation data.frame (remove vivid NAs)

BootN = 100
tempNames = c('low','hgh')
CorVals = array(NA,c(length(tempNames),2,length(tempNames)),dimnames=list('ROI'=tempNames,'Feat'=1:2,'SubGrp'=tempNames))
TVals = CorVals
PVals = CorVals
UBVals = CorVals
LBVals = CorVals
for (grp in tempNames) {
  print(grp)
  if (grp=='low') {
    dat2 = dat[which(dat$sub%in%subsAcndNewAcc[1:13]),]
  } else if (grp=='hgh') {
    dat2 = dat[which(dat$sub%in%subsAcndNewAcc[15:27]),]
  }
  fmla = formula(paste0('scale(acc) ~ scale(seedlowlvllowvis) + scale(seedlowlvlhighvis) + scale(seedhighlvllowvis) + scale(seedhighlvlhighvis) + (1|sub) + (1|image)'))
  nlm = lmer(fmla,data=dat2,control=lmerControl(optimizer="Nelder_Mead"))
  summary(nlm)
  coefN = 5
  CorVals['low',1,grp] = summary(nlm)$coefficients[(coefN*0)+2]
  CorVals['low',2,grp] = summary(nlm)$coefficients[(coefN*0)+3]
  CorVals['hgh',1,grp] = summary(nlm)$coefficients[(coefN*0)+4]
  CorVals['hgh',2,grp] = summary(nlm)$coefficients[(coefN*0)+5]
  TVals['low',1,grp] = summary(nlm)$coefficients[(coefN*2)+2]
  TVals['low',2,grp] = summary(nlm)$coefficients[(coefN*2)+3]
  TVals['hgh',1,grp] = summary(nlm)$coefficients[(coefN*2)+4]
  TVals['hgh',2,grp] = summary(nlm)$coefficients[(coefN*2)+5]
  bootTemp = bootMer(nlm,function (fit) {fixef(fit)[c(2,3,4,5)]},nsim=BootN)$t
  PVals['low',1,grp] = which.min(abs(sort(bootTemp[,1])))/BootN
  PVals['low',2,grp] = which.min(abs(sort(bootTemp[,2])))/BootN
  PVals['hgh',1,grp] = which.min(abs(sort(bootTemp[,3])))/BootN
  PVals['hgh',2,grp] = which.min(abs(sort(bootTemp[,4])))/BootN
  UBVals['low',1,grp] = sort(bootTemp[,1])[.975*BootN]
  UBVals['low',2,grp] = sort(bootTemp[,2])[.975*BootN]
  UBVals['hgh',1,grp] = sort(bootTemp[,3])[.975*BootN]
  UBVals['hgh',2,grp] = sort(bootTemp[,4])[.975*BootN]
  LBVals['low',1,grp] = sort(bootTemp[,1])[.025*BootN]
  LBVals['low',2,grp] = sort(bootTemp[,2])[.025*BootN]
  LBVals['hgh',1,grp] = sort(bootTemp[,3])[.025*BootN]
  LBVals['hgh',2,grp] = sort(bootTemp[,4])[.025*BootN]
}


# is the correlation with low-level reactivation greater than chance in the high-lure-accuracy group?
PVals[1,1,'hgh']

# two-tailed p vals
temp = sapply(c(PVals[,,'hgh']),function (x) {if (x>.5) x = 1+1/BootN-x
                                              x})*2
# FDR corrected
p.adjust(temp,method='fdr')

# is the difference between low- and high-level features sig?
which.min(abs(sort(bootTemp[,1]-bootTemp[,4])))/BootN


# Figure 5d
datTemp = data.frame('coef'=c(CorVals[,,'hgh']),'lb'=c(LBVals[,,'hgh']),'ub'=c(UBVals[,,'hgh']),
                     'roi'=rep(c('1','2'),2),'feat'=rep(c('1','2'),each=2))
limits = aes(ymax = ub, ymin = lb)
dodge = position_dodge(width=0.9)
print(ggplot(datTemp, aes(x=feat, y=coef, fill=roi)) + 
        theme(panel.background = element_rect(fill = 'white', colour = 'grey'),text = element_text(size=20),
              axis.text.x=element_text(colour="black"),axis.text.y=element_text(colour="black")) +
        geom_bar(position=dodge, stat="identity") + 
        geom_errorbar(limits, position=dodge, width=0.20, size=.5) + 
        #coord_cartesian(ylim = c(-.01,.17)) +
        xlab("rein lvl") +
        ylab("coefficient"))



#### Alternative Approach: Reactivation COR Acc (Within-Subject) (Low vs High Lure Acc Subs) ####
## generate stats in Figure 5b without using LMER (LM instead) - results are similar ##
# controls for subject and image by adding two IVs composed of the subject and image DV means
# takes under 1 min

# sort subjects by accuracy
dat = aggregate(ReacRankRecall[,-c(1:4)], list(ReacRankRecall[,'sub']), function (x) {mean(x, na.rm=TRUE)}) # sub average
subsAcndAcc = dat[sort.int(dat$acc,index.return=T)$ix,'Group.1']
subsAcndOldAcc = dat[sort.int(dat$oldAcc,index.return=T)$ix,'Group.1']
subsAcndNewAcc = dat[sort.int(dat$newAcc,index.return=T)$ix,'Group.1']

# get data
dat = ReacRankRecall # reactivation data.frame

BootN = 1000
tempNames = c('low','hgh')
CorVals = array(NA,c(length(tempNames),2,length(tempNames)),dimnames=list('ROI'=tempNames,'Feat'=1:2,'SubGrp'=tempNames))
TVals = CorVals
PVals = CorVals
UBVals = CorVals
LBVals = CorVals
for (grp in c('hgh')) {
  print(grp)
  if (grp=='low') {
    dat2 = dat[which(dat$sub%in%subsAcndNewAcc[1:13]),]
  } else if (grp=='hgh') {
    dat2 = dat[which(dat$sub%in%subsAcndNewAcc[15:27]),]
  }
  
  subN = length(unique(dat2$sub))
  imgN = 90
  
  dat3 = dat2[which(!is.na(dat2$vivid)),]
  dat3$subMeanTemp = NA
  dat3$imgMeanTemp = NA
  
  subMeans = aggregate(formula(paste0('acc~sub')),dat3,FUN=mean)
  rownames(subMeans) = subMeans$sub
  for (sub in subMeans$sub) {
    dat3[which(dat3$sub==sub),'subMeanTemp'] = subMeans[sub,'acc']
  }
  imgMeans = aggregate(formula(paste0('acc~image')),dat3,FUN=mean)
  rownames(imgMeans) = imgMeans$image
  for (img in imgMeans$image) {
    dat3[which(dat3$image==img),'imgMeanTemp'] = imgMeans[img,'acc']
  }
  
  fmla = formula(paste0('scale(acc) ~ scale(seedlowlvllowvis) + scale(seedlowlvlhighvis) + scale(seedhighlvllowvis) + scale(seedhighlvlhighvis) + scale(subMeanTemp) + scale(imgMeanTemp)'))
  nlm = lm(fmla,data=dat3)
  summary(nlm)
  coefN = 7
  CorVals['low',1,grp] = summary(nlm)$coefficients[(coefN*0)+2]
  CorVals['low',2,grp] = summary(nlm)$coefficients[(coefN*0)+3]
  CorVals['hgh',1,grp] = summary(nlm)$coefficients[(coefN*0)+4]
  CorVals['hgh',2,grp] = summary(nlm)$coefficients[(coefN*0)+5]
  TVals['low',1,grp] = summary(nlm)$coefficients[(coefN*2)+2]
  TVals['low',2,grp] = summary(nlm)$coefficients[(coefN*2)+3]
  TVals['hgh',1,grp] = summary(nlm)$coefficients[(coefN*2)+4]
  TVals['hgh',2,grp] = summary(nlm)$coefficients[(coefN*2)+5]
  
  bootTemp1 = 1:BootN
  bootTemp2 = 1:BootN
  bootTemp3 = 1:BootN
  bootTemp4 = 1:BootN
  for (n in 1:BootN) {
    subsBoot = sample(1:subN,replace=T) # sample subjects
    ix = rep(1:imgN,subN) + rep((subsBoot-1)*imgN,each=imgN)
    dat3 = dat2[ix,]
    dat3 = dat3[which(!is.na(dat3$vivid)),]
    dat3$subMeanTemp = NA
    dat3$imgMeanTemp = NA
    
    subMeans = aggregate(formula(paste0('acc~sub')),dat3,FUN=mean)
    rownames(subMeans) = subMeans$sub
    for (sub in subMeans$sub) {
      dat3[which(dat3$sub==sub),'subMeanTemp'] = subMeans[sub,'acc']
    }
    imgMeans = aggregate(formula(paste0('acc~image')),dat3,FUN=mean)
    rownames(imgMeans) = imgMeans$image
    for (img in imgMeans$image) {
      dat3[which(dat3$image==img),'imgMeanTemp'] = imgMeans[img,'acc']
    }
    
    fmla = formula(paste0('scale(acc) ~ scale(seedlowlvllowvis) + scale(seedlowlvlhighvis) + scale(seedhighlvllowvis) + scale(seedhighlvlhighvis) + scale(subMeanTemp) + scale(imgMeanTemp)'))
    nlm = lm(fmla,data=dat3)
    bootTemp1[n] = summary(nlm)$coefficients[2]
    bootTemp2[n] = summary(nlm)$coefficients[3]
    bootTemp3[n] = summary(nlm)$coefficients[4]
    bootTemp4[n] = summary(nlm)$coefficients[5]
  }
  bootTemp = sort(bootTemp1)
  PVals['low',1,grp] = which.min(abs(bootTemp))/BootN
  UBVals['low',1,grp] = bootTemp[.975*BootN]
  LBVals['low',1,grp] = bootTemp[.025*BootN]
  bootTemp = sort(bootTemp2)
  PVals['low',2,grp] = which.min(abs(bootTemp))/BootN
  UBVals['low',2,grp] = bootTemp[.975*BootN]
  LBVals['low',2,grp] = bootTemp[.025*BootN]
  bootTemp = sort(bootTemp3)
  PVals['hgh',1,grp] = which.min(abs(bootTemp))/BootN
  UBVals['hgh',1,grp] = bootTemp[.975*BootN]
  LBVals['hgh',1,grp] = bootTemp[.025*BootN]
  bootTemp = sort(bootTemp4)
  PVals['hgh',2,grp] = which.min(abs(bootTemp))/BootN
  UBVals['hgh',2,grp] = bootTemp[.975*BootN]
  LBVals['hgh',2,grp] = bootTemp[.025*BootN]
}


# is the correlation with low-level reactivation greater than chance in the high-lure-accuracy group?
PVals[1,1,'hgh']

# two-tailed p vals
temp = sapply(c(PVals[,,'hgh']),function (x) {if (x>.5) x = 1+1/BootN-x
x})*2
# FDR corrected
p.adjust(temp,method='fdr')

# is the difference between low- and high-level features sig?
which.min(abs(sort(bootTemp1-bootTemp4)))/BootN


# Figure 5d
datTemp = data.frame('coef'=c(CorVals[,,'hgh']),'lb'=c(LBVals[,,'hgh']),'ub'=c(UBVals[,,'hgh']),
                     'roi'=rep(c('1','2'),2),'feat'=rep(c('1','2'),each=2))
limits = aes(ymax = ub, ymin = lb)
dodge = position_dodge(width=0.9)
print(ggplot(datTemp, aes(x=feat, y=coef, fill=roi)) + 
        theme(panel.background = element_rect(fill = 'white', colour = 'grey'),text = element_text(size=20),
              axis.text.x=element_text(colour="black"),axis.text.y=element_text(colour="black")) +
        geom_bar(position=dodge, stat="identity") + 
        geom_errorbar(limits, position=dodge, width=0.20, size=.5) + 
        #coord_cartesian(ylim = c(-.01,.17)) +
        xlab("rein lvl") +
        ylab("coefficient"))
