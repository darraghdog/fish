library(data.table)
rm(list=ls())
subm_01 <- fread("~/Dropbox/fish/sub/subm_full_conv_20170302.csv")
subm_00 <- fread("~/Dropbox/fish/sub/subm_full_conv_20170302A.csv")
subm_02B <- fread("~/Dropbox/fish/sub/avg_2_best_0103.csv")

# subm_01[,image:= gsub("_200", "", image)]

#subm_02B = subm_02[image %in% subm_01$image]
#subm_02A = subm_02[!image %in% subm_01$image]

#subm_03 <- fread("~/Dropbox/fish/sub/subm_full_conv_20170210.csv")
subm_01 = subm_01[order(image)]
subm_00 = subm_00[order(image)]
subm_02B = subm_02B[order(image)]
#subm_03 = subm_03[order(image)]
subm_01
subm_00
subm_02B
#subm_03 
subm_04 = subm_01
cols = names(subm_02B)[-1]
for (var in cols) subm_04[[var]] = (subm_01[[var]]*.25) + (subm_02B[[var]]*.5) + (subm_00[[var]]*.25) 
#subm_05 = rbind(subm_04, subm_02A)
setwd("~/Dropbox/fish/sub")
write.csv(subm_04, "avg_2_best_50_50_morebags_0303.csv",  row.names = F)

###########################################################
library(data.table)
rm(list=ls())
subm_01 <- fread("~/Dropbox/fish/sub/avg_2_best_50_50_morebags_0303.csv")
subm_02 <- fread("~/Dropbox/fish/sub/subm_full_NoF_20170303A.csv")
subm_01 = subm_01[order(image)]
subm_02 = subm_02[order(image)]

hist(subm_01$NoF[subm_02$NoF<0.4])
subm_01 = data.frame(subm_01)
subm_02 = data.frame(subm_02)

subm_01$NoF[subm_02$NoF<0.4] = subm_02$NoF[subm_02$NoF<0.4]

rsum = rowSums(subm_01[,2:9])

subm_01[,2:9] = subm_01[,2:9]/rsum


write.csv(subm_01, "~/Dropbox/fish/sub/best_0303_separate_nof_classifier.csv",  row.names = F)
