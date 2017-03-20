library(data.table)
rm(list=ls())
subm_01 <- fread("~/Dropbox/fish/sub/subm_full_crop_of_20170310.csv")
subm_02 <- fread("~/Dropbox/fish/sub/avg_2_best_50_50_morebags_0303.csv")
subm_001 <- fread("~/Dropbox/fish/sub/RFCN_AGONOSTICnms_resnet101_rfcn_ohem_iter_30000_CROP_resnet50_clsMaxAve_conf0.80_T2.5_0.2273.csv")
# 0.74700

subm_02B = subm_02[image %in% subm_01$image]
subm_02A = subm_02[!image %in% subm_01$image]

subm_01 = subm_01[order(image)]
subm_02B = subm_02B[order(image)]
#subm_03 = subm_03[order(image)]
subm_01
subm_02B
table(round(subm_02B$NoF, 1))
subm_02B[NoF>0.5]
#subm_03 
subm_04 = subm_02B
cols = names(subm_02B)[-1][-5]
for (var in cols) subm_04[[var]] = (subm_01[[var]]*.30) + (subm_02B[[var]]*.70) 
subm_04 = data.frame(subm_04)
subm_04$NoF = .00999
subm_04[,2:9] = subm_04[,2:9]/rowSums(data.frame(subm_04[,2:9]))
subm_05 = data.table(rbind(subm_04, subm_02A))


# Boost the "LAG" scores.
subm_05[LAG>0.2][["LAG"]] = 1.5
subm_05 = data.frame(subm_05)
subm_05[,2:9] = subm_05[,2:9]/rowSums(data.frame(subm_05[,2:9]))
subm_05 = data.table(subm_05)

subm_05 = subm_05[order(image)]
subm_002 = subm_05
cols = names(subm_05)[-1]
for (var in cols) subm_002[[var]] = (subm_001[[var]]*.5) + (subm_05[[var]]*.5) 

setwd("~/Dropbox/fish/sub")
write.csv(subm_002, "avg_(best)_50_50_(resnet)_1203.csv",  row.names = F)
# 0.74764
