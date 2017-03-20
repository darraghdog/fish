library(data.table)
rm(list=ls())
subm_01 <- fread("~/Dropbox/fish/sub/subm_full_crop_of_20170310.csv")
subm_02 <- fread("~/Dropbox/fish/sub/avg_2_best_50_50_morebags_0303.csv")
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
for (var in cols) subm_04[[var]] = (subm_01[[var]]*.5) + (subm_02B[[var]]*.5) 
subm_04 = data.frame(subm_04)
subm_04$NoF = .00999
subm_04[,2:9] = subm_04[,2:9]/rowSums(data.frame(subm_04[,2:9]))
subm_05 = data.table(rbind(subm_04, subm_02A))
setwd("~/Dropbox/fish/sub")
write.csv(subm_05, "avg_(subm_full_crop_of_20170310)_(avg_2_best_50_50_morebags_0303)_50_50_1103.csv",  row.names = F)
# 0.74764

# Boost the "LAG" scores.
subm_05[LAG>0.2][["LAG"]] = 1.5
subm_05 = data.frame(subm_05)
subm_05[,2:9] = subm_05[,2:9]/rowSums(data.frame(subm_05[,2:9]))
subm_05 = data.table(subm_05)

# Incorporate NoF
dfNof = fread("../yolo_coords/comp4_det_test_FISH.txt")
dfNof = dfNof[,.(paste0(V1, ".jpg"), V2)]
setnames(dfNof, c("image", "proba"))

# Check the images with nothing in Yolo
subm_05[!image %in% dfNof$image][NoF > 0.25]

# Check the images with nothing in Yolo
dfNofmax = dfNof[,max(proba), by=image]
setnames(dfNofmax, c("image", "proba"))
subm_05[image %in% dfNofmax[proba<0.05]$image][NoF>0.7 & NoF<1.0]
