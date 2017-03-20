library(data.table)
rm(list=ls())
subm_01A <- fread("~/Dropbox/fish/sub/subm_full_resnet_cut0.7_20170316.csv")
subm_01B <- fread("~/Dropbox/fish/sub/subm_full_resnet_cut0.7_20170320.csv")
subm_01 = subm_01A
cols = names(subm_01)[-1]
for (var in cols) subm_01[[var]] = (subm_01A[[var]]*.5) + (subm_01B[[var]]*.5)
subm_01A
subm_01B
subm_01
rm(subm_01A, subm_01B)

subm_02 <- fread("~/Dropbox/fish/sub/avg_2_best_50_50_morebags_0303.csv")
setnames(subm_01, "image_file", "image")
subm_01 = subm_01[,colnames(subm_02), with=F]

subm_02B = subm_02[image %in% subm_01$image]
subm_02A = subm_02[!image %in% subm_01$image]

subm_01 = subm_01[order(image)]
subm_02B = subm_02B[order(image)]
subm_01
subm_02B

#subm_03 
subm_04 = subm_02B
cols = names(subm_02B)[-1]
# for (var in cols) subm_04[[var]] = (subm_01[[var]]*.7) + (subm_02B[[var]]*.3) 
for (var in cols) subm_04[[var]] = (subm_01[[var]]*.75) + (subm_02B[[var]]*.25) 
subm_04 = data.frame(subm_04)
subm_04$NoF = .000999
subm_04[,2:9] = subm_04[,2:9]/rowSums(data.frame(subm_04[,2:9]))
subm_05 = data.table(rbind(subm_04, subm_02A))

# Boost the "LAG" scores.
subm_05[LAG>0.5][["LAG"]] = 10
subm_05 = data.frame(subm_05)
subm_05[,2:9] = subm_05[,2:9]/rowSums(data.frame(subm_05[,2:9]))
subm_05 = data.table(subm_05)

setwd("~/Dropbox/fish/sub")
write.csv(subm_05, "avg_boosted_LAG_(resnet_cut_yolo_over_0.7_full_bag)_70_30_(avg_2_best_50_50_morebags_0303)_2003.csv",  row.names = F)
