library(data.table)
rm(list=ls())
#subm_01 <- fread("~/Dropbox/fish/sub/subm_full_crop_of_20170305A.csv")
subm_02 <- fread("~/Dropbox/fish/sub/avg_cropped_with_best_50_50_0503A.csv")

classes = c("SHARK", "DOL", "YFT", "LAG", "BET", "ALB")

for (c in classes){
  yolodf = fread(paste0("~/Dropbox/fish/yolo_coords/comp4_det_test_", c, ".txt"))
  yolodf = yolodf[V2>0.8]
  yolodf = yolodf[,.(max(V2)), by=V1]
  setnames(yolodf, c("V1", "V2"))
  subm_02[image %in% paste0(yolodf$V1, ".jpg")][[c]] = subm_02[image %in% paste0(yolodf$V1, ".jpg")][[c]] + yolodf$V2
}

yolodf = fread(paste0("~/Dropbox/fish/yolo_coords/comp4_det_test_FISH.txt"))
yolodf = yolodf[,.(max(V2)), by=V1]
setnames(yolodf, c("image", "proba"))
sort(yolodf[proba>0.1 & proba<0.15]$image)
subm_02[image %in% paste0(yolodf[proba<0.05]$image, ".jpg")][["NoF"]] = 10
#id = paste0(yolodf[proba < 0.05]$image, ".jpg")
#subm_02[image %in% id]

subm_02[,2:9] = subm_02[,2:9]/rowSums(data.frame(subm_02[,2:9]))
setwd("~/Dropbox/fish/sub")
write.csv(subm_02, "~/Dropbox/fish/sub/best_0803_(avg_cropped_with_best_50_50_0503A.)_yoloA.csv",  row.names = F)

