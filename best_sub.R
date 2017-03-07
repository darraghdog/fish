library(data.table)
rm(list=ls())
#subm_01 <- fread("~/Dropbox/fish/sub/subm_full_crop_of_20170305A.csv")
subm_02 <- fread("~/Dropbox/fish/sub/best_0303_separate_nof_classifier.csv")

classes = c("SHARK", "DOL", "YFT", "LAG", "BET")

for (c in classes){
  yolodf = fread(paste0("~/Dropbox/fish/yolo_coords/comp4_det_test_", c, ".txt"))
  yolodf = yolodf[V2>0.85]
  subm_02[image %in% paste0(yolodf$V1, ".jpg")][[c]] = subm_02[image %in% paste0(yolodf$V1, ".jpg")][[c]] + yolodf$V2
}

yolodf = fread(paste0("~/Dropbox/fish/yolo_coords/comp4_det_test_", "ALB", ".txt"))
for (c in classes) yolodf = rbind(yolodf, fread(paste0("~/Dropbox/fish/yolo_coords/comp4_det_test_", c, ".txt")))
yolodf = yolodf[,.(max(V2)), by=V1]
setnames(yolodf, c("image", "proba"))
subm_02[!image %in% paste0(yolodf$image, ".jpg")][["NoF"]] = subm_02[!image %in% paste0(yolodf$image, ".jpg")][["NoF"]] + 1
#id = paste0(yolodf[proba < 0.05]$image, ".jpg")
#subm_02[image %in% id]

subm_02[,2:9] = subm_02[,2:9]/rowSums(data.frame(subm_02[,2:9]))
setwd("~/Dropbox/fish/sub")
write.csv(subm_02, "~/Dropbox/fish/sub/best_0303_separate_nof_classifier_mixed_with_yolo.csv",  row.names = F)

