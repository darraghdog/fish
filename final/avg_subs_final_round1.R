library(data.table)
rm(list=ls())
gc();gc()

# Set working directory 
setwd("~/fish/")

# Load up the sub files 
subm_01   <- fread("sub/subm_full_conv_anno_1.csv")
subm_02   <- fread("sub/subm_full_conv_relabel_2.csv")
subm_03A  <- fread("sub/subm_full_conv_resnet_3A.csv")
subm_03B  <- fread("sub/subm_full_conv_resnet_3B.csv")
subm_03C  <- fread("sub/subm_full_conv_resnet_3C.csv")
subm_04A  <- fread("sub/subm_full_convsq_resnet_4A.csv")
subm_04B  <- fread("sub/subm_full_convsq_resnet_4B.csv")
subm_04C  <- fread("sub/subm_full_convsq_resnet_4C.csv")

# Load up the yolo bounding boxes
yolo <- data.table(read.table("yolo_coords/comp4_det_test_FISH544.txt", quote="\"", comment.char=""))
setnames(yolo, c("image", "proba", "x0", "y0", "x1", "y1"))

###############################
####  Partial  data set #######
###############################
# Get the average of the original cropped - yolo 414
subm_03 = subm_03A
cols = names(subm_03)[-1]
for (var in cols) subm_03[[var]] = (subm_03A[[var]] + subm_03B[[var]] + subm_03C[[var]])/3
setnames(subm_03, "image_file", "image")
subm_03 = subm_03[order(image)]
rm(subm_03A, subm_03B, subm_03C)

# Get the average of the box cropped - yolo 414
subm_04 = subm_04A
cols = names(subm_04)[-1]
for (var in cols) subm_04[[var]] = (subm_04A[[var]] + subm_04B[[var]] + subm_04C[[var]])/3
setnames(subm_04, "image_file", "image")
subm_04 = subm_04[order(image)]
rm(subm_04A, subm_04B, subm_04C)


###############################
####  Full data set ###########
###############################
subm_01 = subm_01[order(image)]
subm_02 = subm_02[order(image)]

# Get the weighted average
subm_012 = subm_01
cols = names(subm_012)[-1]
for (var in cols) subm_012[[var]] = (subm_01[[var]]*.5) + (subm_02[[var]]*.5) 
subm_012 = subm_012[order(image)]


###############################
##### Merge both ##############
###############################

subm_final = subm_012
id = subm_03$image
cols = names(subm_final)[-1]
for (var in cols) subm_final[image %in% id][[var]] = (subm_012[image %in% id][[var]]*0.3 +    # Original Preds
                                                        (subm_03[image %in% id][[var]]*0.4 + subm_04[image %in% id][[var]]*0.6)*0.7)  # Yolo 414 Preds

###############################
##### Thresholding ############
###############################

# Make the NoF for the yoloimages very low
subm_final[image %in% id]$NoF = .00001
subm_final = data.frame(subm_final)
subm_final[,2:9] = subm_final[,2:9]/rowSums(data.frame(subm_final[,2:9]))
subm_final = data.table(subm_final)

# Boost the scores of the very seldom classes. 
subm_final[LAG>0.5][["LAG"]] = 10
subm_final[NoF>0.5][["NoF"]] = 10
subm_final[OTHER>0.5][["OTHER"]] = 10
subm_final = data.frame(subm_final)
subm_final[,2:9] = subm_final[,2:9]/rowSums(data.frame(subm_final[,2:9]))
subm_final = data.table(subm_final)

# Round #1 Sub
write.csv(subm_final, paste0("sub/final-round1-input-pseudo.csv"), row.names = F)


