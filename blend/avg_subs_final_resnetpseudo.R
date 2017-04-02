library(data.table)
rm(list=ls())
gc();gc()

# Set working directory 
setwd("~/Dropbox/fish/")

# Load up the sub files 
subm_01   <- fread("sub/subm_full_conv_anno_1.csv")
subm_02   <- fread("sub/subm_full_conv_relabel_2.csv")
subm_03A  <- fread("sub/subm_full_conv_resnet_3A.csv")
subm_03B  <- fread("sub/subm_full_conv_resnet_3B.csv")
subm_03C  <- fread("sub/subm_full_conv_resnet_3C.csv")
subm_04A  <- fread("sub/subm_full_convsq_resnet_4A.csv")
subm_04B  <- fread("sub/subm_full_convsq_resnet_4B.csv")
subm_04C  <- fread("sub/subm_full_convsq_resnet_4C.csv")
subm_06   <- fread("sub/subm_full_conv_pseudo_6.csv")
subm_07A  <- fread("sub/subm_full_convsq_resnet_7A.csv")
subm_07B  <- fread("sub/subm_full_convsq_resnet_7B.csv")
subm_07C  <- fread("sub/subm_full_convsq_resnet_7C.csv")
subm_08A  <- fread("sub/subm_full_pseudo_resnet_8A.csv")

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

# Get the average of the box cropped - yolo 414
subm_07 = subm_07A
cols = names(subm_07)[-1]
for (var in cols) subm_07[[var]] = (subm_07A[[var]] + subm_07B[[var]] + subm_07C[[var]])/3
setnames(subm_07, "image_file", "image")
subm_07 = subm_07[order(image)]
rm(subm_07A, subm_07B, subm_07C)

# Get the average of the box cropped - yolo pseudo
subm_08 = subm_08A
setnames(subm_08, "image_file", "image")
subm_08 = subm_08[order(image)]
rm(subm_08A, subm_08B, subm_08C)

###############################
##  Merge partial data set ####
###############################

# Get the weighted average of the 414 predictions
subm_034 = subm_03
cols = names(subm_034)[-1]
for (var in cols) subm_034[[var]] = (subm_03[[var]]*.5) + (subm_04[[var]]*.5) 
subm_034 = subm_034[order(image)]

# Get the outersect of the 414 and 544 images
img_intrsct = intersect(subm_07$image, subm_034$image)
subm_part01 = subm_07[!image %in% img_intrsct]
subm_part02 = subm_034[!image %in% img_intrsct]

# Get the weighted average of the 544 and 414 predictions
subm_part03 = subm_034[image %in% img_intrsct]
for (var in cols) subm_part03[[var]] = (subm_034[image %in% img_intrsct][[var]]*.5) + (subm_07[image %in% img_intrsct][[var]]*.5) 

# Add all the parts together
subm_part = rbind(subm_part01, subm_part02, subm_part03)
subm_part = subm_part[order(image)]
rm(subm_part01, subm_part02, subm_part03)



# Add in the pseudo
id = intersect(subm_08$image, subm_part$image)
for (var in cols) subm_part[image%in% id][[var]] = (subm_part[image%in% id][[var]]*.75) + (subm_08[image%in% id][[var]]*.25) 


###############################
####  Full data set ###########
###############################
subm_01 = subm_01[order(image)]
subm_02 = subm_02[order(image)]
subm_06 = subm_06[order(image)]

# Get the weighted average
subm_012 = subm_01
cols = names(subm_012)[-1]
for (var in cols) subm_012[[var]] = (subm_01[[var]]*.25) + (subm_02[[var]]*.25)  + (subm_06[[var]]*.5)
subm_012 = subm_012[order(image)]
rm(subm_01, subm_02, subm_06)

###############################
##### Merge both ##############
###############################

subm_final = subm_012
id = subm_part$image
cols = names(subm_final)[-1]
for (var in cols) subm_final[image %in% id][[var]] = (subm_012[image %in% id][[var]]*0.3 +    # Original Preds
                                                        (subm_part[image %in% id][[var]])*0.7)  # Yolo 414 and 544 Preds

###############################
##### Thresholding ############
###############################

# Make the NoF for the yoloimages very low
subm_final[image %in% id]$NoF = .00001
subm_final = data.frame(subm_final)
subm_final[,2:9] = subm_final[,2:9]/rowSums(data.frame(subm_final[,2:9]))
subm_final = data.table(subm_final)

# Boost the scores of the very seldom classes. 
subm_final[LAG>0.4][["LAG"]] = 10
subm_final[NoF>0.4][["NoF"]] = 10
subm_final[OTHER>0.4][["OTHER"]] = 10
subm_final = data.frame(subm_final)
subm_final[,2:9] = subm_final[,2:9]/rowSums(data.frame(subm_final[,2:9]))
subm_final = data.table(subm_final)

# Round #1 Sub
write.csv(subm_final, paste0("sub/final-add-544yolo-addResnetpseudo.csv"), row.names = F)


# Apr 1st - Adding pseu VGG : 0.493 from 0.508 
# Apr 1st - Adding yolo 544 : 0.508 from 0.528 

