library(data.table)
rm(list=ls())
gc();gc()
subm_01A <- fread("~/Dropbox/fish/sub/subm_full_resnet_cut0.7_20170316.csv")
subm_01B <- fread("~/Dropbox/fish/sub/subm_full_resnet_cut0.7_20170320.csv")
subm_01C <- fread("~/Dropbox/fish/sub/subm_full_resnet_box_cut0.7_20170322.csv")
subm_01D <- fread("~/Dropbox/fish/sub/subm_full_resnet_box_cut0.7_20170322A.csv")
subm_01E <- fread("~/Dropbox/fish/sub/subm_full_resnet_box_cut0.7_20170324A.csv") # Yolo 544
rfcn <- fread("~/Dropbox/fish/sub/rfcn.csv")

yolo <- data.table(read.table("~/Downloads/comp4_det_test_FISH544.txt", quote="\"", comment.char=""))
setnames(yolo, c("image", "proba", "x0", "y0", "x1", "y1"))


subm_01 = subm_01A
cols = names(subm_01)[-1]
for (var in cols) subm_01[[var]] = (subm_01A[[var]]*.2) + (subm_01B[[var]]*.2) + (subm_01C[[var]]*.3) + (subm_01D[[var]]*.3)
subm_01A
subm_01B
subm_01C
subm_01D
subm_01
rm(subm_01A, subm_01B, subm_01C, subm_01D)

# Bring in the yolo 544
subm_01
subm_01E
length(intersect(subm_01$image_file, subm_01E$image_file))

# Get a dataframe of the intersect and outsersect
id = subm_01$image_file %in% subm_01E$image_file
idE = subm_01E$image_file %in% subm_01$image_file
table(idE)
subm_01tmp = subm_01
for (var in cols) subm_01tmp[id][[var]] = (subm_01[id][[var]]*.5) + (subm_01E[idE][[var]]*.5)
subm_01 = subm_01tmp
subm_01  = rbind(subm_01, subm_01E[!idE])


subm_002A <- fread("~/Dropbox/fish/sub/avg_2_best_50_50_morebags_0303.csv")[order(image)] # VGG
subm_002B <- fread("~/Dropbox/fish/sub/subm_full_pseudo_20170320B.csv")[order(image)]     # Pseudo
subm_002C <- fread("~/Dropbox/fish/sub/subm_full_all_resnet_20170325B.csv")[order(image)] # Resnet
subm_02 = subm_002A
cols = names(subm_02)[-1]

for (var in cols) subm_02[[var]] = 0.7*((subm_002A[[var]]*.3) + (subm_002B[[var]]*.3) + (subm_002C[[var]]*.4)) + 0.3*rfcn[[var]]
rm(subm_002A, subm_002B)

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
for (var in cols) subm_04[[var]] = (subm_01[[var]]*.7) + (subm_02B[[var]]*.3) 
# for (var in cols) subm_04[[var]] = (subm_01[[var]]*.75) + (subm_02B[[var]]*.25) 

# Make the NoF for the yoloimages very low
subm_04 = data.frame(subm_04)
subm_04$NoF = .00001
subm_04[,2:9] = subm_04[,2:9]/rowSums(data.frame(subm_04[,2:9]))
subm_05 = data.table(rbind(subm_04, subm_02A))

#write.csv(subm_05, "check.csv", row.names = F)

subm_05[LAG>0.4 | NoF >0.4 | OTHER>0.4 | BET>0.4 | SHARK>0.4| DOL>0.4 | YFT>0.4 | ALB > 0.8 ]

# Boost the "LAG" scores. (This adds about 0.005 to score - not huge)
subm_05[LAG>0.4][["LAG"]] = 5
subm_05[NoF>0.4][["NoF"]] = 5
subm_05[OTHER>0.5][["OTHER"]] = 5
#subm_05[BET>0.4][["BET"]] = 10
subm_05[SHARK>0.4][["SHARK"]] = 5
subm_05[SHARK>0.6][["YFT"]] = 5
subm_05 = data.frame(subm_05)
subm_05[,2:9] = subm_05[,2:9]/rowSums(data.frame(subm_05[,2:9]))
subm_05 = data.table(subm_05)

# Get the nearest neighbors
same =fread("~/Dropbox/fish/test_fish_same.csv") # run .ipynb Clustering Images in Test Set 
same = same[order(image, nchar(group))]
# We have some duplicates so aggregate over them
same = same[,(paste(group, collapse=" ")), by="image"]
same
same$group = unlist(lapply(strsplit(same$V1, " "), 
                           function(x) paste(sort(unique(x)), collapse=" ")))
same[,V1:=NULL]
# Now lets add the groups to the submission
subm_06 = merge(subm_05, same, all.x=T, by="image")
subm_06[["group"]][is.na(subm_06[["group"]])] = paste0("single_", 1:sum(is.na(subm_06[["group"]])))
subm_06tmp =copy(subm_06)

setDT(subm_06)[, ALB := mean(ALB), by = "group"]
setDT(subm_06)[, BET := mean(BET), by = "group"]
setDT(subm_06)[, DOL := mean(DOL), by = "group"]
setDT(subm_06)[, LAG := mean(LAG), by = "group"]
setDT(subm_06)[, NoF := mean(NoF), by = "group"]
setDT(subm_06)[, OTHER := mean(OTHER), by = "group"]
setDT(subm_06)[, SHARK := mean(SHARK), by = "group"]
setDT(subm_06)[, YFT := mean(YFT), by = "group"]

subm_06[order(group)]
subm_06tmp[order(group)]
rowSums(subm_06[,2:9,with=F])
subm_06[,group:=NULL]

setwd("~/Dropbox/fish/sub")
  write.csv(subm_06, "similar_groups_avg_(boostALL)(resnet_box_over_0.7_full)_70_30_(avg_2_best)_2803B.csv",  row.names = F)
