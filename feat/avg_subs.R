library(data.table)
subm_01 <- fread("~/Dropbox/fish/sub/subm_full_conv_20170220.csv")
subm_02 <- fread("~/Dropbox/fish/sub/avg_2_best_1902.csv")
#subm_03 <- fread("~/Dropbox/fish/sub/subm_full_conv_20170210.csv")
subm_01 = subm_01[order(image)]
subm_02 = subm_02[order(image)]
#subm_03 = subm_03[order(image)]
subm_01
subm_02
#subm_03 
subm_04 = subm_01
cols = names(subm_02)[-1]
for (var in cols) subm_04[[var]] = (subm_01[[var]]*.65) + (subm_02[[var]]*.35) 
setwd("~/Dropbox/fish/sub")
write.csv(subm_04, "avg_2_best_2002.csv",  row.names = F)

###########################################################
library(data.table)
subm <- fread("~/Dropbox/fish/sub/avg_2_best_0702.csv")
nof = c("img_00119.jpg", "img_00196.jpg", "img_00676.jpg", "img_00763.jpg", "img_01171.jpg")
subm[image %in% nof]
