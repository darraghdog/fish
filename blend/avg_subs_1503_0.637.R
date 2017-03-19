library(data.table)
rm(list=ls())
subm_01 <- fread("~/Dropbox/fish/sub/subm_part_resnet_of_20170315.csv")
subm_02 <- fread("~/Dropbox/fish/sub/avg_2_best_50_50_morebags_0303.csv")
setnames(subm_01, "image_file", "image")
subm_01 = subm_01[,colnames(subm_02), with=F]


subm_01 <- subm_01[, lapply(.SD,mean), by=image]

subm_02B = subm_02[image %in% subm_01$image]
subm_02A = subm_02[!image %in% subm_01$image]

subm_01 = subm_01[order(image)]
subm_02B = subm_02B[order(image)]
subm_01
subm_02B

#subm_03 
subm_04 = subm_02B
cols = names(subm_02B)[-1]
for (var in cols) subm_04[[var]] = (subm_01[[var]]*.50) + (subm_02B[[var]]*.50) 
subm_04 = data.frame(subm_04)
subm_04$NoF = .000999
subm_04[,2:9] = subm_04[,2:9]/rowSums(data.frame(subm_04[,2:9]))
subm_05 = data.table(rbind(subm_04, subm_02A))

# # Boost the "LAG" scores.
# subm_05[LAG>0.2][["LAG"]] = 1.5
# subm_05 = data.frame(subm_05)
# subm_05[,2:9] = subm_05[,2:9]/rowSums(data.frame(subm_05[,2:9]))
# subm_05 = data.table(subm_05)

setwd("~/Dropbox/fish/sub")
write.csv(subm_05, "avg_(resnet_cut_yolo_over_0.8)_50_50_(avg_2_best_50_50_morebags_0303)_1503.csv",  row.names = F)
# 0.63732


# Use only resnet
subm_04 = subm_02B
cols = names(subm_02B)[-1]
for (var in cols) subm_04[[var]] = (subm_01[[var]]*.99) + (subm_02B[[var]]*.01) 
subm_04 = data.frame(subm_04)
subm_04$NoF = .000999
subm_04[,2:9] = subm_04[,2:9]/rowSums(data.frame(subm_04[,2:9]))
subm_05 = data.table(rbind(subm_04, subm_02A))

# # Boost the "LAG" scores.
# subm_05[LAG>0.2][["LAG"]] = 1.5
# subm_05 = data.frame(subm_05)
# subm_05[,2:9] = subm_05[,2:9]/rowSums(data.frame(subm_05[,2:9]))
# subm_05 = data.table(subm_05)

setwd("~/Dropbox/fish/sub")
write.csv(subm_05, "avg_(resnet_cut_yolo_over_0.8)_0.99_0.01_(avg_2_best_50_50_morebags_0303)_1503.csv",  row.names = F)
# 0.68344
