################################################################################################
# Get the non rare entities per document
################################################################################################
rm(list=ls())
if(Sys.info()['sysname'] == "Linux"){
  source("~/Dropbox/utils.R")
  setwd("~/Dropbox/fish")
}else{
  source("C:/Users/IBM_ADMIN/Dropbox/utils.R")
  setwd("C:/Users/IBM_ADMIN/Dropbox/fish")
}
getPacks()
gc();gc();gc();gc();gc();gc()

dirs = list.files("data/train-all")

files = c()
for( i in dirs ){
  dir = paste0("data/train-all/", i)
  files = c(files, paste0(dir, "/", list.files(dir)))
}

for( d in dirs ) system(paste0("mkdir data/train/", d))
for( d in dirs ) system(paste0("mkdir data/valid/", d))

set.seed(100)
files_trn = sample(files, 3277, replace = F)
files_val = setdiff(files, files_trn)

for(f in files_trn) system(paste("cp", f, gsub("-all", "", f)))
for(f in files_val) system(paste("cp", f, gsub("train-all", "valid", f)))
