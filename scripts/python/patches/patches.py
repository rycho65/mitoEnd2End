### patches.py


### input: paths: linux path for rawTIff and ilastikMasks
### parameters: rawTiff, ilastikMasks
### logic: split up each image pair (rawTiff, ilastikMasks) into a separate patches
# 
# return: folder with patches


### return folder shape:
### ret_folder/
###   patch_0.tiff (2 channels, 1st channel is rawTiff, 2nd channel is ilastikMasks)
###                 (X, Y, 2)





### desired patch amount: 
## patches = 6 (only even)
## 