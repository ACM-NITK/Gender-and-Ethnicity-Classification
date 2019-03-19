import shutil
import os
import cv2

source2='/home/arqum/arqum/GandEclassification/crop_part1/'
source1='/home/arqum/arqum/GandEclassification/UTKFace/'

count=0
files1 = os.listdir(source1)
files2=os.listdir(source2)

for f1 in files1:
	for f2 in files2:
		if(f1==f2):
			count=count+1
			break
print(count)

