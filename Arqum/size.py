import shutil
import os
import cv2

source1='/home/arqum/arqum/GandEclassification/crop_part1/'
source2='/home/arqum/arqum/GandEclassification/UTKFace/'

files=os.listdir(source1)


mi=10000

for f in files :
	img = cv2.imread(source1+f, cv2.IMREAD_UNCHANGED)
	#print(img.shape)
	#break
	

files=os.listdir(source1)

for f in files:
	img = cv2.imread(source2+f, cv2.IMREAD_UNCHANGED)
	#print(img.shape)
	#break	
