import shutil
import os
import cv2
import numpy as np

source2='/home/arqum/arqum/GandEclassification/crop_part1/'
source1='/home/arqum/arqum/GandEclassification/UTKFace/'
dest1='/home/arqum/arqum/GandEclassification/train9696/'
dest2='/home/arqum/arqum/GandEclassification/test9696/'
files = os.listdir(source1)
#files1=os.listdir()
#print(
dim=(96,96)
i=0

np.random.shuffle(files)

for f in files:
	img = cv2.imread(source1+f, cv2.IMREAD_UNCHANGED)
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	if(f.endswith('.jpg')):
		if(i>15000):
			cv2.imwrite(dest2+f,resized)
		else:
			cv2.imwrite(dest1+f,resized)
	i=i+1
	#print(i)

# files=os.listdir(source2)
# for f in files:
# 	img = cv2.imread(source2+f, cv2.IMREAD_UNCHANGED)
# 	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# 	if(f.endswith('.jpg')):
# 		#if(i<16000):
# 		#	cv2.imwrite(dest1+f,resized)
# 		#else:
# 		cv2.imwrite(dest2+f,resized)
# 	i=i+1
	#print(i)
#files=os.listdir(source3)
#for f in files:
	#if(i<21869 or i>21875):
	#	img = cv2.imread(source3+f, cv2.IMREAD_UNCHANGED)
	#	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	#	if(f.endswith('.jpg')):
	#		cv2.imwrite(dest2+f,resized)
	#i=i+1
	#print(i)