
import matplotlib.pyplot as plt 
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
#import keras
import os
from scipy import misc,ndimage

gen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

dest1='/home/arqum/arqum/GandEclassification/train9696/'

files=os.listdir(dest1)

i=1
for file in files:
	st=file.split('.')
	#if(len(st)>2):continue
	# if(file=='Canny5841.jpg' or file == '50_0_2_20170116173533123.jpg.chip.jpg' or file == 'Canny632.jpg'
	#  or file == '53_0_0_20170117190930257.jpg.chip.jpg' or file == '24_1_0_20170117150644092.jpg.chip.jpg' or file == 'Canny14151.jpg'
	#  or file == 'Canny10955.jpg'):continue
	image=np.expand_dims(ndimage.imread(dest1+file),0)
	aug_iter = gen.flow(image)
	aug_image = next(aug_iter)[0].astype(np.uint8)
	string = file.split('_')
	if(len(string)<4):continue
	string=string[0]+'_'+string[1]+'_'+string[2]+'_'+str(i)+string[3]
	misc.imsave(dest1+string, aug_image)
	i=i+1