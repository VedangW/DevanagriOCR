#! usr/bin/python

import os
import cv2
import numpy as np

def process_names(images, image_names):
	names = []
	for img in image_names:
	    name = img[:-4]
	    name = name.split("_")
	    name = name[3:]
	    names.append(name)

	#This gave the distinct classes in entire data
	classes = set([])
	for name in names:
	    for c in name:
	        classes.add(int(c))
	classes = list(classes)

	#This gave the mapping from UNICODE to classes
	y = np.zeros((len(images),len(classes)))
	uni_classes = {}
	i = 0
	for w in classes:
	    uni_classes[w] = i
	    i += 1

	#This converts the classes in unicode to 0-79 classes
	new_classes = []    
	for name in names:
	    cl = []
	    for c in name:
	        cl.append(uni_classes[int(c)])
	    new_classes.append(cl)

	#One hot
	y = np.zeros((len(images),80))
	for i in range(len(y)):
	    y[i][new_classes[i]] = 1

	return y

def get_images(directory):
	images = []
	image_names = []
	for filename in os.listdir(directory):
		img = cv2.imread(directory + '/' + filename, cv2.IMREAD_GRAYSCALE)
		image_names.append(filename)
		images.append(img)

	return images, image_names

def preprocess():
	imgs, imgnms = get_images('/home/vedang/Desktop/DevanagriOCR/processed')
	labels = process_names(imgs, imgnms)

	imgs = np.array(imgs)
	labels = np.array(labels)

	x_train = imgs[0:1766]
	x_test = imgs[-200:]

	y_train = labels[0:1766]
	y_test = labels[-200:]

	return x_train, y_train, x_test, y_test


def main():
	# imgs, imgnms = get_images('/home/vedang/Desktop/DevanagriOCR/processed')
	# print ("Images -> ", imgs)
	# print ("No. of images = ", len(imgs))
	# print ("Image names -> ", imgnms)

	# labels = process_names(imgs, imgnms)
	# print ("No. of labels = ", len(labels))
	# print (labels)

	# x_tr, y_tr, x_te, y_te = preprocess()
	# print ("x_train: ", x_tr)
	# print ("y_train: ", y_tr)
	# print ("x_test: ", x_te)
	# print ("y_test: ", y_te)

	# print ("x_train size = ", len(x_tr))
	# print ("y_train size = ", len(y_tr))
	# print ("x_test size = ", len(x_te))
	# print ("y_test size = ", len(y_te))

if __name__ == "__main__":
	main()