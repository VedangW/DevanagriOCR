import os
import cv2
import numpy as np

def process_image(kernel, filepath):
	# Read the image as grayscale
	img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

	resized = cv2.resize(img, (218, 192), interpolation=cv2.INTER_AREA)

	# Denoising by blurring
	img1 = cv2.fastNlMeansDenoising(resized, 60, 7, 21)

	# Applying Otsu's binarization
	ret, thresh = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# Morphological transforms - opening and closing the picture
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

	final = cv2.resize(closing, (130, 118), interpolation=cv2.INTER_AREA)

	return img, final

def process_folder():
	# Paths to images
	train_path = '/home/vedang/Desktop/DevanagriOCR/train_images/'
	process_path = '/home/vedang/Desktop/DevanagriOCR/processed/'

	# Kernel as 5 x 5 matrix of ones
	kernel = np.ones((5,5),np.uint8)

	# Open all files in the folder, process them and store
	for filename in os.listdir(train_path):
		filepath = train_path + filename
		img, p_image = process_image(kernel, filepath)
		proc_name = process_path + filename
		print ("Processing " + filename)
		cv2.imwrite(proc_name, p_image)


def main():
	kernel = np.ones((5,5),np.uint8)

	filepath = '/home/vedang/Desktop/DevanagriOCR/train_images/page2_19_9_2405.png'

	i, p = process_image(kernel, filepath)

	cv2.imwrite('/home/vedang/Desktop/DevanagriOCR/processed/written.png', p)

	cv2.imshow('original image', i)
	cv2.imshow('final', p)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	"""
	process_folder()
	"""

if __name__ == "__main__":
	main()
