import numpy as np
import cv2
import glob
import os


# Help function to display an image
def display(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# Helper function to save each cropped image as a new file
def save(index, breed, img):
	resize_path = os.path.join(path, "cropped/")
	cv2.imwrite(os.path.join(resize_path, breed + "_cropped_" + str(index) + ".jpg"), img)

breeds = 	["Najdi",
			"Nuaimi",
			"Harri",
			"Sawakni"]

for breed in breeds:

	# Load all .jpg files in the test_data folder
	path = "../Data/" + breed + "/"
	filenames = glob.glob(path + "*.jpg")
	filenames.sort()

	# Load all images as numpy arrays
	images = [(name, cv2.imread(name)) for name in filenames]

	# Initialize X_train
	IMAGE_SIZE = 224
	num_examples = len(images)
	X_train = np.zeros((num_examples, IMAGE_SIZE, IMAGE_SIZE, 3))

	# Print the name and shape of each image
	for m, (name, img) in enumerate(images):

	    # Ensure that each image is square
		H, W, _ = img.shape
		if W > H:
			delta = (W - H) // 2
			img = img[:, delta:delta+H, :]

		elif W < H:
			delta = (H - W) //2
			img = img[delta:delta+W, :, :]

		# Resize all images to have shape IMAGE_SIZE x IMAGE_SIZE x 3
		img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
		
		# Save cropped image
		save(m + 1, breed, img)

		X_train[m] = img

	print('X_train shape: ', X_train.shape)