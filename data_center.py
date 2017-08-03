import os
import cv2
from numpy.random import permutation as perm
import numpy as np
from shutil import copyfile

def original_classes_division():
	train_data_folder_path = os.path.join("database", "train", "train")
	train_root_folder_path = os.path.join("database", "train")
	for each_file in os.listdir(train_data_folder_path):
		each_img = cv2.imread(os.path.join(train_data_folder_path, each_file))
		print(each_file)
		if each_file.split('.')[0] == "dog":
			cv2.imwrite(os.path.join(train_root_folder_path, "dogs", each_file), each_img)
		elif each_file.split('.')[0] == "cat":
			cv2.imwrite(os.path.join(train_root_folder_path, "cats", each_file), each_img)
		else:
			print("Wrong file format..")


def split_CV(root_folder_path, train_rates=0.8):
	cats_folder_path = os.path.join(root_folder_path, "cats")
	dogs_folder_path = os.path.join(root_folder_path, "dogs")
	
	# cats folder -> train/cats, validation/cats
	files = [f for f in os.listdir(cats_folder_path)]
	files_size = len(files)
	print("training cats size : ", files_size)
	shuffle_idx = perm(np.arange(files_size))
	trainval_size = int(files_size*train_rates)
	train_idx = shuffle_idx[:trainval_size]
	validation_idx = shuffle_idx[trainval_size:]

	for i in train_idx:
		print("train cat : ", files[i])
		copyfile(os.path.join(cats_folder_path, files[i]), os.path.join(root_folder_path, "train", "cats", files[i]))


	for i in validation_idx:
		print("validation cat : ", files[i])
		copyfile(os.path.join(cats_folder_path, files[i]), os.path.join(root_folder_path, "validation", "cats", files[i]))

	# dogs folder -> train/dogs, validation/dogs
	files = [f for f in os.listdir(dogs_folder_path)]
	files_size = len(files)
	print("training dogs size : ", files_size)
	shuffle_idx = perm(np.arange(files_size))
	trainval_size = int(files_size*train_rates)
	train_idx = shuffle_idx[:trainval_size]
	validation_idx = shuffle_idx[trainval_size:]

	for i in train_idx:
		print("train dog : ", files[i])
		copyfile(os.path.join(dogs_folder_path, files[i]), os.path.join(root_folder_path, "train", "dogs" , files[i]))

	for i in validation_idx:
		print("validation dog : ", files[i])
		copyfile(os.path.join(dogs_folder_path, files[i]), os.path.join(root_folder_path, "validation", "dogs", files[i]))



if __name__ == "__main__":
	split_CV(os.path.join("database", "train"))