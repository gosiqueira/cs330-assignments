import numpy as np
import os
import random
from imageio import imread
from sklearn.utils import shuffle
from pprint import pprint


def get_images(paths, labels, nb_samples=None, shuffle=True):
	"""
	Takes a set of character folders and labels and returns paths to image files
	paired with labels.
	Args:
		paths: A list of character folders
		labels: List or numpy array of same length as paths
		nb_samples: Number of images to retrieve per character
	Returns:
		List of (label, image_path) tuples
	"""
	if nb_samples is not None:
		sampler = lambda x: random.sample(x, nb_samples)
	else:
		sampler = lambda x: x
	images_labels = [(i, os.path.join(path, image))
					 for i, path in zip(labels, paths)
					 for image in sampler(os.listdir(path))]
	if shuffle:
		random.shuffle(images_labels)
	return images_labels


def image_file_to_array(filename, dim_input):
	"""
	Takes an image path and returns numpy array
	Args:
		filename: Image filename
		dim_input: Flattened shape of image
	Returns:
		1 channel image
	"""
	image = imread(filename)
	image = image.reshape([dim_input])
	image = image.astype(np.float32) / 255.0
	image = 1.0 - image
	return image


class DataGenerator(object):
	"""
	Data Generator capable of generating batches of Omniglot data.
	A "class" is considered a class of omniglot digits.
	"""

	def __init__(self, num_classes, num_samples_per_class, config={}):
		"""
		Args:
			num_classes: Number of classes for classification (K-way)
			num_samples_per_class: num samples to generate per class in one batch
			batch_size: size of meta batch size (e.g. number of functions)
		"""
		self.num_samples_per_class = num_samples_per_class
		self.num_classes = num_classes

		data_folder = config.get('data_folder', './omniglot_resized/')
		self.img_size = config.get('img_size', (28, 28))

		self.dim_input = np.prod(self.img_size)
		self.dim_output = self.num_classes

		character_folders = [os.path.join(data_folder, family, character)
							 for family in os.listdir(data_folder)
							 if os.path.isdir(os.path.join(data_folder, family))
							 for character in os.listdir(os.path.join(data_folder, family))
							 if os.path.isdir(os.path.join(data_folder, family, character))]

		random.seed(1)
		random.shuffle(character_folders)
		num_val = 100
		num_train = 1100
		self.metatrain_character_folders = character_folders[: num_train]
		self.metaval_character_folders = character_folders[
			num_train:num_train + num_val]
		self.metatest_character_folders = character_folders[
			num_train + num_val:]

	def sample_batch(self, batch_type, batch_size):
		"""
		Samples a batch for training, validation, or testing
		Args:
			batch_type: train/val/test
		Returns:
			A a tuple of (1) Image batch and (2) Label batch where
			image batch has shape [B, K, N, 784] and label batch has shape [B, K, N, N]
			where B is batch size, K is number of samples per class, N is number of classes
		"""
		if batch_type == "train":
			folders = self.metatrain_character_folders
		elif batch_type == "val":
			folders = self.metaval_character_folders
		else:
			folders = self.metatest_character_folders

		#############################
		#### YOUR CODE GOES HERE ####
		all_image_batches, all_label_batches = [], []

		for _ in range(batch_size):
			sampled_character_folders = random.sample(folders, self.num_classes)
			labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False)
			labels = [li[0] for li in labels_and_images]
			images = [image_file_to_array(li[1], self.dim_input) for li in labels_and_images]
			ims = np.stack(images)

			labels = np.array(labels)
			labels = np.reshape(labels, (self.num_classes, self.num_samples_per_class))
			labels = np.eye(self.num_classes)[labels]

			ims = np.reshape(ims, (self.num_classes, self.num_samples_per_class, -1))

			batch = np.concatenate([labels, ims], 2)
			
			for p in range(self.num_samples_per_class):
				np.random.shuffle(batch[:, p])

			labels = batch[:, :, :self.num_classes]
			ims = batch[:, :, self.num_classes:]
			
			labels = np.swapaxes(labels, 0, 1)
			ims = np.swapaxes(ims, 0, 1)

			all_image_batches.append(ims)
			all_label_batches.append(labels)
		all_image_batches = np.stack(all_image_batches)
		all_label_batches = np.stack(all_label_batches)
		#############################

		return all_image_batches, all_label_batches
