import pandas as pd
import torch
import glob
import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image
import torchvision.transforms as transforms
import os
# transforms = use flips
main_dir = os.path.dirname(__file__)

TRAIN_FILES_PATH = main_dir + '/training'
VAL_FILES_PATH = main_dir + '/validation'
TEST_FILES_PATH = main_dir + '/test'
TRAIN_LENSES_CSV_PATH = main_dir + '/training-lenses.csv'
VAL_LENSES_CSV_PATH = main_dir + '/validation-lenses.csv'
TRAIN_LABELS_CSV_PATH = main_dir + '/training-labels.csv'
VAL_LABELS_CSV_PATH = main_dir + '/validation-labels.csv'


class ClassifierDataset(torch.utils.data.Dataset):

	def __init__(self, file_path, labels_csv, augment=False, num_channels=3):
		self.files = sorted(glob.glob(file_path + '/*'))
		self.labels_csv = pd.read_csv(labels_csv) if labels_csv else None
		self.len = len(self.files)
		self.augment = augment
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		if augment:
			transform_list = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.Resize((224,224)), transforms.ToTensor(), normalize]
		else:
			transform_list = [transforms.Resize((224,224)), transforms.ToTensor(), normalize]
		self.transform = transforms.Compose(transform_list)
		self.num_channels = num_channels

	def __len__(self):
		return self.len

	def __getitem__(self,i):
		if self.num_channels == 3:
			image = np.array(Image.open(self.files[i])).reshape(256, 256, 1)
			image = np.concatenate([image, image, image], axis=2)
			image = Image.fromarray(image)
			image = self.transform(image)
		elif self.num_channels == 1:
			image = torch.tensor(np.array(Image.open(self.files[i])).reshape(1,256, 256).astype(np.float32))
		if self.labels_csv is not None:
			label = torch.tensor(np.asscalar(np.squeeze(self.labels_csv.iloc[i]['label'])), dtype=torch.int64)
		else:
			label = torch.tensor([], dtype=torch.int64)
		return image, label

class DistanceDataset(torch.utils.data.Dataset):

	def __init__(self, file_path, lenses_csv, augment=False, skip_no_lenses_frames=True, watershed_endpoints=None):
		'''
			watershed_levels: either None or a list [d0,d1, d2, ..., dn] where d0 = 0 < d1 < d2 < ... < dn. Bin distances [d0,d1], ..., [dn-1,dn]
		'''
		self.files = sorted(glob.glob(file_path + '/*'))
		self.lenses_csv = pd.read_csv(lenses_csv)
		self.len = len(self.files)
		self.skip_no_lenses_frames = skip_no_lenses_frames
		if watershed_endpoints is not None:
			assert watershed_endpoints[0] == 0
			for i in range(len(watershed_endpoints)-1):
				assert watershed_endpoints[i] < watershed_endpoints[i+1]
		self.watershed_endpoints= watershed_endpoints
		self.augment = augment

	def __len__(self):
		return self.len

	def _get_centers(self, i):
		df = self.lenses_csv
		return get_centers(df,i)

	def get_image(self,i):
		i = i % self.len
		image = np.array(Image.open(self.files[i])).reshape(1,256,256).astype(np.float32)
		return image

	def _dist_to_watershed(self, dist_mask):
		endpoints = self.watershed_endpoints
		return dist_to_watershed(dist_mask,endpoints)

	def get_label(self, i):
		i = i % len(self.lenses_csv)
		label = get_discretized_distance_mask(self.lenses_csv, i)
		if self.watershed_endpoints:
			label = self._dist_to_watershed(label)
		return label

	def __getitem__(self, i):
		image = self.get_image(i)
		centers = self._get_centers(i)
		if len(centers) == 0 and self.skip_no_lenses_frames:
			print('No lenses found. Skipping datapoint %d' % i)
			return self.__getitem__(i+1)
		label = self.get_label(i)
		if self.augment: # random horizontal/vertical flips
			if np.random.random() < 0.5:
				image, label = np.flip(image,axis=-1), np.flip(label,axis=-1)
			if np.random.random() < 0.5:
				image, label = np.flip(image, axis=-2), np.flip(label, axis=-2)
		image = torch.from_numpy(image.copy())
		label = torch.LongTensor(label.copy())
		return image, label

def get_train_dataset(augment=False, skip_no_lenses_frames=True, watershed_endpoints=None):
	return DistanceDataset(TRAIN_FILES_PATH, TRAIN_LENSES_CSV_PATH, augment=augment,skip_no_lenses_frames=skip_no_lenses_frames,watershed_endpoints=watershed_endpoints)

def get_classifier_train_dataset(augment=False):
	return ClassifierDataset(TRAIN_FILES_PATH, TRAIN_LABELS_CSV_PATH, augment=augment)

def get_val_dataset(skip_no_lenses_frames=True, watershed_endpoints=None):
	return DistanceDataset(VAL_FILES_PATH,VAL_LENSES_CSV_PATH, augment=False,skip_no_lenses_frames=skip_no_lenses_frames, watershed_endpoints=watershed_endpoints)

def get_classifier_val_dataset():
	return ClassifierDataset(VAL_FILES_PATH, VAL_LABELS_CSV_PATH, augment=False)

def get_test_dataset(num_channels):
	return ClassifierDataset(TEST_FILES_PATH, labels_csv=None,augment=False, num_channels=num_channels)

def get_dataloaders(batch_size, augment, skip_no_lenses_frames=True, watershed_endpoints=None, train_shuffle=True, val_shuffle=False):
	train_dataset = get_train_dataset(augment=augment,skip_no_lenses_frames=skip_no_lenses_frames,watershed_endpoints=watershed_endpoints)
	val_dataset = get_val_dataset(skip_no_lenses_frames=skip_no_lenses_frames,watershed_endpoints=watershed_endpoints)
	train_dataloder = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
	val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=val_shuffle)
	return train_dataloder, val_dataloader

def get_classifier_dataloaders(batch_size, augment, train_shuffle=True, val_shuffle=False):
	train_dataset = get_classifier_train_dataset(augment=augment)
	val_dataset = get_classifier_val_dataset()
	train_dataloder = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
	val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=val_shuffle)
	return train_dataloder, val_dataloader

def get_test_dataloader(batch_size, num_channels):
	test_dataset = get_test_dataset(num_channels)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
	return test_dataloader

def dist_to_watershed(dist_mask, endpoints):
	dist_mask = dist_mask.copy()
	N = len(endpoints) - 1
	for i in range(N):
		dist_mask[(endpoints[i] <= dist_mask) & (dist_mask < endpoints[i+1])] = i
	dist_mask[endpoints[-1] <= dist_mask] = N
	return dist_mask

def get_centers(df,i):
	# df is a lenses csv
	df = df[df['sky-id'] == i]
	centers = list(zip(df['row'], df['column']))
	return np.array(centers)

def get_discretized_distance_mask(df,i, watershed_endpoints=None):
	# df is a lenses csv
	centers = get_centers(df,i)
	if len(centers) == 0:
		label = (256*np.sqrt(2)*np.ones((1, 256, 256))).round().astype(np.int32)
	else:
		grid_points = np.flip(np.array(list(zip(*map(lambda x: x.reshape(-1), np.meshgrid(np.arange(256), np.arange(256)))))),axis=-1)  # [256^2,2], (y,x) coordinates of a 256x256 array
		label = np.min(cdist(grid_points, centers), axis=1).reshape(1, 256, 256).round().astype(np.int32)
	if watershed_endpoints:
		label = dist_to_watershed(label, watershed_endpoints)
	return label