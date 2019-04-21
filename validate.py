import dataset
import skimage.measure
import numpy as np
import unet
import torch
import pandas as pd
import os
from PIL import Image
import torchvision

WATERSHED_ENDPOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 10, 15, 25, 35, 55, 100, 200, 362]
NUM_CLASSES = len(WATERSHED_ENDPOINTS)

WEIGHTS_WATERSHED_PATH = os.getcwd() + '/weights/weights_watershed.pth' # point to trained watershed weights file
WEIGHTS_CLASSIFIER_PATH = os.getcwd() + '/weights/weights_classifier.pth' # point to trained ternary classifier weights file


def watershed_to_num_lenses(watershed_mask, cutoff_level):
	'''
		Computes number of lenses via the number of connected components of watershed levels up to the given cutoff level.
	'''
	predictions = np.zeros_like(watershed_mask).astype(int)
	predictions[watershed_mask <= cutoff_level] = 1
	component_image = skimage.measure.label(predictions)
	num_lenses = np.max(component_image)
	return num_lenses

def evaluate_watershed(weights_path):
	'''
		Evaluates model on given dataset and saves images as .png. Also saves predicted of number of classes based on watershed predictions in a csv.
	'''
	for ds in ['train', 'validation', 'test']:
		if ds == 'train':
			data_loader, _ = dataset.get_dataloaders(batch_size=10, augment=False, skip_no_lenses_frames=False, train_shuffle=False)
		elif ds == 'validation':
			_, data_loader = dataset.get_dataloaders(batch_size=10, augment=False, skip_no_lenses_frames=False, train_shuffle=False)
		else:
			data_loader = dataset.get_test_dataloader(batch_size=10, num_channels=1)
		model = unet.unet_model.UNet(1,NUM_CLASSES)
		D = torch.load(weights_path)
		model.load_state_dict(D)
		model = model.cuda()
		model.eval()
		weights_dir = os.path.dirname(weights_path)
		os.makedirs(weights_dir + '/%s_prediction/' % ds, exist_ok=True)

		ret = {n:[] for n in range(NUM_CLASSES)}
		img_no = -1
		for i, (image, _) in enumerate(data_loader):
			if i % 10 == 0:
				print('Processing batch %d' % i)
			output = model(image.cuda())
			output = output.detach().cpu()
			predictions = torch.argmax(output,dim=1)
			for prediction in predictions:
				img_no += 1
				prediction_np = prediction.numpy()
				pil = Image.fromarray((prediction_np * 255.0 / np.clip(np.max(prediction_np), 1, None)).astype(np.uint8))
				pil.save(weights_dir + '/%s_prediction/%s-%s-prediction.png' % (ds, ds, str(img_no).zfill(5)))
				for n in range(NUM_CLASSES):
					ret[n].append(watershed_to_num_lenses(prediction.numpy(), n))
		df=pd.DataFrame(ret)
		df.to_csv(os.path.dirname(weights_path) + '/%s_watershed_pred.csv' % ds, index=False)
		df = pd.DataFrame({'label': df[9]})
		df.to_csv(os.path.dirname(weights_path) + '/%s_pred.csv' % ds, index=False)

def evaluate_classifier(weights_path):
	train_loader, val_loader = dataset.get_classifier_dataloaders(augment=False, batch_size=10, train_shuffle=False)
	test_loader = dataset.get_test_dataloader(batch_size=10, num_channels=3)
	model = torchvision.models.resnet34().cuda()
	D = torch.load(weights_path)
	model.load_state_dict(D)
	weights_dir = os.path.dirname(weights_path)
	model.eval()

	for name, loader in zip(['training','validation','test'],[train_loader, val_loader, test_loader]):
		ret = {'label': []}
		for i, (image,_) in enumerate(loader):
			if i % 10 == 0:
				print('Processing batch %d' % i)
			output = model(image.cuda())
			output = output.detach().cpu()
			predictions = torch.argmax(output, dim=1)
			for prediction in predictions:
				prediction_np = np.asscalar(prediction.numpy())
				ret['label'].append(prediction_np)
			df = pd.DataFrame(ret)
			df.to_csv(weights_dir + '/%s_classifier_pred.csv' % name, index=False)

if __name__ == '__main__':
	#evaluate_watershed(WEIGHTS_WATERSHED_PATH)
	evaluate_classifier(WEIGHTS_CLASSIFIER_PATH)