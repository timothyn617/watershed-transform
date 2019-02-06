import data
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

WEIGHTS_WATERSHED_PATH = os.getcwd() + '/weights_watershed.pth' # point to trained watershed weights file
WEIGHTS_CLASSIFIER_PATH = os.getcwd() + '/weights_classifier.pth' # point to trained ternary classifier weights file


def watershed_to_num_lenses(watershed_mask, cutoff_level):
	predictions = np.zeros_like(watershed_mask).astype(int)
	predictions[watershed_mask <= cutoff_level] = 1
	component_image = skimage.measure.label(predictions)
	num_lenses = np.max(component_image)
	return num_lenses

def watershed_predictions_df(weights_path, dataset='train'):
	assert dataset in ['train', 'validation']
	train_loader, val_loader = data.get_dataloaders(batch_size=10, augment=False, skip_no_lenses_frames=False, train_shuffle=False)
	data_loader = train_loader if dataset=='train' else val_loader
	model = unet.unet_model.UNet(1,NUM_CLASSES)
	D = torch.load(weights_path)
	model.load_state_dict(D)
	model.cuda().eval()
	weights_dir = os.path.dirname(weights_path)
	os.makedirs(weights_dir + '/%s_prediction/' % dataset, exist_ok=True)

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
			pil.save(weights_dir + '/%s_prediction/%s-%s-prediction.png' % (dataset, dataset, str(img_no).zfill(5)))
			for n in range(NUM_CLASSES):
				ret[n].append(watershed_to_num_lenses(prediction.numpy(), n))
	df=pd.DataFrame(ret)
	df.to_csv(os.path.dirname(weights_path) + '/%s_watershed_pred.csv' % dataset, index=False)
	df = pd.DataFrame({'label': df[9]})
	df.to_csv(os.path.dirname(weights_path) + '/%s_pred.csv' % dataset, index=False)

def watershed_predictions_on_test_df(weights_path):
	test_loader = data.get_test_dataloader(batch_size=10, num_channels=1)
	model = unet.unet_model.UNet(1, NUM_CLASSES).cuda()
	D = torch.load(weights_path)
	model.load_state_dict(D)
	model.eval()
	weights_dir = os.path.dirname(weights_path)
	os.makedirs(weights_dir + '/test_prediction/', exist_ok=True)

	ret = {n: [] for n in range(NUM_CLASSES)}
	img_no = -1
	for i, (image,_) in enumerate(test_loader):
		if i % 10 == 0:
			print('Processing batch %d' % i)
		output = model(image.cuda())
		output = output.detach().cpu()
		predictions = torch.argmax(output, dim=1)
		for prediction in predictions:
			img_no += 1
			prediction_np = prediction.numpy()
			pil = Image.fromarray((prediction_np*255.0/np.clip(np.max(prediction_np),1,None)).astype(np.uint8))
			pil.save(weights_dir + '/test_prediction/test-%s-prediction.png' % str(img_no).zfill(5))
			for n in range(NUM_CLASSES):
				ret[n].append(watershed_to_num_lenses(prediction_np, n))
		df = pd.DataFrame(ret)
		df.to_csv(weights_dir + '/test_watershed_pred.csv', index=False)
		df = pd.DataFrame({'label':df[9]}) # use lens predictions based on level sets up to the 9th
		df.to_csv(weights_dir + '/test_pred.csv', index=False)

def test_classifier_df(weights_path):
	train_loader, val_loader = data.get_classifier_dataloaders(augment=False, batch_size=10,train_shuffle=False)
	test_loader = data.get_test_dataloader(batch_size=10)
	model = torchvision.models.resnet34().cuda()
	D = torch.load(weights_path)
	model.load_state_dict(D)
	weights_dir = os.path.dirname(weights_path)
	model.eval()

	for name, loader in zip(['train','val','test'],[train_loader, val_loader, test_loader]):
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

	watershed_predictions_df(WEIGHTS_WATERSHED_PATH,'train')
	# watershed_predictions_df(WEIGHTS_WATERSHED_PATH, 'validation')
	# watershed_predictions_on_test_df(WEIGHTS_WATERSHED_PATH)
	#
	# test_classifier_df(WEIGHTS_CLASSIFIER_PATH)