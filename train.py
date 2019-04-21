import torch
import torch.nn as nn
import torch.optim
from tensorboardX import SummaryWriter
import os
import numpy as np
import glob
import shutil
import dataset
import unet
import copy
import torchvision

WATERSHED_ENDPOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 10, 15, 25, 35, 55, 100, 200, 362]
# neural net will classify pixels as label i if its distance (rounded to the nearest integer) to a lens center is in the half-open interval [WATERSHED_ENDPOINTS[i], WATERSHED_ENDPOINTS[i+1])

NUM_WATERSHED_CLASSES = len(WATERSHED_ENDPOINTS) #= 16

# weighting based on relative sizes of level sets and some ad-hoc normalization/clipping/upweighting of center pixel (level = 0); definitely not optimized!
LOSS_WEIGHTS = np.array([40, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 12.96, 2.6244, 1.,1.]).astype(np.float32)

class Trainer():

	def __init__(self, args):
		self.args = copy.deepcopy(args)
		if args.watershed:
			model = unet.unet_model.UNet(1,NUM_WATERSHED_CLASSES)
			self.num_classes = NUM_WATERSHED_CLASSES
		else:
			model = torchvision.models.resnet34(num_classes=3)
			self.num_classes = 3
		self.model=model
		self.build_datasets()
		self.build_optimizer()

	def build_optimizer(self):
		model = self.model
		args = self.args
		parameters = model.parameters()
		if args.SGD:
			self.optimizer = torch.optim.SGD(parameters, args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
		else:
			self.optimizer = torch.optim.Adam(parameters, args.learning_rate, weight_decay=args.weight_decay)

	def build_datasets(self):
		args = self.args
		if args.watershed:
			self.train_loader, self.val_loader = dataset.get_dataloaders(args.batch_size, augment=True, skip_no_lenses_frames=False, watershed_endpoints=WATERSHED_ENDPOINTS)
		else:
			self.train_loader, self.val_loader = dataset.get_classifier_dataloaders(args.batch_size, augment=True)

	def train(self):
		args = self.args
		args.print_out()
		setup_save(self.model, args)
		self.model = self.model.cuda()
		self.model.train()
		self.writer = SummaryWriter(args.save_dir + '/tensorboard')
		self.writer.add_text('args', str(sorted(args.__dict__.items())))
		for i in range(1,args.epochs+1):
			self.train_epoch(i)
			self.validate_epoch(i)
			self.save_checkpoint(i)
			if args.debug: break


	def loss(self, image, output, label):
		args = self.args
		if args.watershed:
			loss_function = nn.CrossEntropyLoss(weight=torch.tensor(LOSS_WEIGHTS)).cuda()
			# output from (B,n,H,W) to (B,H,W,n)
			for i, j in [[1, 2], [2, 3]]:
				output = torch.transpose(output, i, j)
			loss = loss_function(output.contiguous().view(-1, self.num_classes), label.view(-1))
		else:
			loss_function = nn.CrossEntropyLoss().cuda()
			loss = loss_function(output.view(-1,self.num_classes), label.view(-1))
		return loss


	def train_epoch(self, epoch):
		args = self.args
		model, train_loader, optimizer,writer = self.model, self.train_loader, self.optimizer, self.writer
		model.train()
		train_len = len(train_loader)
		global_step = train_len * (epoch-1)
		adjust_learning_rate(self.optimizer, args, epoch)
		self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
		for i, (image, label) in enumerate(train_loader):
			global_step += 1
			image = image.cuda()
			label = label.cuda()
			output = model(image).cuda()
			loss = self.loss(image,output,label)
			optimizer.zero_grad()
			loss.backward()
			for p in model.parameters():
				if p.grad is not None:
					p.grad.data.clamp_(-args.grad_clip_by_value,args.grad_clip_by_value)
			optimizer.step()
			self.iteration_hook(global_step, loss, image, output, label)
			if args.debug: break

			if i == 0:
				for j in range(image.shape[0]):
					writer.add_image('input/train/%d'%i, viz_format(image[j:j+1]), global_step)
			if args.watershed:
				prediction_mask = torch.argmax(output, dim=1, keepdim=True).view(-1, 1, 256, 256)
				for j in range(image.shape[0]):
					writer.add_image('label/train/%d' % i, viz_format(label[j:j + 1]), global_step)
					writer.add_image('prediction/train/%d'%i, viz_format(prediction_mask[j:j+1]), global_step)

	def save_checkpoint(self, epoch):
		args = self.args
		model = self.model
		if args.save:
			if args.save_freq and epoch % args.save_freq == 0: # if args.save_freq > 0, save with that frequency
				torch.save(model.state_dict(), args.save_dir + '/epoch%s.pth' % str(epoch).zfill(3))
				print('Saving model at epoch %d' % epoch)
			elif epoch == args.epochs: # save at end of training
				torch.save(model.state_dict(), args.save_dir + '/epoch%s.pth' % str(epoch).zfill(3))
				print('Saving model at epoch %d' % epoch)

	def iteration_hook(self, global_step, loss, image, output, label):
		args = self.args
		if summary_checkpoint(global_step) or args.debug or True:
			self.writer.add_scalar('loss/training_loss', loss.data.item(), global_step)
			print('training_loss %.2f' % loss.data.item())

	def validate_epoch(self, epoch):
		args = self.args
		val_loader, model, writer = self.val_loader, self.model, self.writer
		losses = AverageMeter()

		# switch to evaluate mode
		model.eval()
		for i, (image, label) in enumerate(val_loader):
			label = label.cuda()
			image = image.cuda()
			output = model(image).cuda()

			loss = self.loss(image, output, label)
			losses.update(loss.data.item(), image.size(0))

			# images
			if epoch == 0 and i == 0:
				for j in range(image.shape[0]):
					writer.add_image('input/val/%d' % j, viz_format(image[j:j + 1]), epoch)

			if args.watershed and i == 0:
				prediction_mask = torch.argmax(output, dim=1, keepdim=True)
				for j in range(image.shape[0]):
					writer.add_image('label/val/%d' % j, viz_format(label[j:j + 1]), epoch)
					writer.add_image('prediction/val/%d' % j, viz_format(prediction_mask[j:j + 1]), epoch)
			if args.debug: break

		# scalars
		writer.add_scalar('loss/val_loss_avg', losses.avg, epoch)
		print('===== Epoch %d, val_loss_avg: %.2f' % (epoch, losses.avg))

		return losses.avg


def adjust_learning_rate(optimizer, args, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 after args.lr_decay_step steps"""
	lr = args.learning_rate * (0.1 ** (epoch // args.lr_decay_step))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def summary_checkpoint(step):
	if step == 1 or step % 10 == 0: return True

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def viz_format(img):
	assert len(img.shape) == 4, img.shape
	if img.shape[1] == 1:
		img = torch.tensor(img[0,0]*255/torch.max(img[0,0]),dtype=torch.uint8)
	elif img.shape[1] == 3:
		img = torch.tensor(img[0] * 255 / torch.max(img[0]), dtype=torch.uint8)
	return img

def get_save_dir(args):
	return os.getcwd() + '/%s/%s' % (args.exp_folder, args.exp_name)


def setup_save(model, args):
	save_dir = args.save_dir
	os.makedirs(save_dir)
	with open(save_dir + '/args.txt', 'w+') as f:
		f.write(str(vars(args)))
		f.write(str(model))
	os.makedirs(save_dir + '/code')
	scripts = glob.glob(os.getcwd() + '/*.py')
	for script in scripts:
		shutil.copy(script, save_dir + '/code/' + os.path.basename(script))  # save state of code, for reproducibility
	if args.save:
		torch.save(model.state_dict(), save_dir + '/epoch000.pth')
