import argparse
import datetime
import os

main_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_dir', default=main_dir + '/data')
parser.add_argument('--exp_folder', default='watershed') # where model is saved
parser.add_argument('--batch_size', type=int, default=3) # 3 for watershed model (due to memory limitations), use larger batch size for ternary classifier
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--SGD', default='False') # Adam if True
parser.add_argument('--momentum', type=float, default=0.9) # only used if SGD is True
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--lr_decay_step', type=int, default=40)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--debug', default='False', help='Break training after a few steps in debug mode.')
parser.add_argument('--save', default=True, type=bool, help='Save model weights. By default, initialized model and end of training weights are saved.')
parser.add_argument('--save_freq', default=20, type=int, help='Epoch frequency to save model weights. If -1, only do default saving (initialization and end of training)')
parser.add_argument('--name', default='lenses', help='Experiment name prefix')
parser.add_argument('--grad_clip_by_value', type=float, default=1.0)
parser.add_argument('--watershed', default='True', help='Use watershed model or a ternary resnet34 classifier')

class Args(object):

	def __init__(self,D=None):
		'''
			Args object is a vanilla class with attributes corresponding to model hyperperparameters / experiment info.
			Override by passing in a dictionary D.
		'''
		cmd_args = parser.parse_args()
		# strings to boolean
		for k, v in vars(cmd_args).items():
			if v in ['T', 'True']:
				v = True
			elif v in ['F', 'False']:
				v = False
			setattr(self,k,v)
		if D:
			self.update(D)
		self.format()

	def update(self,D):
		for k,v in D.items():
			assert hasattr(self,k)
			print('updated value of %s:' % k, v)
			setattr(self,k,v)
		self.format()


	def print_out(self):
		print('========= Arg attribute/value pairs:')
		for k,v in sorted(vars(self).items()):
			print('%s: ' % k,v)
		print('=========')

	def format(self):
		self.exp_name = self.get_exp_name()
		self.save_dir = self.get_save_dir()

	def get_exp_name(self):

		args = self
		timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

		# name of experiment
		name = args.name

		# optimizer
		args.SGD = True if args.SGD in ['True', True] else False
		if args.SGD:
			if args.learning_rate is None:
				args.learning_rate = 1e-2
			name += '_lr{0}_mmt{1}'.format(args.learning_rate, args.momentum)
		else:
			if args.learning_rate is None:
				args.learning_rate = 1e-4
			name += '_lr{0}_adam'.format(args.learning_rate)

		# timestemp
		name += '_{0}'.format(timestamp)

		return name

	def get_save_dir(self):
		if not self.debug:
			return os.getcwd() + '/%s/%s' % (self.exp_folder, self.exp_name)
		return os.getcwd() + '/debug/%s/%s' % (self.exp_folder, self.exp_name)
