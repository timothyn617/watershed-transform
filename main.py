import cmd_args
from train import Trainer
import os
import zipfile


def _maybe_unpack_files(dataset_dir):
	for mode in ['training', 'validation', 'test']:
		zip_path = dataset_dir + '/%s.zip' % mode
		target_dir = dataset_dir + '/%s' % mode
		if os.path.exists(target_dir): continue
		with zipfile.ZipFile(zip_path, 'r') as f:
			print('Extracting %s' % zip_path)
			f.extractall(target_dir)

if __name__ == '__main__':
	args = cmd_args.Args()
	_maybe_unpack_files(args.dataset_dir)
	trainer = Trainer(args)
	trainer.train()
