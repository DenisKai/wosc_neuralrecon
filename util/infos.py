from PIL import Image
import glob
import os

path = os.path.expanduser('~/projects/Replica/office0/')

imgs = sorted(glob.glob(path + 'results/frame*.jpg'))
deps = sorted(glob.glob(path + 'results/depth*.png'))
print('n_rgb:', len(imgs), '| n_depth:', len(deps))
print('resolution:', Image.open(imgs[0]).size)  # (W, H)