import os, sys, shutil
from scipy import io

"""
Usage:
  python scripts/dog.py /data/dog
"""

def read(filename):
    with open(filename) as f:
        return f.readlines()

def main():
    datadir = sys.argv[1]
    count = 0
    for split in ['train', 'test']:
        for c in os.listdir(os.path.join(datadir, 'Images')):
            os.makedirs(os.path.join(datadir, split, c))
        files = io.loadmat(os.path.join(datadir, split + '_list.mat'))['file_list']
        for f in files:
            shutil.copy(os.path.join(datadir, 'Images', f[0][0]),
                        os.path.join(datadir, split, f[0][0]))
            count += 1
    print(count, 'Done')

if __name__ == '__main__':
    main()
