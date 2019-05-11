import os, sys, shutil

"""
Usage:
  python cub200.py /data/CUB_200_2011
"""

def read(filename):
    with open(filename) as f:
        return f.readlines()

def main():
    datadir = sys.argv[1]
    images = read(os.path.join(datadir, 'images.txt'))
    splits = read(os.path.join(datadir, 'train_test_split.txt'))
    assert len(images) == len(splits)
    paths = {'train': [], 'test': []}
    for filename, split in zip(images, splits):
        idx1, filename = filename.split()
        idx2, split = split.split()

        assert idx1 == idx2
        if split == '1':
            paths['train'].append(filename)
        else:
            paths['test'].append(filename)
    print('# of training images:', len(paths['train']))
    print('# of test images:', len(paths['test']))

    counter = 0
    for split in ['train', 'test']:
        for d in sorted(os.listdir(os.path.join(datadir, 'images'))):
            os.makedirs(os.path.join(datadir, split, d))

        for p in paths[split]:
            shutil.copy(os.path.join(datadir, 'images', p),
                        os.path.join(datadir, split, p))
            counter += 1
            if counter % 100 == 0:
                print('.', end='')
    print('Done')

if __name__ == '__main__':
    main()
