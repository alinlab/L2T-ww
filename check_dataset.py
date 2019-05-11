import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.utils.data as data


class FolderSubset(data.Dataset):
    def __init__(self, dataset, classes, indices):
        self.dataset = dataset
        self.classes = classes
        self.indices = indices

        self.update_classes()

    def update_classes(self):
        for i in self.indices:
            img_path, cls = self.dataset.samples[i]
            cls = self.classes.index(cls)
            self.dataset.samples[i] = (img_path, cls)
            
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class CIFARSubset(data.Dataset):
    def __init__(self, dataset, classes, indices):
        self.dataset = dataset
        self.classes = classes
        self.indices = indices

        self.update_classes()

    def update_classes(self):
        for i in self.indices:
            if self.dataset.train:
                self.dataset.train_labels[i] = self.classes.index(self.dataset.train_labels[i])
            else:
                self.dataset.test_labels[i] = self.classes.index(self.dataset.test_labels[i])

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

class SVHNSubset(CIFARSubset):
    def __init__(self, dataset, classes, indices):
        super(SVHNSubset, self).__init__(dataset, classes, indices)

        # self.dataset = dataset
        # self.classes = classes
        # self.indices = indices

        # self.update_classes()

    def update_classes(self):
        for i in self.indices:
            self.dataset.labels[i] = self.classes.index(self.dataset.labels[i])

    # def __getitem__(self, idx):
    #     return self.dataset[self.indices[idx]]

    # def __len__(self):
    #     return len(self.indices)


class FashionMNISTSubset(data.Dataset):
    def __init__(self, dataset, classes, indices):
        self.dataset = dataset
        self.classes = classes
        self.indices = indices


    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class STL10Subset(data.Dataset):
    def __init__(self, dataset, classes, indices):
        self.dataset = dataset
        self.classes = classes
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def check_split(opt):
    if opt.datasplit == 'random':
        import random

        if opt.dataset in ['tinyimagenet', 'tinyimagenet-small']:
            num_classes = 200
            num_train_images = 100000
            num_test_images = 10000

        elif opt.dataset.startswith('cifar'):
            num_classes = int(opt.dataset[5:])
            num_train_images = 50000
            num_test_images = 10000

        else:
            raise Exception('Unknown dataset')
        
        n = random.randint(num_classes // 4, num_classes) # sample a subset of classes
        classes = list(random.sample(list(range(num_classes)), n))
        val_indices = []
        train_indices = []

        m = num_train_images // num_classes
        for c in classes:
            indices = list(range(c*m, (c+1)*m))
            for _ in range(m//10):
                val_indices.append(indices.pop(random.randrange(len(indices))))
            train_indices.extend(indices)

        test_indices = []
        m2 = num_test_images // num_classes
        for c in classes:
            test_indices = test_indices + list(range(c*m2, (c+1)*m2))

        splits = [(classes, train_indices),
                  (classes, val_indices),
                  (classes, test_indices)]

    else:
        splits = []
        for split in ['train', 'val', 'test']:
            splits.append(torch.load('split/' + opt.datasplit + '-' + split))

    return splits


def check_dataset(opt):
    normalize_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406),
                                                                   (0.229, 0.224, 0.225))])
    train_large_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip()])
    val_large_transform = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224)])
    train_small_transform = transforms.Compose([transforms.Pad(4),
                                                transforms.RandomCrop(32),
                                                transforms.RandomHorizontalFlip()])

    if opt.dataset != 'flowers102':
        splits = check_split(opt)

    if opt.dataset in ['tinyimagenet', 'cub200', 'indoor', 'stanford40', 'dog']:
        if opt.dataset == 'tinyimagenet':
            train, val = 'train2', 'val2'
        else:
            train, val = 'train', 'test'
        train_transform = transforms.Compose([train_large_transform, normalize_transform])
        val_transform = transforms.Compose([val_large_transform, normalize_transform])
        sets = [dset.ImageFolder(root=os.path.join(opt.dataroot, train), transform=train_transform),
                dset.ImageFolder(root=os.path.join(opt.dataroot, train), transform=val_transform),
                dset.ImageFolder(root=os.path.join(opt.dataroot, val), transform=val_transform)]
        sets = [FolderSubset(dataset, *split) for dataset, split in zip(sets, splits)]

        opt.num_classes = len(splits[0][0])

    elif opt.dataset == 'flowers102':
        train_transform = transforms.Compose([train_large_transform, normalize_transform])
        val_transform = transforms.Compose([val_large_transform, normalize_transform])
        sets = [dset.ImageFolder(root=os.path.join(opt.dataroot, 'trn'), transform=train_transform),
                dset.ImageFolder(root=os.path.join(opt.dataroot, 'val'), transform=val_transform),
                dset.ImageFolder(root=os.path.join(opt.dataroot, 'tst'), transform=val_transform)]

        opt.num_classes = 102

    elif opt.dataset == 'tinyimagenet-small':
        train_transform = transforms.Compose([transforms.Resize(32),
                                              train_small_transform, normalize_transform])
        val_transform = transforms.Compose([transforms.Resize(32), normalize_transform])
        sets = [dset.ImageFolder(root=os.path.join(opt.dataroot, 'train2'), transform=train_transform),
                dset.ImageFolder(root=os.path.join(opt.dataroot, 'train2'), transform=val_transform),
                dset.ImageFolder(root=os.path.join(opt.dataroot, 'val2'), transform=val_transform)]
        sets = [FolderSubset(dataset, *split) for dataset, split in zip(sets, splits)]

        opt.num_classes = len(splits[0][0])

    elif opt.dataset in ['cifar10', 'cifar100']:
        train_transform = transforms.Compose([train_small_transform, normalize_transform])
        val_transform = normalize_transform
        CIFAR = dset.CIFAR10 if opt.dataset == 'cifar10' else dset.CIFAR100

        sets = [CIFAR(opt.dataroot, download=True, train=True, transform=train_transform),
                CIFAR(opt.dataroot, download=True, train=True, transform=val_transform),
                CIFAR(opt.dataroot, download=True, train=False, transform=val_transform)]
        sets = [CIFARSubset(dataset, *split) for dataset, split in zip(sets, splits)]

        opt.num_classes = len(splits[0][0])

    elif opt.dataset == 'fashion-mnist':
        train_transform = transforms.Compose([transforms.Resize(32), transforms.Grayscale(num_output_channels=3), 
                                              train_small_transform, normalize_transform])
        val_transform = transforms.Compose([transforms.Resize(32), transforms.Grayscale(num_output_channels=3), normalize_transform])
        sets = [dset.FashionMNIST(opt.dataroot, transform=train_transform, download=True, train=True),
                dset.FashionMNIST(opt.dataroot, transform=val_transform, download=True, train=True),
                dset.FashionMNIST(opt.dataroot, transform=val_transform, download=True, train=False)]
        sets = [FashionMNISTSubset(dataset, *split) for dataset, split in zip(sets, splits)]

        opt.num_classes = 10

    elif opt.dataset == 'svhn':
        train_transform = transforms.Compose([train_small_transform, normalize_transform])
        val_transform = normalize_transform
        sets = [dset.SVHN(opt.dataroot, split='train', transform=train_transform, download=True),
                dset.SVHN(opt.dataroot, split='train', transform=val_transform, download=True),
                dset.SVHN(opt.dataroot, split='test', transform=val_transform, download=True)]
        sets = [SVHNSubset(dataset, *split) for dataset, split in zip(sets, splits)]

        opt.num_classes = 10

    elif opt.dataset == 'lsun':
        train_transform = transforms.Compose([transforms.Resize(32),
                                              train_small_transform, normalize_transform])
        val_transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), normalize_transform])
        sets = [dset.LSUN(opt.dataroot, transform=train_transform, classes='train'),
                dset.LSUN(opt.dataroot, transform=val_transform, classes='val'),
                dset.LSUN(opt.dataroot, transform=val_transform, classes='test')]

        opt.num_classes = 10

    elif opt.dataset == 'stl10':
        train_transform = transforms.Compose([transforms.Resize(32),
                                              train_small_transform, normalize_transform])
        val_transform = transforms.Compose([transforms.Resize(32), normalize_transform])
        sets = [dset.STL10(opt.dataroot, split='train', transform=train_transform),
                dset.STL10(opt.dataroot, split='train', transform=val_transform),
                dset.STL10(opt.dataroot, split='test', transform=val_transform)]
        sets = [STL10Subset(dataset, *split) for dataset, split in zip(sets, splits)]

        opt.num_classes = len(splits[0][0])

    elif opt.dataset == 'emnist':
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        val_transform = transforms.ToTensor()
        sets = [EMNIST(opt.dataroot, train=True,  transform=train_transform),
                EMNIST(opt.dataroot, train=True,  transform=val_transform),
                EMNIST(opt.dataroot, train=False, transform=val_transform)]
        sets = [STL10Subset(dataset, *split) for dataset, split in zip(sets, splits)]

        opt.num_classes = len(splits[0][0])

    else:
        raise Exception('Unknown dataset')

    loaders = [torch.utils.data.DataLoader(dataset,
                                           batch_size=opt.batchSize,
                                           shuffle=True,
                                           num_workers=0) for dataset in sets]
    return loaders
