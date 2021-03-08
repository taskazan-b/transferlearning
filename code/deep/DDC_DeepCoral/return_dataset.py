import os
import torch
from torchvision import transforms
import numpy as np
import os.path
from PIL import Image
import data_loader
import fileinput
from glob import glob


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

def return_SSDA(args):
    base_path = args.data + 'txt/%s' % args.dataset
    root = args.data + '%s/' % args.dataset
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.src + '.txt')
    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.tar + '_%d.txt' % (args.num))
    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.tar + '_3.txt')
    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.tar + '_%d.txt' % (args.num))

    if args.model == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    source_dataset = Imagelists_VISDA([image_set_file_s], root=root,
                                      transform=data_transforms['train'])
    source_dataset_train = Imagelists_VISDA_TrainSrc([image_set_file_s], root=root,
                                      transform=data_transforms['train'], bs = args.batchsize)

    target_dataset = Imagelists_VISDA([image_set_file_t,image_set_file_t_val,image_set_file_unl], root=root,
                                      transform=data_transforms['train'])
    target_dataset_train = Imagelists_VISDA([image_set_file_t,image_set_file_t_val], root=root,
                                          transform=data_transforms['train'])
    target_dataset_test = Imagelists_VISDA([image_set_file_unl], root=root,
                                           transform=data_transforms['test'])
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))

    bs = args.batchsize


    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=4, shuffle=True,
                                                drop_last=True)
    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=4,
                                    shuffle=True, drop_last=True)
    target_train_loader = \
        torch.utils.data.DataLoader(target_dataset_train,
                                    batch_size=args.num+3,
                                    num_workers=4,
                                    shuffle=False, drop_last=False)
    source_train_loader = \
        torch.utils.data.DataLoader(source_dataset_train,
                                    batch_size=1,
                                    num_workers=4,
                                    shuffle=False, drop_last=False)
    target_test_loader = \
        torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=bs, num_workers=4,
                                    shuffle=False, drop_last=False)
    return source_loader, target_loader, target_test_loader, target_train_loader, source_train_loader, class_list




def return_dataset_test(args):
    base_path = args.data + 'txt/%s' % args.dataset
    root = args.data + '%s/' % args.dataset
    image_set_file_s = os.path.join(base_path, args.src + '_all' + '.txt')
    image_set_file_test = os.path.join(base_path,
                                       'unlabeled_target_images_' +
                                       args.tar + '_%d.txt' % (args.num))
    if args.model == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    target_dataset_unl = Imagelists_VISDA(image_set_file_test, root=root,
                                          transform=data_transforms['test'],
                                          test=True)
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.model == 'alexnet':
        bs = 32
    else:
        bs = 32
        # bs = 24
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=3,
                                    shuffle=False, drop_last=False)
    return target_loader_unl, class_list


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset_fromlist(image_list):
    image_index = []
    label_list = []
    
    for txtfile in image_list:
        with open(txtfile) as f:
            for line in f.readlines():
                img, label = line.split(' ')
                image_index.append(img.split(' ')[0])
                label_list.append(int(label.strip()))
    image_index = np.array(image_index)
    label_list = np.array(label_list)
    sortidx = np.argsort(label_list)
    image_index = image_index[sortidx]
    label_list = label_list[sortidx]
    return image_index, label_list


def return_classlist(image_list):
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:
                label_list.append(str(label))
    return label_list


class Imagelists_VISDA(object):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, target_transform=None, test=False):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            return img, target
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)



class Imagelists_VISDA_TrainSrc(object):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, test=False, bs = 32):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.loader = pil_loader
        self.root = root
        self.test = test
        self.bs = bs
        batch_indices = [0]
        l0=labels[0]
        for i,l in enumerate(labels):
            if l != l0:
                batch_indices.append(i)
                l0 = l
        batch_indices.append(i)
        self.batch_indices = batch_indices

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        start_idx = self.batch_indices[index]
        end_idx = self.batch_indices[index+1]
        if self.batch_indices[index+1]-self.batch_indices[index]>self.bs:
            import random
            selected = random.sample(range(start_idx,end_idx),self.bs)
        else:
            selected = list(range(start_idx,end_idx))

        target = self.labels[selected]
        path = os.path.join(self.root, self.imgs[selected[0]])
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
       
        for i in selected[1:]:
            path = os.path.join(self.root, self.imgs[i])
            im = self.loader(path)
            if self.transform is not None:
                im = self.transform(im)
            img = np.concatenate((img,im),axis=0)

        return img, target
       

    def __len__(self):
        return len(self.batch_indices) - 1