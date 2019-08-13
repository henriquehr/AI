import random
import glob
import torch
from skimage import io, transform, color
import numpy as np
import math
from torch.utils.data import Dataset


class Rescale(object):
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size

    def __call__(self, data):
        image = data[0]
        label = data[1]

        h, w = image.shape[:2]

        if isinstance(self.output_size,int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (self.output_size, self.output_size), mode='constant', preserve_range=True)
        label = transform.resize(label, (self.output_size, self.output_size), mode='constant', order=0, preserve_range=True)

        return image, label


class CenterCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        image = data[0]
        label = data[1]
        
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        assert((h >= new_h) and (w >= new_w))

        h_offset = int(math.floor((h - new_h)/2))
        w_offset = int(math.floor((w - new_w)/2))

        image = image[h_offset: h_offset + new_h, w_offset: w_offset + new_w]
        label = label[h_offset: h_offset + new_h, w_offset: w_offset + new_w]

        return image, label


class RandomCrop(object):
    def __init__(self,output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        image = data[0]
        label = data[1]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]

        return image, label


class ToTensor(object):    
    def normalize(self, x):
        min_val = np.min(x)
        max_val = np.max(x)
        return (x - min_val) / (max_val - min_val)
        
    def __call__(self, data):
        image = data[0]
        label = data[1]

        image = self.normalize(image)
        label = self.normalize(label)

        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        # if image.shape[2] == 1:
        #     tmpImg[:,:,0] = image[:,:,0]
        #     tmpImg[:,:,1] = image[:,:,0]
        #     tmpImg[:,:,2] = image[:,:,0]
        # else:
        #     tmpImg[:,:,0] = image[:,:,0]
        #     tmpImg[:,:,1] = image[:,:,1]
        #     tmpImg[:,:,2] = image[:,:,2]
        if image.shape[2] == 1:
            tmpImg[:,:,0] = (image[:,:,0] - 0.485) / 0.229
            tmpImg[:,:,1] = (image[:,:,0] - 0.485) / 0.229
            tmpImg[:,:,2] = (image[:,:,0] - 0.485) / 0.229
        else:
            tmpImg[:,:,0] = (image[:,:,0] - 0.485) / 0.229
            tmpImg[:,:,1] = (image[:,:,1] - 0.456) / 0.224
            tmpImg[:,:,2] = (image[:,:,2] - 0.406) / 0.225

        image = tmpImg.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))

        return torch.from_numpy(image), torch.from_numpy(label)


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        image = data[0]
        label = data[1]

        if random.random() < self.prob:
            image = image[:,::-1,:]
            label = label[:,::-1,:]

        return image, label


class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        image = data[0]
        label = data[1]

        if random.random() < self.prob:
            image = image[::-1,:,:]
            label = label[::-1,:,:]

        return image, label


class DatasetLoader(Dataset):
    def __init__(self, img_name_list, lbl_name_list, transform=None):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,idx):
        image = io.imread(self.image_name_list[idx])

        if(len(self.label_name_list) == 0):
            _label = np.zeros(image.shape)
        else:
            _label = io.imread(self.label_name_list[idx])

        label = np.zeros(_label.shape[0:2])
        if(len(_label.shape) ==  3):
            label = _label[:,:,0]
        elif(len(_label.shape) == 2):
            label = _label

        if(len(image.shape) == 3 and len(label.shape) == 2):
            label = label[:,:,np.newaxis]
        elif(len(image.shape) == 2 and len(label.shape) == 2):
            image = image[:,:,np.newaxis]
            label = label[:,:,np.newaxis]

        if self.transform:
            image, label = self.transform([image, label])

        return image.type(torch.FloatTensor), label.type(torch.FloatTensor)
