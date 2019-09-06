#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks
#TODO
'''
  check https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class 
  To learn how to create a pytorch dataset format
  and be able to use torchvision.transforms
'''

import os

import numpy as np
import time 
import torch
from torch.utils.data import Dataset, DataLoader
from   tqdm import tqdm
from   PIL  import Image
import cv2



from .utils import resize_and_crop, get_square, normalize, hwc_to_chw, only_resize, only_resizeCV

formatter = "{:02d}".format

def get_ids(dir):
    """Returns a list of the ids in the directory"""
    """ Remove value x to get full list of ids, now just gets 1901 minus x where x [x:] """
    return (f[:-4] for f in os.listdir(dir)[1801:]) #

def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id)  for id in ids for i in range(n))


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        yield get_square(im, pos)

def get_hdr_label(ids, dir, suffix):
    """From a list of ids , return the set of ldr images"""   
    for id in ids:
        img_name = dir + id + '/Results/' + suffix
        img = cv2.imread(img_name, -1) #flags = cv2.IMREAD_ANYDEPTH
        img = only_resizeCV(img,w=224,h=224)
        
        # chann 1
        c1_max = img[..., 0].min()
        print('max c1: {0:}'.format(c1_max))
        img = np.asarray(img)
        for i in range(0, 15):
            # imageIO
            #img = np.expand_dims(img,0)
            yield img
        
def get_ldr_set(ids, dir, suffix): 
    counter     = 0
    smlst_h     = 2000
    smlst_w     = 2000
    max_h       = 0 
    max_w       = 0
    need_upsamp = 0
    for id in ids:
        measured = 1
        ldr_set    = []
        for i in range(0, 15):
            img_name = dir + id + '/exVivo_' + str(formatter(i)) + suffix 
            img = Image.open(img_name)
            img = only_resize(img,224,224)
            # print('//////////ldr image shape: ' , img_array.shape)
            #ldr_set.append(img_array)
            '''
            if measured == 0:
                if img.size[0] < 236 or img.size[1] < 236:
                    need_upsamp += 1
                    print(img.size)
                    measured = 1
            '''
            ldr_set.append(img)
        counter += 1
        yield ldr_set

def get_ldr(ids, dir, suffix): 
    for id in ids:
        # print('Getting set: ',id)
        for i in range(0, 15):
            img_name = dir + id + '/exVivo_' + str(formatter(i)) + suffix
            img = Image.open(img_name)   
            img = only_resize(img,(224,224))
            ''' Array must be:
                (batch_size, height, width, channels)
             '''
            yield img

def switch_and_normalize(imgs):
    for subset in imgs:
        imgs_switched   = map(hwc_to_chw, subset)
        imgs_normalized = map(normalize, imgs_switched)

        yield imgs_normalized

def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""
    #imgs = to_cropped_imgs(ids, dir_img, '.jpeg', scale)
    imgs = get_ldr(ids, dir_img, '.png')
    # need to transform from HWC to CHW if more than 1 channel
    imgs_switched   = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    #=====When using subsets of ldr images===========
    #imgs_normalized = switch_and_normalize(imgs)
    hdr_suffix = 'stack_hdr_image.hdr' #'hdrReinhard_local.png'
    masks = get_hdr_label(ids, dir_mask, hdr_suffix)
    masks = map(hwc_to_chw, masks)
    return zip(imgs_normalized, masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)

class HdrDataset(Dataset):
    "HDR fake compression dataset."

    def __init__(self,ids, dir_img,dir_mask, transform=None):
        """
        Args:
            ids (type): Description
            dir_img (string): 
            dir_mask (string): 
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ids = ids
        self.dir_img = dir_img
        self.dir_mask = dir_mask
        self.transform = transform

    def __len__(self):
        return len(self.ids)*15

    def __getitem__(self, idx):

        #modify to return one label 
        images, targets = get_imgs_and_masks(ids, dir_img, dir_mask, scale)
        for i in range(15):
            tensor_x = torch.Tensor(images[i])
            tensor_y = torch.Tensor(target)   
            sample = {'input': tensor_x, 'target': tensor_y}

        if self.transform:
            sample = self.transform(sample)

        return sample