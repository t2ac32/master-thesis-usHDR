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
from   tqdm import tqdm
from   PIL  import Image
import cv2

import torch
from torch.utils.data import Dataset, DataLoader



try:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    tb = True
    print('Using Tensorboard in load.py')
except ImportError:
    print('Counld not Import Tensorboard')
    try: 
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()
        tb = True
        print('Using Tensorboard X')
    except ImportError:
        print('Could not import Tensorboard X')
        tb = False

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw, only_resize, only_resizeCV, map_range, cv2torch

formatter = "{:02d}".format
tensorboard= tb
def get_ids(dir):
    """Returns a list of the ids in the directory"""
    """ Remove value x to get full list of ids, now just gets 1901 minus x where x [x:] """                                                                                             
    return (f[:-4]  for f in os.listdir(dir)) #[1801:]

def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id)  for id in ids for i in range(n))


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        yield get_square(im, pos)

def get_hdr_label(id, dir, suffix):
    """From a list of ids , return the set of ldr images"""   
    img_name = dir + id + '/Results/' + suffix
    
    # When using .hdr files
    img = cv2.imread(img_name, -1) #flags = cv2.IMREAD_ANYDEPTH
    img = only_resizeCV(img,w=224,h=224)  
    img = np.asarray(img, dtype=np.float32)
    #img = map_range(img)
    torch_hdr = cv2torch(img)
    '''
    # WHEN USING TONEMAP.PNG (do not change axis this are in CHW form already)
    img = Image.open(img_name)   
    img = only_resize(img,(224,224))
    img = np.expand_dims(img, axis=0)
    '''
    #print('HDR max:', np.amax(img))
    #print('HDR min:', np.amin(img))
    #print('HDR shape:',torch_hdr.shape)
    return torch_hdr

def get_ldr(id, dir, suffix,exposition_num): 
    # print('Getting set: ',id)
    i = exposition_num
    img_name = dir + id + '/exVivo_' + str(formatter(i)) + suffix
    img = Image.open(img_name)   
    img = only_resize(img,(224,224))
    #print('ldr shape; ', img.shape)
    #print('ldr max:', np.amax(img))
    
    ''' Array must be:
        (batch_size, height, width, channels)
    '''
    yield img

def switch_and_normalize(imgs):
    for subset in imgs:
        imgs_switched   = map(hwc_to_chw, subset)
        imgs_normalized = map(normalize, imgs_switched)

        yield imgs_normalized

def get_imgs_and_masks(id, dir_img, dir_mask, exposition_num):
    """Return all the couples (img, mask)"""
    #imgs = to_cropped_imgs(ids, dir_img, '.jpeg', scale)
    img = get_ldr(id, dir_img, '.png',exposition_num)
    # need to transform from HWC to CHW if more than 1 channel
    imgs_switched   = map(hwc_to_chw, img)
    imgs_normalized = map(normalize, imgs_switched)
    imgs_normalized = np.array([l for l in imgs_normalized]).astype(np.float32)

    #=====When using subsets of ldr images===========
    #imgs_normalized = switch_and_normalize(imgs)
    hdr_suffix =  'stack_hdr_image.hdr' #'hdrReinhard_local.png'
    mask = get_hdr_label(id, dir_mask, hdr_suffix)
    return imgs_normalized, mask


class HdrDataset(Dataset):
    "HDR fake compression dataset."

    def __init__(self,ids, dir_img, dir_mask,expositions,
                 transform=None):
        """
        Args:
            ids (type): array of id that conform the images in dataset
            dir_img (string): directory of id where inputs will be
            dir_mask (string): directory where labels are found 
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ids = ids
        self.dir_img = dir_img
        self.dir_mask = dir_mask
        self.expositions = expositions
        self.transform = transform
       
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
 
        img_id = self.ids[idx]
        
        #print('Getting id:',idx,'for image:', img_id)
        exposure = idx%(self.expositions)
        image, target = get_imgs_and_masks(img_id, self.dir_img,
                                            self.dir_mask,
                                            exposure)
        
        
        tensor_x = torch.Tensor(image)
        tensor_x = tensor_x.squeeze()
        tensor_y = torch.Tensor(target)   
        sample = {'input': tensor_x, 'target': tensor_y,'id': img_id}
        #print(idx, sample['input'].size(), sample['target'].size())

        if self.transform:
            sample = self.transform(sample)
        
        return sample  