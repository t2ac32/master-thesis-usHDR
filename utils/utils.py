import os
import sys
import random
import numpy as np
import PIL.Image
import cv2
import torch
from torchvision.utils import save_image

def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]

def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def resize_and_crop(pilimg, scale=0.0, final_height=None):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return np.array(img, dtype=np.float32)

def only_resize(pilimg, final_height= (224,224)):
    # print(pilimg.filename)
    # print('----------arr original shape:', arr.shape)
    w = pilimg.size[0]
    h= pilimg.size[1]
    img = pilimg.resize(final_height)
    img = np.asarray(img, dtype=np.float32)
    # print('-----------ONly resize: Label resized shape:', img.shape)
    return img

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []
        
    if len(b) > 0:
        yield b

def split_train_val(dataset, expositions=15,val_percent=0.20):
    #print('splitting into train an validation sets:')
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    t = dataset[:-n]
    v =  dataset[-n:]
    train = []
    val = []

    for im_id in t:
        for e in range(expositions):
            train.append(im_id)
    for im_id in v:
        for e in range(expositions):
            val.append(im_id)
    
    return {'train': train , 'val':val}


def normalize(x):
    return x / 255

# CV2 utilities
def only_resizeCV(cvImage, w=224,h=224):
    img = cv2.resize(cvImage,(w,h))
    return img 
def cv2torch(np_img):
    rgb = np_img[:, :, (2, 1, 0)]
    return torch.from_numpy(rgb.swapaxes(1, 2).swapaxes(0, 1))

def map_range(x, low=0, high=1):
    return np.interp(x, [x.min(), x.max()], [low, high]).astype(x.dtype)

def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new

def check_dir (dir_path ):
    dir_exists = False
    if os.path.isdir(dir_path):
        dir_exists = True
    else:
        try:
            os.mkdir(dir_path)
        except OSError:
            #print ("Creation of the directory %s failed" % dir_path)
            dir_exists = False
        else:
            dir_exists = True
            #print ("Successfully created the directory %s " % dir_path)
    return dir_exists 

def saveTocheckpoint(folder,experiment_name,img_id,epoch,input_img,grnd_img,pred_img):
    dir_path = os.path.join(folder,experiment_name)
    #print('dir_path', dir_path)
   
    if check_dir(dir_path): 
        file_name  = dir_path + '/epoch{0:02d}_{1:}'.format(epoch+1,img_id) + '.png'  
        e_imgs = [input_img, grnd_img, pred_img]
        save_image(e_imgs, file_name,nrow=3)

# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs

def exist_program():
    print('existing program')
    sys.exit()