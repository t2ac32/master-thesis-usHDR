
import time, math, os, sys, random
import datetime
from optparse import OptionParser
import numpy as np
import threading
import scipy.stats as st

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, transforms


from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch, exist_program

from pushbullet import Pushbullet
# Setup date/ time
currentDT = datetime.datetime.now()

# FLAGS
use_PN = False

# Push notifications Setup
if use_PN:
    api_key = 'o.iiEaKoSMwHp5iFJkG4VGV4rIXE5YWKss'
    pb = Pushbullet(api_key)
    #  SEND PUSH NOTIFICATION PROGRAM STARTED
    start_msg = "Running started at: {}(".format(str(currentDT))
    push = pb.push_note("pycharm: Running", )




# === Settings =================================================================

sys.path.insert(0, "../")
wk_dir = os.path.curdir

eps = 1.0 / 255.0
sx = 224
sy = 224


# dataSets_dir = 'D:/TUM/Master Thesis/Images/DataSets/LDR2_fakeComp_DataSet/'
dataSets_dir = os.path.join(wk_dir, "LDR_DataSet")
data_dir = os.path.join(dataSets_dir)
data_dir_jpg = os.path.join(data_dir, "jpg")


# log_dir = os.path.join(output_dir, "logs")
# im_dir = os.path.join(output_dir, "im")


# =========HDR loss =============================================================
def Hdr_loss(input_y, true_x, logits, eps=eps, sep_loss=False,gpu=False):

    in_y = input_y.permute(0 ,2 ,3 ,1).to(dtype=torch.double)
    target = true_x.permute(0 ,2 ,3 ,1).to(dtype=torch.double)
    y      = logits.permute(0, 2, 3, 1).to(dtype=torch.double)
    # print('input y ' , in_y.dtype)
    # print('target', target.dtype)ju
    # print('y'     , y.dtype)
    '''
        Args:
                    true: a tensor of shape [B, 1, H, W].
                    logits: a tensor of shape [B, C, H, W]. Corresponds to
                            the raw output or logits of the model.
                    eps: added to the denominator for numerical stability.
        Remember: [batch_size, channels, height, width]
    '''

    # TODO input_ and target might already be torch tensors
    #sinput_         = input_.permute((0, 2, 3, 1))
    # For masked loss, only using information near saturated image regions
    thr       = 0.05 # Threshold for blending
    zeros     = torch.DoubleTensor([0.0]).to(dtype=torch.double)
    thrTensor = torch.DoubleTensor([0.05]).to(dtype=torch.double)
    oneT      = torch.DoubleTensor([1.0]).to(dtype=torch.double)
    if gpu:
        zeros     = zeros.cuda()
        thrTensor = thrTensor.cuda()
        oneT      = oneT.cuda()
    # print('input_ shape:', in_y.shape)
    msk       = torch.max(in_y ,3 ,keepdim=True)[0]
    #print('msk',msk.shape)
    th        = torch.max(zeros, msk - 1.0 + thr)
    #print('th', th)
    th        = torch.div(th ,thrTensor)
    #print('tj', th)
    msk       = torch.min(oneT, th ) 
    #print('msk', msk)
    msk       = msk.repeat(1,1,1, 3)
    
    y_log_    = torch.log(in_y + eps).to(dtype=torch.double)
    if y_log_.nelement() == 0: 
        print(' ................................. y_log_ is Empty')
        #y_log_    = y_log_.permute((0 ,2 ,3 ,1))
    x_log     = torch.log(torch.pow(target, 2.0 ) +eps)
    # print('y_log_: ' ,y_log_.shape)
    # print('x_log: ' ,x_log.shape)

    # Loss separated into illumination and reflectance terms
    if sep_loss:

        filterx = np.zeros((1 ,1 ,3 ,1))
        filterx[:, :, 0, 0] = 0.213
        filterx[:, :, 1, 0] = 0.715
        filterx[:, :, 2, 0] = 0.072
        filterx = torch.from_numpy(filterx)
        if gpu:filterx = filterx.cuda()

        conv = torch.nn.Conv2d(1 ,1 ,1 , bias = False)
        filterx = filterx.permute(3 ,2 ,0 ,1).cuda()
        # print('filter: ', filterx.shape)
        ## y_lum_lin_ 
        conv.weight = torch.nn.Parameter(filterx)

        ##this y must be y_ input of network
        z = in_y.permute(0,3,1,2)
        l = conv(z)
        #l = l.detach().numpy()
        y_lum_lin_ = l.permute(0,2,3,1)
        # print('input: ' ,in_y.shape)
        # print('z: ', z.shape)
        # print(l.shape)

        ## y_lum_lin 
        e = (torch.exp(in_y)) - eps
        z = e.permute(0,3,1,2); print('z: ', z.shape)
        l_ = conv(z)
        y_lum_lin = l_.permute(0,2,3,1)

        ##x_lum_lin
        z = target.permute(0,3,1,2); 
        l_x = conv(z)
        x_lum_lin = l_x.permute(0,2,3,1)

        # print("y_lum_lin_:", y_lum_lin_.shape)
        # print("y_lum_lin:", y_lum_lin.shape)
        # print("x_lum_lin:", x_lum_lin.shape)

        y_lum_ = torch.log(y_lum_lin_ + eps)
        y_lum  = torch.log(y_lum_lin  + eps)
        x_lum  = torch.log(x_lum_lin  + eps)
        # print('y_lum_: ' ,y_lum_.shape)
        # print('y_lum' ,y_lum.shape)
        # print('x_lum' ,x_lum.shape)

        # gaussian kernel
        nsig = 2
        filter_size = 13
        interval = ( 2 *nsig + 1. ) /(filter_size)
        ll = np.linspace(-nsig - interval /2., nsig + interval /2., filter_size +1)
        kern1d = np.diff(st.norm.cdf(ll))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw /kernel_raw.sum()

        # Illumination, approximated by means of Gaussian filtering
        weights_g = np.zeros((filter_size, filter_size, 1, 1))
        weights_g[:, :, 0, 0] = kernel
        weights_g = torch.from_numpy(weights_g)

        padding_same = 6

        # y_ill_ = torch.nn.Conv2d(y_lum_, weights_g, [1, 1, 1, 1], padding='SAME')
        ill_conv = torch.nn.Conv2d(1,1,1 , padding=6, bias = False)
        filterw = weights_g.permute(3,2,0,1)
        if gpu:filterw = filterw.cuda() 
        ill_conv.weight = torch.nn.Parameter(filterw)
        ill_conv_in_ = y_lum_.permute(0,3,1,2)
        ill_conv_in_ = ill_conv(ill_conv_in_)
        y_ill_ = ill_conv_in_.permute(0,2,3,1)

        #y_ill = torch.nn.Conv2d(y_lum, weights_g, [1, 1, 1, 1], padding='SAME')
        ill_conv = torch.nn.Conv2d(1,1,1 , padding=6, bias = False)
        filterw =  weights_g.permute(3,2,0,1)
        if gpu:filterw = filterw.cuda()
        ill_conv.weight = torch.nn.Parameter(filterw)
        ill_conv_in_ = y_lum.permute(0,3,1,2);
        ill_conv_in_ = ill_conv(ill_conv_in_)
        y_ill = ill_conv_in_.permute(0,2,3,1)

        # x_ill = torch.nn.Conv2d(x_lum, weights_g, [1, 1, 1, 1], padding='SAME')
        ill_conv = torch.nn.Conv2d(1,1,1 , padding=6, bias = False)
        filterw = weights_g.permute(3,2,0,1)
        if gpu: filterw = filterw.cuda()
        ill_conv.weight = torch.nn.Parameter(filterw)
        ill_conv_in_ = x_lum.permute(0,3,1,2)
        ill_conv_in_ = ill_conv(ill_conv_in_)
        x_ill = ill_conv_in_.permute(0,2,3,1)

        # print('y_ill_: ', y_ill_.shape)
        # print('y_ill: ', y_ill.shape)
        # print('x_ill: ', x_ill.shape)

        # PYtorch convention of tensor: NxCx(H*W)
        # Reflectance
        y_ill_ = y_ill_.repeat(1, 1, 1, 3)
        y_ill  = y_ill.repeat(1, 1, 1, 3)
        x_refl = x_ill.repeat(1, 1, 1, 3)

        y_refl_ = y_log_ - y_ill_  # tf.tile(y_ill_, [1,1,1,3])
        y_refl  = y - y_ill
        x_refl  = target - x_ill

        # print(y_refl_.shape)
        # print(y_refl.shape)
        # print(x_refl.shape)

        sub_yill    = torch.sub(y_ill, y_ill_)
        square_yill = torch.mul(sub_yill ,sub_yill)

        sub_refl    = torch.sub(y_refl ,y_refl_)
        square_refl = torch.mul(sub_refl ,sub_refl)

        sub_xyill    = torch.sub(x_ill, y_ill_)
        square_xyill = torch.mul(sub_xyill ,sub_xyill)

        sub_refl    = torch.sub(x_refl ,y_refl_)
        square_refl = torch.mul(sub_refl ,sub_refl)

        cost =              torch.mean((lambda_ir * square_yill + (1.0 - lambda_ir) * square_refl) * msk)
        cost_input_output = torch.mean((lambda_ir * square_xyill + (1.0 - lambda_ir) * square_refl) * msk)
        #print('cost: ', cost)
        print('cost_input_output', cost_input_output)
    else:
        #print('>>>>> y:', len(y.size()))
        #print('>>>>> y_log_:', len(y_log_.size()))
        
        
        out_sub_log = torch.sub(y, y_log_) * msk
        if msk.nelement() == 0: 
            print(' ................................. msk is Empty')
        sqr_of_log  = torch.mul(out_sub_log, out_sub_log)

        cost        = torch.mean(sqr_of_log)
        # print('>>>>> out_sub_log:', out_sub_log)
        # print('>>>>> sqr_of_log:', sqr_of_log)
        
        print('cost: ', cost)

        # TODO test
        trgt_sub_log = torch.sub(y_log_, x_log) * msk
        trgt_square  = torch.mul(trgt_sub_log, trgt_sub_log)
        cost_input_output = torch.mean(trgt_square)
        print('cost_input_output', cost_input_output)
    return cost, cost_input_output
# =====Learning parameters ======================================================
num_epochs = 100
start_step = 0.0
Learning_rate = 0.00005
batch_size = 15
sep_loss = True
lambda_ir = 0.5
tran_size = 0.99
buffer_size = 256
rand_data = True
print('setup finished')


def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.20,
              save_cp=False,
              gpu=False,
              img_scale=0.5):
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    train_summary = SummaryWriter()
    valid_summary = SummaryWriter()
    # === Localize training data ===================================================
    print("Getting images directory")
    dir_img = os.path.join(data_dir, 'Org_images/')
    dir_compressions = os.path.join(data_dir, 'c_images/')
    dir_mask = os.path.join(data_dir, 'c_images/')
    dir_checkpoint = 'checkpoints/'

    ids = get_ids(dir_img)
    ids = split_ids(ids)
    # print(sum(1 for i in ids))

    # === Load Training/Validation data =====================================================
    iddataset = split_train_val(ids, val_percent)
    print(iddataset['train'])
    # print(iddataset['val'])
    print('''
        Training SETUP:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
        '''.format(epochs, batch_size, lr, len(iddataset['train']),
                   len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])

    # optimizer = optim.SGD(net.parameters(),
    #    lr=lr,
    #    momentum=0.9,
    #    weight_decay=0.0005)

    optimizer = optim.Adagrad(net.parameters(),
                              lr=lr,
                              weight_decay=0.0005)
    # Binary cross entropy
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], dir_compressions, dir_mask, img_scale)
        val = get_imgs_and_masks(iddataset['val'], dir_compressions, dir_mask, img_scale)

        epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            
            true_masks = np.array([i[1] for i in b])
            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)
            
            # print('imgs type: ', type(imgs))
            
            #print('>>>>>>> Batch SHAPE: ' , imgs.shape)
            #print('>>>>>>> True masks: ', true_masks.shape)
            #print('>>>>>>> Img size', imgs.size())
            
            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            # grid = torchvision.utils.make_grid(imgs)
            writer.add_image('images', imgs[0], 0, dataformats='CHW')
            # writer.add_graph(net)
            writer.close()

            # Predicted mask images
            masks_pred = net(imgs)
            #print('>>>>>> masks_pred: ', masks_pred.shape)
            #masks_probs_flat = masks_pred.view(-1)
            # Ground Truth images of masks
            #true_masks_flat = true_masks.view(-1)
            #print('--------------masks_pred', masks_pred.shape)
            #print('--------------masks_pred_flat', masks_probs_flat.shape)
            

            #loss = criterion(masks_probs_flat, true_masks_flat)
            cost, cost_input_output = Hdr_loss(imgs, true_masks, masks_pred,sep_loss=False,gpu=gpu)
            #loss is torch tensor
            epoch_loss += cost.item()

            print('cost:', cost.item, 'cost_input_output:', cost_input_output.item)
            print('Current Epoch loss:', cost.item())
            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, cost.item()))

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        if 1:
            val_loss = eval_hdr_net(net, val, gpu)
            print('Validation loss: {:05.4f}'.format(val_loss))
            valid_summary.add_scalar('Loss/validation_loss',val_loss, epoch)
        
        # Training and validation loss for Tensorboard
        
        #file_writer.add_summary(valid_summary, step)
        train_summary.add_scalar('Loss/training_loss',epoch_loss,epoch)
        #file_writer.add_summary(train_summary, step)

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))

        if use_PN:
            end_msg = "train.py finisheed at: {}(".format(str(currentDT))
            push = pb.push_note("pycharm: Finish", end_msg)

def eval_hdr_net(net, dataset, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot_loss = 0
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[1]

        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)
        #print('val img shape:', img.shape)
        #print('val true shape:', true_mask.shape)
        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img)[0]
        mask_pred = mask_pred.expand(1,-1,-1,-1)
        #mask_pred = (mask_pred > 0.5).float()
        #print('val pred shape:', mask_pred.shape)

        #tot += dice_coeff(mask_pred, true_mask).item()
        cost, cost_input_output = Hdr_loss(img, true_mask, mask_pred,sep_loss=False,gpu=gpu)
        tot_loss += cost.item()
        return tot_loss

def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=15,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=3, n_classes=3)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        print("Trying... train")
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')

        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)



