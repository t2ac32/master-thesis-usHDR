
import time, math, os, sys, random
import datetime
from optparse import OptionParser
import numpy as np
import threading
import scipy.stats as st

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path

from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch, exist_program, HdrDataset

from pushbullet import Pushbullet

try:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
except ImportError:
    print('Counld not Import module Tensorboard')
    try: 
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()
    except ImportError:
        print('Could not import TensorboardX')
# Setup date/ time
currentDT = datetime.datetime.now()

# FLAGS




# === Settings =================================================================

sys.path.insert(0, "../")
wk_dir = os.path.curdir

eps = 1.0 / 255.0
sx = 224
sy = 224

# dataSets_dir = 'D:/TUM/Master Thesis/Images/DataSets/LDR2_fakeComp_DataSet/'
# log_dir = os.path.join(output_dir, "logs")
# im_dir = os.path.join(output_dir, "im")


# =========HDR loss =============================================================
def Hdr_loss(input_y, true_x, logits, eps=eps, sep_loss=False, gpu=False, tb=False):

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

    # REVISAR MASCARA
    msk       = torch.max(in_y ,3 ,keepdim=True)[0]
    th        = torch.max(zeros, msk - 1.0 + thr)
    th        = torch.div(th ,thrTensor)
    msk       = torch.min(oneT, th ) 
    msk       = msk.repeat(1,1,1, 3)
    if tb:
        writer.add_images('loss_mask',msk,0, dataformats='NHWC')
            
    
    y_log_    = torch.log(in_y + eps).to(dtype=torch.double)
    x_log     = torch.log(torch.pow(target, 2.0 ) + eps)
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
        filterx = filterx.permute(3 ,2 ,0 ,1).cuda() #permute(0,3,1,2)
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
        e = (torch.exp(y)) - eps
        z = e.permute(0,3,1,2); #print('z: ', z.shape)
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
        #print('cost_input_output', cost_input_output)
    else:
        #print('>>>>> y:', len(y.size()))
        #print('>>>>> y_log_:', len(y_log_.size()))

        out_sub_log = torch.mul(torch.sub(y, y_log_), msk)
        #out_sub_log = torch.sub(y, x_log) * msk
        sqr_of_log  = torch.mul(out_sub_log, out_sub_log)
        cost        = torch.mean(sqr_of_log)
        # print('>>>>> out_sub_log:', out_sub_log)
        # print('>>>>> sqr_of_log:', sqr_of_log)
        
        #print('cost: ', cost)
        # TODO test
        trgt_sub_log = torch.mul(torch.sub(y_log_, x_log), msk)
        trgt_square  = torch.mul(trgt_sub_log, trgt_sub_log)
        cost_input_output = torch.mean(trgt_square)
        #print('cost_input_output', cost_input_output)

    return cost, cost_input_output
# =====Learning parameters ======================================================
#num_epochs = 100
#start_step = 0.0
#Learning_rate = 0.00005
#batch_size = args.batchsize
lambda_ir = 0.5
#tran_size = 0.99
#buffer_size = 256
#rand_data = True
print('setup finished')


def train_net(net, epochs=5, batch_size=1, lr=0.1, val_percent=0.20,
              save_cp=False,
              gpu=False,
              img_scale=0.5,
              expositions_num=15,
              tb=False,
              use_notifications=False,
              polyaxon=False):
    
    # Writer will output to ./runs/ directory by default
        #train_summary = SummaryWriter()
        #valid_summary = SummaryWriter()
    # === Localize training data ===================================================
    if polyaxon:
        data_paths = get_data_paths()
        outputs_path = get_outputs_path()
        dataset = "LDR_DataSet"
        #dataSets_dir = os.path.join(wk_dir,"data/", "LDR_DataSet")
        dataSets_dir = training_data_path = data_paths['data1']  + dataset
    else:
        dataSets_dir = os.path.join(wk_dir, "LDR_DataSet")
    print(dataSets_dir)
    dir_img = os.path.join(dataSets_dir, 'Org_images/')
    dir_compressions = os.path.join(dataSets_dir, 'c_images/')
    dir_mask = os.path.join(dataSets_dir, 'c_images/')
    dir_checkpoint = 'checkpoints/'

    print("Getting images directory")
    ids = get_ids(dir_img)
    #print(sum(1 for i in ids))
    # === Load Training/Validation data =====================================================
    iddataset = split_train_val(ids, expositions_num, val_percent )
    #print(iddataset['train']) 
    #print(iddataset['val'])
    N_train = len(iddataset['train'])
    N_val = len(iddataset['val'])
    # optimizer = optim.SGD(net.parameters(),
    #    lr=lr,
    #    momentum=0.9,
    #    weight_decay=0.0005)

    optimizer = optim.Adagrad(net.parameters(),
                              lr=lr,
                              weight_decay=0.0005)
    # Binary cross entropy
    criterion = nn.MSELoss()
    since = time.time()
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
    
    train_dataset = HdrDataset(iddataset['train'], dir_compressions, dir_mask,
                               expositions_num)
    train_data_loader = DataLoader(train_dataset,batch_size=batch_size,
                             num_workers=1)
    
    
    for epoch in range(epochs):
        print('-' * 50)
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        begin_of_epoch = time.time()
        net.train()
            # reset the generators
            #train = get_imgs_and_masks(iddataset['train'], dir_compressions, dir_mask, img_scale, expositions_num)
            #val   = get_imgs_and_masks(iddataset['val'], dir_compressions, dir_mask, img_scale,expositions_num)


        epoch_loss = 0
        train_loss = 0
        step = 0
        save_step_sample = True
        for i, b in enumerate(train_data_loader):
            step += 1
            imgs, true_masks = b['input'], b['target']
            #print(i, b['input'].size(), b['target'].size())
            #print(len(b))
            #print('imgs type: ', type(imgs))
            #print('>>>>>>> Input max: ' , torch.max(imgs[0]))
            #print('>>>>>>> mask max : ', torch.max(true_masks[0]))
            #print('>>>>>>> Img size', imgs.size())
            
            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()
            
            # Predicted mask images
            optimizer.zero_grad()
            masks_pred = net(imgs)
                #print('>>>>>>>> Pred max: ', torch.max(masks_pred[0]))
                #masks_probs = F.sigmoid(masks_pred)
                #masks_probs_flat = masks_probs.view(-1)
                #true_masks_flat = true_masks.view(-1)
                
                #masks_probs_flat = masks_pred.view(-1)
                # Ground Truth images of masks
                #true_masks_flat = true_masks.view(-1)
                #print('--------------masks_pred', masks_pred.shape)
                #print('--------------masks_pred_flat', masks_probs_flat.shape)
                #loss = criterion(masks_probs_flat, true_masks_flat)
            
            cost, cost_input_output = Hdr_loss(imgs, true_masks, masks_pred, sep_loss=False, gpu=gpu, tb=tb)
            #cost = criterion(masks_pred,true_masks)
            #print('cost:', cost, 'cost_input_output:', cost_input_output)
            #loss is torch tensor
            train_loss += cost.item() * imgs.size(0)
            cost.backward()
            optimizer.step()
            epoch_loss = train_loss / len(train_dataset)
            
            if tb:
                writer.add_scalar('Loss/training_loss',epoch_loss,epoch )
                
                if save_step_sample:
                    print('saving train step {0:} sample : input,target & pred'.format(step))
                    train_sample =  [imgs[0],true_masks[0], masks_pred[0]]   
                    grid = torchvision.utils.make_grid(train_sample,nrow=3)
                    writer.add_image('train_sample', grid, 0)
                
            if step % 30 == 0:
                print('Step: {0:}, cost:{1:}, Epoch Loss:{2:.9f}, train_loss:{3:.6f}'.format(step,cost,epoch_loss,train_loss))
                
                
                
            save_step_sample = False

        print('-' * 15)
        print('Epoch finished !')
        print('-' * 15)
        print('Epoch Loss:{:.6f}'.format(epoch_loss))
        time_epoch = time.time() - begin_of_epoch 
        print('Epoch ETC: {:.0f}m {:.0f}s'.format(time_epoch // 60, time_epoch % 60))   
        if 1:
            val_dataset = HdrDataset(iddataset['val'], dir_compressions, 
                                     dir_mask,
                                     expositions_num)

            val_loss = eval_hdr_net(net, val_dataset, criterion, gpu,
                                                   batch_size,
                                                   expositions_num=15,
                                                   tb=tb)
            epoch_val_loss = val_loss / len(val_dataset)
            print('Validation loss: {0:}'.format(epoch_val_loss))
            if tb:
                writer.add_scalar('Loss/validation_loss',epoch_val_loss, epoch)
                
        # Training and validation loss for Tensorboard
        #file_writer.add_summary(valid_summary, step)
        #file_writer.add_summary(train_summary, step)

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))

    time_elapsed = time.time() - since   
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))     
    if tb:
        writer.close()
    if use_notifications:
        end_msg = "train.py finisheed at: {}(".format(str(currentDT))
        push = pb.push_note("pycharm: Finish", end_msg)
    

def eval_hdr_net(net, dataset,criterion, gpu=False,
              batch_size=1,
              expositions_num=15, tb=False):
    """Evaluation without the densecrf with the dice coefficient"""
    val_data_loader = DataLoader(dataset,batch_size=batch_size,
                             num_workers=1)
    net.eval()
    tot_loss = 0
    step = 0
    for i, b in enumerate(val_data_loader):
        step += 1
        imgs, true_masks = b['input'], b['target']
            
        #img = b[0]
        #true_mask = b[1]
        #img = torch.from_numpy(img).unsqueeze(0)
        #true_mask = torch.from_numpy(true_mask).unsqueeze(0)
        #print('val img shape:', img.shape)
        #print('val true shape:', true_mask.shape)
        if gpu:
            imgs = imgs.cuda()
            true_masks = true_masks.cuda()

        pred = net(imgs)
        #mask_pred = mask_pred.expand(1,-1,-1,-1)
        #mask_pred = (mask_pred > 0.5).float()
            
        cost, cost_input_output = Hdr_loss(imgs, true_masks, pred,sep_loss=False,gpu=gpu, tb=tb)
        #cost = criterion(pred,true_masks)
        tot_loss += cost.item()*imgs.size(0)

        return tot_loss 

def get_args():
    parser = OptionParser()
    parser.add_option('-p', '--polyaxon', action='store_true', dest='polyaxon',
                      default=False, help='set data dirs to use polyaxon')
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=15,
                      type='int', help='batch size')
    parser.add_option('-x', '--expo-num', dest='expositions', default=15,
                      type='int', help='number of exposition that compund an HDR.')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')
    parser.add_option('-n', '--notifications', action='store_true', dest='pushbullet',
                      default=False, help='use pushbullet notifications')
    parser.add_option('-t', '--tensorboard', action='store_true', dest='tensorboard',
                      default=False, help='use tensorboard logging')
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

    if args.pushbullet:
        api_key = 'o.iiEaKoSMwHp5iFJkG4VGV4rIXE5YWKss'
        pb = Pushbullet(api_key)
        #  SEND PUSH NOTIFICATION PROGRAM STARTED
        start_msg = "Running started at: {}(".format(str(currentDT))
        push = pb.push_note("usHDR: Running", start_msg )
    
    try:
        print("Trying... train")
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale,
                  expositions_num= args.expositions,
                  tb=args.tensorboard,
                  use_notifications=args.pushbullet,
                  polyaxon=args.polyaxon)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')

        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)



