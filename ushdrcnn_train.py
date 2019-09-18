
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

from pushbullet import Pushbullet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch, exist_program, HdrDataset, saveTocheckpoint

try:
    print('LOading TEnsorboard')
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    print('Using Tensorboard in train.py')
except ImportError:
    print('Counld not Import module Tensorboard')
    try: 
        print('loading tensorboard X')
        from tensorboardX import SummaryWriter
        try:
            outputs_path = get_outputs_path()
            writer = SummaryWriter(outputs_path)
            experiment = Experiment()
            print('Using Tensorboard X')
        except ImportError:
            writer = SummaryWriter()
            print('Using Tensorboard X')   
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

# =========HDR EpxandNet loss =============================================================

class ExpandNetLoss(nn.Module):
    def __init__(self, loss_lambda=5):
        super(ExpandNetLoss, self).__init__()
        self.similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-20)
        self.l1_loss = nn.L1Loss()
        self.loss_lambda = loss_lambda

    def forward(self, x, y):
        cosine_term = (1 - self.similarity(x, y)).mean()
        return self.l1_loss(x, y) + self.loss_lambda * cosine_term

# =========HDR loss =============================================================
def Hdr_loss(input_y, true_x, logits, eps=eps, sep_loss=False, gpu=False, tb=False):

    in_y = input_y.permute(0 ,2 ,3 ,1).to(dtype=torch.double)
    target = true_x.permute(0 ,2 ,3 ,1).to(dtype=torch.double)
    y      = logits.permute(0, 2, 3, 1).to(dtype=torch.double)
    #print('input y ' , in_y.shape)
    #print('target', target.shape)
    #print('y (logits)'     , y.shape)
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
    thr       = 0.1 # Threshold for blending
    zero      = torch.DoubleTensor([0.0]).to(dtype=torch.double)
    thrTensor = torch.DoubleTensor([0.05]).to(dtype=torch.double)
    oneT      = torch.DoubleTensor([1.0]).to(dtype=torch.double)
    if gpu:
        zero      = zero.cuda()
        thrTensor = thrTensor.cuda()
        oneT      = oneT.cuda()

    # REVISAR MASCARA
    msk   = torch.max(in_y ,3)[0]
    add   = torch.add(torch.sub(msk, oneT), thr)
    div   = torch.div(add, thr)
    th_op = torch.max(zero, div)
    msk = torch.min(oneT, th_op )
    msk = msk.expand(1,-1,-1, -1)
    msk = msk.repeat(3,1,1,1)
    msk = msk.permute(1,2,3,0)

    '''
    msk       = torch.max(in_y ,3)[0]
    print('reduced_msk: ', msk.shape)
    th        = torch.max(zeros, (msk - 1.0 + thr)/thr)
    msk       = torch.min(oneT, th )
    msk = msk.expand(1,-1,-1, -1)
    msk = msk.permute(1,0,2,3)
    msk = msk.repeat(1,3,1,1)
    '''
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
        sqr_of_log  = torch.pow(out_sub_log, 2.0)
        cost        = torch.mean(sqr_of_log)
        # print('>>>>> out_sub_log:', out_sub_log)
        # print('>>>>> sqr_of_log:', sqr_of_log)
        
        #print('cost: ', cost)
        # TODO test
        trgt_sub_log = torch.mul(torch.sub(y_log_, x_log), msk)
        trgt_square  = torch.pow(trgt_sub_log, 2.0)
        cost_input_output = torch.mean(trgt_square)
        #print('cost_input_output', cost_input_output)

    return cost, cost_input_output
# =====Learning parameters ======================================================

lambda_ir = 0.5
print('setup finished')

def train_net(net, epochs=5, batch_size=1, lr=0.001, val_percent=0.20,
              save_cp=True,
              gpu=False,
              img_scale=0.5,
              expositions_num=15,
              logg_freq = 15,
              tb=False,
              use_notifications=False,
              polyaxon=False,
              outputs_path='checkpoints'):
    
    # === Localize training data ===================================================
    if polyaxon:
        data_paths = get_data_paths()
        dir_checkpoints = get_outputs_path()
        #dataSets_dir = os.path.join(wk_dir,"data/", "LDR_DataSet")
        dataSets_dir = os.path.join(data_paths['data1'] , 'LDR_DataSet')

    else:
        dataSets_dir = os.path.join(wk_dir, "LDR_DataSet")
        dir_checkpoints = os.path.join(wk_dir, outputs_path)
    print('Dataset_dir' , dataSets_dir)
    print('Outputs_path', dir_checkpoints)
    experiment_id = datetime.datetime.now().strftime('%d%m_%H%M_')
    experiment_name = 'MSELoss_{}_bs{}_lr{}_exps{}'.format(experiment_id,batch_size,lr,expositions_num)
    dir_img = os.path.join(dataSets_dir, 'Org_images/')
    dir_compressions = os.path.join(dataSets_dir, 'c_images/')
    dir_mask = os.path.join(dataSets_dir, 'c_images/')
    
    #if tb:
        #dummy_input = torch.rand(1, 3, 128, 128)
        #writer.add_graph(net, (dummy_input,))
        #writer.close()
    # === Load Training/Validation data =====================================================
    ids = get_ids(dir_img)
    iddataset = split_train_val(ids, expositions_num, val_percent )
    #print(iddataset['train']) 
    #print(iddataset['val'])
    N_train = len(iddataset['train'])
    N_val = len(iddataset['val'])
    # optimizer = optim.SGD(net.parameters(),
    #    lr=lr,
    #    momentum=0.9,
    #    weight_decay=0.0005)

    #=====CHOOSE Loss Criterion=============================================================
    criterion = nn.MSELoss()
    #criterion = ExpandNetLoss()
    optimizer = optim.Adagrad(net.parameters(),
                              lr=lr,
                              weight_decay=0.0005)
   
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
        '''.format(epochs, batch_size, lr, N_train,
                  N_val, str(save_cp), str(gpu)))
    
    train_dataset = HdrDataset(iddataset['train'], dir_compressions, dir_mask,
                               expositions_num)
    train_data_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    
    val_dataset = HdrDataset(iddataset['val'], dir_compressions, 
                                         dir_mask,
                                         expositions_num)
    val_data_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True)
    
    for epoch in range(epochs):
        print('\n')
        print('{}{}{}'.format('+', '=' * 78 , '+'))       
        print('| Starting epoch {}/{}. {}'.format(epoch + 1, epochs,(' '*57) + '|'))
        print('{}{}{}'.format('|', '-' * 78 , '|')) 
        begin_of_epoch = time.time()
        tot_steps = math.trunc(N_train/batch_size)
        net.train()
        train_loss = 0
        val_loss = 0
        step = 0
        for i, b in enumerate(train_data_loader):
            step += 1
            imgs, true_masks, imgs_ids = b['input'], b['target'], b['id'] 
            #print(i, b['input'].size(), b['target'].size())
            #input: [15, 3, 224, 224]), target: [15, 3, 224, 224]
            #print('>>>>>>> Input max: ' , torch.max(imgs[0]))
            #print('>>>>>>> mask max : ', torch.max(true_masks[0]))
            
            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()
            else:
                print(' GPU not available')
            
            # Predicted mask images
            optimizer.zero_grad()
            prediction = net(imgs) #prediction shape: [B, 3, 224, 224]
            #cost, cost_input_output = Hdr_loss(imgs, true_masks, prediction, sep_loss=False, gpu=gpu, tb=tb)
            cost = criterion(prediction,true_masks)
            #loss is torch tensor
            train_loss += cost.item()
            cost.backward()
            optimizer.step()

            #print('>>>>>>>> Pred max: ', torch.max(prediction[0]))
            #masks_probs = F.sigmoid(prediction)
            #masks_probs_flat = masks_probs.view(-1)
            #true_masks_flat = true_masks.view(-1)
            
            #masks_probs_flat = prediction.view(-1)
            # Ground Truth images of masks
            #true_masks_flat = true_masks.view(-1)
            # Save a sample of input, output prediction on the last step - 2 of the epoch
            
            if step==1 or step % logg_freq == 0: 
                print('| Step: {0:}, cost:{1:}, Train Loss:{2:.9f}, Val Loss:{3:.6f}'.format(step,cost, train_loss,val_loss))   
            
               
            #Last Step
            if step ==  math.trunc(tot_steps):
                num_in_batch = random.randrange(imgs.size(0))
                train_sample_name = imgs_ids[num_in_batch]
                train_sample = [imgs[num_in_batch],true_masks[num_in_batch], prediction[num_in_batch]]

                t_exp_name = 'Train_' + experiment_name
                saveTocheckpoint(dir_checkpoints, t_exp_name, train_sample_name, epoch,
                                     train_sample[0],
                                     train_sample[1],
                                     train_sample[2])
                
                val_loss = eval_hdr_net(net,dir_checkpoints,experiment_name, val_data_loader,
                                                    criterion, epoch, gpu,
                                                    batch_size,
                                                    expositions_num=15, tb=tb)
                    

                if tb:
                    print('| saving train step {0:} sample : input,target & pred'.format(step))
                    grid = torchvision.utils.make_grid(train_sample,nrow=3)
                    writer.add_image('train_sample', grid, 0)
                        
        
        if tb:
                train_loss = (train_loss/N_train)
                val_loss   = (val_loss /N_val)
                writer.add_scalar('training_loss: ', train_loss, epoch )
                writer.add_scalar('validation_loss', val_loss, epoch )
                writer.add_scalars('losses', { 'training_loss': train_loss,
                                               'val_loss': val_loss}, epoch)
               
                if polyaxon:
                    experiment.log_metrics(training_loss=train_loss, validation_loss=val_loss)


        print('{}{}{}'.format('+', '=' * 78 , '+'))
        print('| {0:} Epoch {1:} finished ! {2:}|'.format(' '*28 ,(epoch + 1),' '*29 ))
        print('{}{}{}'.format('+', '-' * 78 , '+'))
        print('| Summary: Train Loss:{0:0.07}, Val Loss:{1:}'.format(train_loss, val_loss))
        time_epoch = time.time() - begin_of_epoch 
        print('| Epoch ETC: {:.0f}m {:.0f}s'.format(time_epoch // 60, time_epoch % 60))   
        print('{}{}{}'.format('+', '=' * 78 , '+'))
        
                
        # Training and validation loss for Tensorboard
        #file_writer.add_summary(valid_summary, step)
        #file_writer.add_summary(train_summary, step)

        if save_cp:
            model_path = os.path.join(dir_checkpoints, 'CP{}.pth'.format(epoch + 1))
            torch.save(net.state_dict(),
                       model_path)
            print('Checkpoint {} saved !'.format(epoch + 1))
    print('>' * 80)
    time_elapsed = time.time() - since   
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))     
    if tb:
        writer.close()
    if use_notifications:
        end_msg = "train.py finisheed at: {}(".format(str(currentDT))
        push = pb.push_note("usHDR: Finish", end_msg)
    

def eval_hdr_net(net,dir_checkpoints, experiment_name, dataloader,criterion,epoch, gpu=False,
              batch_size=1,
              expositions_num=15, tb=False):
    """Evaluation without the densecrf with the dice coefficient"""

    val_data_loader = dataloader
    net.eval()
    tot_loss = 0
    step = 0
    N_val =  len(val_data_loader)
    tot_steps = N_val/batch_size                                                                                            

    for i, b in enumerate(val_data_loader):
        
        step += 1
        imgs, true_masks, imgs_ids = b['input'], b['target'], b['id']
            
        #img = b[0]
        #true_mask = b[1]
        #img = torch.from_numpy(img).unsqueeze(0)
        #true_mask = torch.from_numpy(true_mask).unsqueeze(0)
        if gpu:
            imgs = imgs.cuda()
            true_masks = true_masks.cuda()

        pred = net(imgs)
        #cost, cost_input_output = Hdr_loss(imgs, true_masks, pred,sep_loss=False,gpu=gpu, tb=tb)
        cost = criterion(pred,true_masks)
        tot_loss += cost.item()
               
        # Last - 1 step
        if step == math.trunc(tot_steps):
            num_in_batch = random.randrange(imgs.size(0))
            val_sample_name = imgs_ids[num_in_batch]
            img_s  = imgs[num_in_batch]
            gt_s   = true_masks[num_in_batch]
            pred   = pred[num_in_batch]
            val_exp_name = 'Val_' + experiment_name
            saveTocheckpoint(dir_checkpoints, val_exp_name, val_sample_name, epoch,
                            img_s,
                            gt_s,
                            pred)
            
    return tot_loss

def get_args():
    parser = OptionParser()
    
    parser.add_option('-b', '--batch-size', dest='batchsize', default=15,
                      type='int', help='batch size')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-f', '--logg-freq', dest='frequency', default=15,
                      type='int', help='requency for loggind data to terminal')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-m', '--save-cp', dest='save',
                      default=False, help='save model')
    parser.add_option('-n', '--notifications', action='store_true', dest='pushbullet',
                      default=False, help='use pushbullet notifications')
    parser.add_option('-o', '--outputs-path', action= 'store',dest='outputs',
                      default='checkpoints', help='Define outputs folder')
    parser.add_option('-p', '--polyaxon', action='store_true', dest='polyaxon',
                      default=False, help='set data dirs to use polyaxon')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')
    parser.add_option('-t', '--tensorboard', action='store_true', dest='tensorboard',
                      default=False, help='use tensorboard logging')
    parser.add_option('-x', '--expo-num', dest='expositions', default=15,
                      type='int', help='number of exposition that compund an HDR.')
    
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
                  save_cp= args.save,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale,
                  expositions_num= args.expositions,
                  logg_freq=args.frequency,
                  tb=args.tensorboard,
                  use_notifications=args.pushbullet,
                  polyaxon=args.polyaxon,
                  outputs_path=args.outputs)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')

        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)



