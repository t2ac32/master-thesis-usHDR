
import time, math, os, sys, random
import datetime
from optparse import OptionParser
import numpy as np
import threading
import scipy.stats as st
import sklearn
from sklearn.model_selection import KFold

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

#from GPUtil import showUtilization as gpu_usage

#### Accuracy measurements
from psnrhvsm import psnrhvsm
import pytorch_ssim  
#from skimage.measure import compare_ssim as ssim



from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path

from eval import eval_net
from unet import UNet

from pushbullet import Pushbullet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch, exist_program, HdrDataset, saveTocheckpoint, tmo
from utils import (
    tone_map,
    create_tmo_param_from_args,
)


try:
    print('Loading Tensorboard')
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

# =====Learning parameters ======================================================

print('setup finished')

def train_net(net, epochs=5, batch_size=1, lr=0.003, val_percent=0.20,loss_lambda=5,
              save_cp=True,
              gpu=False,
              img_scale=0.5,
              expositions_num=15,
              logg_freq = 15,
              tb=False,
              w_decay=0.0005,
              use_notifications=False,
              polyaxon=False,
              outputs_path='checkpoints'):
    
    # === Localize training data ===================================================
    if polyaxon:
        data_paths = get_data_paths()
        dir_checkpoints = get_outputs_path()
        dataSets_dir = os.path.join(data_paths['data1'] , 'eprado', 'USLDR-DataSet')
        #dataSets_dir = os.path.join(data_paths['data1'] , 'eprado', 'LDR_DataSet')

    else:
        dataSets_dir = os.path.join(wk_dir, "LDR_DataSet")
        dir_checkpoints = os.path.join(wk_dir, outputs_path)
    print('Dataset_dir' , dataSets_dir)
    print('Outputs_path', dir_checkpoints)
    experiment_id = datetime.datetime.now().strftime('%d%m_%H%M_')
    experiment_name = 'ExpandnetL_psn_{}_bs{}_lr{}_exps{}'.format(experiment_id,batch_size,lr,expositions_num)
    dir_img = os.path.join(dataSets_dir, 'Org_images/')
    dir_compressions = os.path.join(dataSets_dir, 'c_images/')
    dir_mask = os.path.join(dataSets_dir, 'c_images/')
    
    #if tb:
        #dummy_input = torch.rand(1, 3, 128, 128)
        #writer.add_graph(net, (dummy_input,))
        #writer.close()
    # === Load Training/Validation data =====================================================
    ids = get_ids(dir_compressions)
    # Split into train test
    idsset= list(ids)

    kf = KFold(n_splits=5, shuffle=False)
    #print('Train splits: ',kf.get_n_splits(dataset))

    best_psnr_m = 0 
    best_psnr_hvs = 0 
    #for train_index, test_index in kf.split(idsset):
    iddataset = split_train_val(idsset, expositions_num, val_percent )
    #test_set = [] 
    #for im_id in test_index:
    #    for e in range(expositions_num):
    #        test_set.append(idsset[im_id])
    
    N_train = len(iddataset['train'])
    N_val = len(iddataset['val'])
    N_test = 0 #len(test_set)
    

    #=====CHOOSE Loss Criterion=============================================================
    #criterion = nn.MSELoss(reduction='mean')
    criterion = ExpandNetLoss(loss_lambda=loss_lambda)
    optimizer = optim.Adagrad(net.parameters(),
                              lr=lr,
                              lr_decay= 0.000001,
                              weight_decay=w_decay)
    #optimizer = optim.SGD(net.parameters(),
    #   lr=lr,
    #   momentum=0.9,
    #   weight_decay=0.0005)

    since = time.time()
    print('''
        Training SETUP:
        Epochs: {0:}
        Batch size: {1:}
        Optimizer: Adagrad
        Learning rate: {2:}
        Weight decay: {3:}
        Training size: {4:}
        Validation size: {5:}
        Test size: {6:}
        Checkpoints: {7:}
        CUDA: {8:}
        '''.format(epochs, batch_size, lr,w_decay, N_train,
                  N_val,N_test, str(save_cp), str(gpu)))
    
    train_dataset = HdrDataset(iddataset['train'], dir_compressions, dir_mask,
                               expositions_num)
    val_dataset = HdrDataset(iddataset['val'], dir_compressions, 
                                         dir_mask,
                                         expositions_num)
    #test_dataset = HdrDataset(test_set, dir_compressions, dir_mask,expositions_num)

    train_data_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=False)
    val_data_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,drop_last=False)
    #test_data_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
    best_hvsm = 0.0
    global_psnr_m = []
    global_psnr_hvs = []
    for epoch in range(epochs):
        print('\n')
        print('{}{}{}'.format('+', '=' * 78 , '+'))       
        print('| Starting epoch {}/{}. {}'.format(epoch + 1, epochs,(' '*57) + '|'))
        print('{}{}{}'.format('|', '-' * 78 , '|')) 
        begin_of_epoch = time.time()
        tot_steps = math.trunc(N_train/batch_size)
        net.train()
        train_loss = 0
        losses = []
        val_loss = 0
        step = 0
        train_sample = []
        train_acc = 0 
        val_hvsm = 0
        val_hvs = 0
        model_pnsr_m = 0
        
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
            losses.append(cost.item())
            
            train_loss = np.mean(losses) 
            cost.backward()
            optimizer.step()
            
            if step==1 or step % logg_freq == 0: 
                #print('| Step: {0:}, cost:{1:}, Train Loss:{2:.9f}, Train Acc:{3:.9f}'.format(step,cost, train_loss,train_acc/step)) 
                print('| Step: {0:}, cost:{1:}, Train Loss:{2:.9f}'.format(step,cost, train_loss)) 
               
            #Last Step of this Epoch
            if step ==  math.trunc(tot_steps):
                num_in_batch = random.randrange(imgs.size(0))
                train_sample_name = imgs_ids[num_in_batch]
                train_sample = [imgs[num_in_batch],true_masks[num_in_batch], prediction[num_in_batch]]

                t_exp_name = 'Train_' + experiment_name
                saveTocheckpoint(dir_checkpoints, t_exp_name, train_sample_name, epoch,
                                     train_sample[0],
                                     train_sample[1],
                                     train_sample[2])
                
                if tb:
                    print('| saving train step {0:} sample : input,target & pred'.format(step))
                    grid = torchvision.utils.make_grid(train_sample,nrow=3)
                    writer.add_image('train_sample', grid, 0)
        
        #if  epoch == 1 or epoch % 15 == 0 or epoch == epochs: 
        
        val_loss, val_hvsm, val_hvs  = eval_hdr_net(net,dir_checkpoints,experiment_name, val_data_loader,
                                    criterion, epoch, gpu,
                                    batch_size,
                                    expositions_num=15, tb=tb)   
        if tb:
                writer.add_scalar('training_loss: ', train_loss, epoch )
                writer.add_scalar('validation_loss', val_loss, epoch )
                writer.add_scalar('val_hvsm', val_hvsm, epoch )
                writer.add_scalar('val_hvs', val_hvs , epoch )
                writer.add_scalars('losses', { 'training_loss': train_loss,
                                               'val_loss': val_loss}, epoch)   
                if polyaxon:
                    experiment.log_metrics(step=epoch,training_loss=train_loss,
                                            validation_loss=val_loss, val_hvsm= val_hvsm, val_hvs=val_hvs )


        print('{}{}{}'.format('+', '=' * 78 , '+'))
        print('| {0:} Epoch {1:} finished ! {2:}|'.format(' '*28 ,(epoch + 1),' '*29 ))
        print('{}{}{}'.format('+', '-' * 78 , '+'))
        print('| Summary: Train Loss: {0:0.07}, Val Loss:{1:}'.format(train_loss, val_loss))
        print('|          Avrg psnr-hvs_m :{0:0.04},Avrg psnr-hvs :{1:0.04}'.format(val_hvsm, val_hvs))
        time_epoch = time.time() - begin_of_epoch 
        print('| Epoch ETC: {:.0f}m {:.0f}s'.format(time_epoch // 60, time_epoch % 60))   
        print('{}{}{}'.format('+', '=' * 78 , '+'))     
            
        if save_cp and (val_hvsm > best_hvsm):
            best_hvsm = val_hvsm
            model_path = os.path.join(dir_checkpoints, 'BestCP.pth')
            torch.save(net.state_dict(), model_path)
            print('Checkpoint saved !')
        global_psnr_hvs.append(val_hvs)
        global_psnr_m.append(val_hvsm)
    '''
    test_psnr_m, test_psnr_hvs = test_hdr_net(model_path,dir_checkpoints,
                                                experiment_name,
                                                test_data_loader,
                                                criterion,gpu,tb)
                                                if save_cp and (test_psnr_m > best_psnr_m):
    best_psnr_m = test_psnr_m
    best_model_path = os.path.join(dir_checkpoints, 'Best_CP.pth')
    torch.save(net.state_dict(),best_model_path)
    print('Best model saved !')
    '''
    print('>' * 80)
    time_elapsed = time.time() - since   
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) 
    print('Final Average psnr-hvs_m: {:.0f}, psnr-hvs: {:.0f}'.format(np.mean(global_psnr_m), np.mean(global_psnr_hvs)))
    if tb:
        writer.close()
    if use_notifications:
        end_msg = "train.py finished at: {}(".format(str(datetime.datetime.now()))
        push = pb.push_note("usHDR: Finish", end_msg)
    

def eval_hdr_net(net,dir_checkpoints, experiment_name, dataloader,criterion,epoch, gpu=False,
              batch_size=1,
              expositions_num=15, tb=False):
    """Evaluation without the densecrf with the dice coefficient"""
    print('{}{}{}'.format('+', '=' * 78 , '+'))
    print('| {0:} validating {2:} |'.format(' '*32 ,(epoch + 1),' '*32 ))
    
    t_map = 'durand'
    val_data_loader = dataloader
    net.eval()
    losses = []
    step = 0
    N_val =  len(val_data_loader)
                                                                                            
    tot_psnrm = 0
    tot_psnrhvs = 0
    with torch.no_grad():
        for i, b in enumerate(val_data_loader):
            
            step += 1
            imgs, true_masks, imgs_ids = b['input'], b['target'], b['id']
                

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()
            else:
                print(' GPU not available')

            pred = net(imgs)
            
            #cost, cost_input_output = Hdr_loss(imgs, true_masks, pred,sep_loss=False,gpu=gpu, tb=tb)   
            cost = criterion(pred,true_masks)
            losses.append(cost.item())    
            
            batch_hvsm, batch_hvs = get_psnrhs(true_masks,pred,1)
            tot_psnrm += batch_hvsm
            tot_psnrhvs += batch_hvs

            # Last - 1 step
            if step == math.trunc(N_val):
                num_in_batch = random.randrange(imgs.size(0))
                val_sample_name = imgs_ids[num_in_batch]
                img_s  = imgs[num_in_batch]
                gt_s   = true_masks[num_in_batch]
                pred_img   = pred[num_in_batch]
                val_exp_name = 'Val_' + experiment_name
                saveTocheckpoint(dir_checkpoints, val_exp_name, val_sample_name, epoch,
                                img_s,
                                gt_s,
                                pred_img)
                tmo_img =  tone_map(pred_img,t_map)

                dir_path = os.path.join(dir_checkpoints,val_exp_name,'durandTMO')
                file_name  = dir_path + '/epoch{0:02d}_{1:}'.format(epoch+1,val_sample_name) + '.jpg'  
                cv2.imwrite(file_name, (tmo_img * 255).astype(int))

                
        return np.mean(losses), tot_psnrm/N_val, tot_psnrhvs/N_val

def test_hdr_net(model_path,dir_checkpoints, experiment_name, dataloader,criterion,
                gpu=False,
                expositions_num=15,
                tb=False):
    
    print('{}{}{}'.format('+', '=' * 78 , '+'))
    print('| {0:} Testing {1:}|'.format(' '*30 ,' '*30 ))
    print('{}{}{}'.format('+', '=' * 78 , '+'))
    tot_psnrm = 0
    tot_psnrhvs = 0
    steps = 0
    for i, b in enumerate(dataloader):
        steps += 1
        imgs, true_masks, imgs_ids = b['input'], b['target'], b['id'] 
        net = UNet(n_channels=3, n_classes=3)
        net.load_state_dict(torch.load(model_path))

        if gpu:
            net.cuda()
            imgs = imgs.cuda()
            true_masks = true_masks.cuda()
        else:
            print(' GPU not available')
        
        
        pred = net(imgs)
        
        batch_hvsm, batch_hvs = get_psnrhs(true_masks,pred,1)
        tot_psnrm += batch_hvsm
        tot_psnrhvs += batch_hvs

    avg_psnr_m = tot_psnrm/steps
    avg_psnr_hvs = tot_psnrhvs/steps
    print('| AVG PSNR-HVS-M: {0:0.04} | AVG PSNR-hvs: {1:0.04} '.format(avg_psnr_m,avg_psnr_hvs  ))
    print('{}{}{}'.format('+', '-' * 78 , '+'))
    return  avg_psnr_m, avg_psnr_hvs

def get_psnrhs(masks,preds,batch_size):
    batch_hvsm = 0 
    batch_hvs = 0 

    if masks.size(0) < batch_size:
        batch_size = masks.size(0)

    for index in range(batch_size):
        mask = masks[index]
        pred = preds[index]
        p_hvs_m, p_hvs = psnrhvsm(mask, pred)
        batch_hvsm += p_hvs_m
        batch_hvs += p_hvs
    batch_hvsm = p_hvs_m / batch_size
    batch_hvs = p_hvs / batch_size 
       
    return batch_hvsm, batch_hvs

    
def get_acc (masks,preds,batch_size):
    mssim = 0 
    if masks.size(0) < batch_size:
        batch_size = masks.size(0)

    for index in range(batch_size):

        mask = Variable( masks[index].unsqueeze(0) )
        pred = Variable( preds[index].unsqueeze(0) )
        mssim += pytorch_ssim.ssim(mask, pred)
        #mssim = ssim(mask,pred,multichannel=True,gaussian_weights=True)
        
    return mssim / batch_size

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
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.001,
                      type='float', help='learning rate')
    parser.add_option('-L', '--loss-lambda', dest='loss_lambda', default=5,
                      type='float', help='Loss function lambda param')
    parser.add_option('-m', '--save-cp', action='store_true', dest='save',
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
    parser.add_option('-w', '--weight-decay', dest='wd', default=0.0005,
                      type='float', help='weight decay')
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
                  loss_lambda=args.loss_lambda,
                  gpu=args.gpu,
                  img_scale=args.scale,
                  expositions_num= args.expositions,
                  logg_freq=args.frequency,
                  tb=args.tensorboard,
                  w_decay=args.wd,
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



