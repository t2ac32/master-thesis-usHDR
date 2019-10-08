import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
from ..._jit_internal import weak_module, weak_script_method



# ==== R ====
#  arguments
#  y_ =
@weak_module
class HdrLoss(_WeightedLoss):

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