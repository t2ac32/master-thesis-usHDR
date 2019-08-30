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



    def __init__(self, x=None, y=None, sy=None, sx=None, sep_loss=False, reduction='mean'):
        super(HdrLoss, self).__init__(x, y, sx, sy, reduce, reduction)
        # === Loss function formulation ================================================
        thr = 0.05
        msk = torch.max(y,dim = 3, keepdim=False)
        msk = torch.min(1.0, torch.max(0.0, msk-1.0+thr)/thr)
        msk = torch.reshape(msk,[-1, sy, sx, 1])

        eps = 1.0 / 255.0

        buffer_size = 256
        lambda_ir = 0.5
        
        # input_ = np.random.rand(4,230,230,3)
        # target = np.random.rand(4,230,230,3)
        y = np.random.rand(4,230,230,3)

    # Loss separated into illumination and reflectance terms
        if  sep_loss:
            y_log = torch.log(y + eps)
            x_log =  torch.log(torch.pow(x,2.0) + eps)

            # Luminance
            lum_kernel = (1, 1, 3, 1)
            lum_kernel[:, :, 0, 0] = 0.213
            lum_kernel[:, :, 1, 0] = 0.715
            lum_kernel[:, :, 2, 0] = 0.072
            ##TODO: the padding must be SAME but pytorch needs to be calculated by hand
            ## first check what the y output size must boe the get P in formula for padding same
            ## replace padding='SAME' in nn.conv2d for results of P computation
            ## Formula to solve:
            ## o = output, p =padding, k =kernel_size, s =stride, d=dilation
            ## o == [i + 2*p - k - (k - 1)*(d-1)]/s + 1
            y_lum_lin_ = nn.conv2d(y_, lum_kernel, (1, 1, 1, 1), padding='SAME')
            y_lum_lin = nn.conv2d(tf.exp(y) - eps, lum_kernel, [1, 1, 1, 1], padding='SAME')
            x_lum_lin = nn.conv2d(x, lum_kernel, [1, 1, 1, 1], padding='SAME')


