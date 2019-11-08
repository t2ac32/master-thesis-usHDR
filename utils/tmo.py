#tmo.py

import os
import numpy as np
from numpy.random import uniform
import torch
from torch.utils.data import Dataset
import cv2

class BaseTMO(object):
    def __call__(self, img):
        return self.op.process(img)

class Reinhard(BaseTMO):
    def __init__(
        self,
        intensity=-1.0,
        light_adapt=0.8,
        color_adapt=0.0,
        gamma=2.0,
        randomize=False,
    ):
        if randomize:
            gamma = uniform(1.8, 2.2)
            intensity = uniform(-1.0, 1.0)
            light_adapt = uniform(0.8, 1.0)
            color_adapt = uniform(0.0, 0.2)
        self.op = cv2.createTonemapReinhard(
            gamma=gamma,
            intensity=intensity,
            light_adapt=light_adapt,
            color_adapt=color_adapt,
        )

class Durand(BaseTMO):
    def __init__(
        self,
        contrast=3,
        saturation=1.0,
        sigma_space=8,
        sigma_color=0.4,
        gamma=2.0,
        randomize=False,
    ):
        if randomize:
            gamma = uniform(1.8, 2.2)
            contrast = uniform(3.5)
        self.op = cv2.createTonemapDurand(
            contrast=contrast,
            saturation=saturation,
            sigma_space=sigma_space,
            sigma_color=sigma_color,
            gamma=gamma,
        )
TMO_DICT = {
    'reinhard': Reinhard,
    'durand': Durand,
}

def tone_map(img, tmo_name, **kwargs):
    return TMO_DICT[tmo_name](**kwargs)(img)

def create_tmo_param_from_args(opt):
	if opt.tmo == 'exposure':
	    return {k: opt.get(k) for k in ['gamma', 'stops']}
	else:  # TODO: Implement for others
	    return {}