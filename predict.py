import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
import cv2

from unet import UNet
from utils import only_resizeCV, map_range,cv2torch,process_path normalize, split_img_into_squares, hwc_to_chw, merge_masks, dense_crf, preprocess


from tmo import (
    process_path,
    split_path,
    map_range,
    tone_map,
    create_tmo_param_from_args,
)



from torchvision import transforms

'''
tmo choices
['exposure', 'reinhard', 'durand']
'''

def predict_img(net,
                ldr_img,
                scale_factor=0.5,
                out_threshold=0.5,
                use_dense_crf=True,
                use_gpu=False,
                output_path: '',
                tone_map: 'durand'):

    img_height = ldr_img.size[1]
    img_width = ldr_img.size[0]

    # preprocess
    img = ldr_img.astype('float32')
    img = only_resizeCV(img,224,224)
    ldr_input = map_range(img)

    t_ldr = cv2torch(ldr_input)

    if use_gpu:
        t_ldr = t_ldr.cuda()
        
    with torch.no_grad():
        pred = net(t_ldr)
        pred = map_range(torch2cv(pred).cpu(),0,1)

        # extension = 'exr' if opt.use_exr else 'hdr'
        extension = 'hdr'
        out_name =  create_name(ldr_img,'prediction_{0}'.format(),extension,output_path)

        print(f'Writing {out_name}')
        cv2.imwrite(out_name, pred)
        tmo_img =  tone_map(pred,tone_map,**create_tmo_param_from_args(tone_map))

        out_name = create_name(
                ldr_img,
                'prediction_{0}'.format(tone_map),
                'jpg',
                output_path
            )

        cv2.imwrite(out_name, (tmo_img * 255).astype(int))

    if use_dense_crf:
        full_mask = dense_crf(np.array(ldr_img).astype(np.uint8), full_mask)

    return full_mas k > out_threshold



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--no-crf', '-r', action='store_true',
                        help="Do not use dense CRF postprocessing",
                        default=False)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()

def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=1)

    print("Loading model {}".format(args.model))

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))

        #OPen with PIL
        #img = Image.open(fn)
        # Open with opencv
        img = cv2.imread(fn,flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR) 
        if img is None:
            print('Could not load {0}'.format(ldr_file))
            continue

        if img.size[0] < img.size[1]:
            print("Error: image height larger than the width")

        mask = predict_img(net=net,
                           ldr_img=img,
                           scale_factor=args.scale,
                           use_dense_crf= not args.no_crf,
                           use_gpu=not args.cpu)

        if args.viz:
            print("Visualizing results for image {}, close to continue ...".format(fn))
            print('TODO: plot tmo image no mask needed to be plotted')
            plot_img_and_mask(img, mask)

        if not args.no_save:
            out_fn = out_files[i]
            print('TODO save tmo_image no mask ')
            result = mask_to_image(mask)
            result.save(out_files[i])

            print("Mask saved to {}".format(out_files[i]))

def create_name(inpath, tag,ext,out,extra_tag):
    root, name, _ = split_path(inpath)

    if extra_tag is not None:
        tag = '{0}_{1}'.format(tag,extra_tag)
    if out is not None:
        root = out
    return os.path.join(root, '{0}_{1}.{2}'.format(name,tag,ext))

