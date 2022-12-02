import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from collections import OrderedDict
import numpy as np
import cv2
import os
import glob
import math
import time
import argparse

from film_arch import FILM

device = torch.device('cuda')
#device = torch.device('cpu')

def load_network(net, weight_path):
    print('loading weights from {}...'.format(weight_path))

    if isinstance(net, nn.DataParallel) or isinstance(net, DistributedDataParallel):
        net = net.module
    load_net = torch.load(weight_path)['params']
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    net.load_state_dict(load_net_clean, strict=True)
    return net

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='photos', help='Input image or folder')
    parser.add_argument('--model_path', type=str, default='film.pth', help='Path to the pre-trained model')
    parser.add_argument('--output', type=str, default='results', help='Output folder')
    parser.add_argument('--suffix', type=str, default='film', help='Suffix of the restored image')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    net = FILM()
    net = load_network(net, args.model_path)
    net.to(device)
    net.eval()

    img = cv2.imread(f'{args.input}/one.png')
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).flip(0).float()
    I0 = img.unsqueeze(0).to(device) / 255.

    img = cv2.imread(f'{args.input}/two.png')
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).flip(0).float()
    I1 = img.unsqueeze(0).to(device) / 255.

    with torch.no_grad():
        out = net.forward(I0, I1, 0.5)

    out = out.squeeze(0).permute(1, 2, 0).flip(2).cpu().numpy()
    cv2.imwrite(os.path.join(args.output, 'out_'+args.suffix+'.png'), out*255)