import os
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from argparse import Namespace
from restyle_encoder.models.psp import pSp
from restyle_encoder.utils.inference_utils import run_on_batch

from util import *

def get_avg_image(net, device):
    avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    avg_image = avg_image.to(device).float().detach()
    return avg_image

@ torch.no_grad()
def projection(img, name, device='cuda'):
    model_path = 'models/restyle_psp_ffhq_encode.pt'
    ensure_checkpoint_exists(model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts['n_iters_per_batch'] = 5
    opts['device'] = device
    opts= Namespace(**opts)
    opts.resize_outputs = False
    net = pSp(opts).eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    img = transform(img).unsqueeze(0).to(device)
    avg_image = get_avg_image(net, device)
    result_batch, result_latents = run_on_batch(img, net, opts, avg_image)
    final_latent = result_latents[0][-1]
    result_file = {}
    result_file['latent'] = final_latent
    torch.save(result_file, name)
    return 

