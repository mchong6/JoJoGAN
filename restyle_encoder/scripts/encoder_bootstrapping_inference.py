import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys

from utils.inference_utils import get_average_image

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from options.test_options import TestOptions
from models.psp import pSp
from models.e4e import e4e
from utils.model_utils import ENCODER_TYPES
from utils.common import tensor2im


def run():
    test_opts = TestOptions().parse()

    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    os.makedirs(out_path_results, exist_ok=True)

    # load model used for initializing encoder bootstrapping
    ckpt = torch.load(test_opts.model_1_checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts['checkpoint_path'] = test_opts.model_1_checkpoint_path
    opts = Namespace(**opts)
    if opts.encoder_type in ENCODER_TYPES['pSp']:
        net1 = pSp(opts)
    else:
        net1 = e4e(opts)
    net1.eval()
    net1.cuda()

    # load model used for translating input image after initialization
    ckpt = torch.load(test_opts.model_2_checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts['checkpoint_path'] = test_opts.model_2_checkpoint_path
    opts = Namespace(**opts)
    if opts.encoder_type in ENCODER_TYPES['pSp']:
        net2 = pSp(opts)
    else:
        net2 = e4e(opts)
    net2.eval()
    net2.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=False)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    # get the image corresponding to the latent average
    avg_image = get_average_image(net1, opts)

    resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)

    global_i = 0
    global_time = []
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break
        with torch.no_grad():
            input_cuda = input_batch.cuda().float()
            tic = time.time()
            result_batch = run_on_batch(input_cuda, net1, net2, opts, avg_image)
            toc = time.time()
            global_time.append(toc - tic)

        for i in range(input_batch.shape[0]):
            results = [tensor2im(result_batch[i][iter_idx]) for iter_idx in range(opts.n_iters_per_batch + 1)]
            im_path = dataset.paths[global_i]

            input_im = tensor2im(input_batch[i])

            # save step-by-step results side-by-side
            res = np.array(results[0].resize(resize_amount))
            for idx, result in enumerate(results[1:]):
                res = np.concatenate([res, np.array(result.resize(resize_amount))], axis=1)
            res = np.concatenate([res, input_im.resize(resize_amount)], axis=1)
            Image.fromarray(res).save(os.path.join(out_path_results, os.path.basename(im_path)))

            global_i += 1

    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)


def run_on_batch(inputs, net1, net2, opts, avg_image):
    y_hat, latent = None, None
    results_batch = {idx: [] for idx in range(inputs.shape[0])}

    # initialize using the first net
    avg_image_for_batch = avg_image.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
    x_input = torch.cat([inputs, avg_image_for_batch], dim=1)
    y_hat, latent = net1.forward(x_input,
                                 latent=latent,
                                 randomize_noise=False,
                                 return_latents=True,
                                 resize=opts.resize_outputs)
    for idx in range(inputs.shape[0]):
        results_batch[idx].append(y_hat[idx])
    y_hat = net1.face_pool(y_hat)

    # iteratively translate using the resulting latent and generated image
    for iter in range(opts.n_iters_per_batch):
        x_input = torch.cat([inputs, y_hat], dim=1)
        y_hat, latent = net2.forward(x_input,
                                     latent=latent,
                                     randomize_noise=False,
                                     return_latents=True,
                                     resize=opts.resize_outputs)
        for idx in range(inputs.shape[0]):
            results_batch[idx].append(y_hat[idx])
        y_hat = net1.face_pool(y_hat)

    return results_batch


if __name__ == '__main__':
    run()
