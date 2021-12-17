# ReStyle: A Residual-Based StyleGAN Encoder via Iterative Refinement (ICCV 2021)

> Recently, the power of unconditional image synthesis has significantly advanced through the use of Generative Adversarial Networks (GANs). The task of inverting an image into its corresponding latent code of the trained GAN is of utmost importance as it allows for the manipulation of real images, leveraging the rich semantics learned by the network. Recognizing the limitations of current inversion approaches, in this work we present a novel inversion scheme that extends current encoder-based inversion methods by introducing an iterative refinement mechanism. Instead of directly predicting the latent code of a given image using a single pass, the encoder is tasked with predicting a residual with respect to the current estimate of the inverted latent code in a self-correcting manner. Our residual-based encoder, named ReStyle, attains improved accuracy compared to current state-of-the-art encoder-based methods with a negligible increase in inference time. We analyze the behavior of ReStyle to gain valuable insights into its iterative nature. We then evaluate the performance of our residual encoder and analyze its robustness compared to optimization-based inversion and state-of-the-art encoders.

<a href="https://arxiv.org/abs/2104.02699"><img src="https://img.shields.io/badge/arXiv-2104.02699-b31b1b.svg" height=22.5></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" height=22.5></a>  

<a href="https://www.youtube.com/watch?v=9RzCZZBjlxM"><img src="https://img.shields.io/static/v1?label=Two Minute Papers&message=ReStyle Video&color=red" height=22.5></a>  
<a href="https://youtu.be/6pGzLECSIWM"><img src="https://img.shields.io/static/v1?label=ICCV 2021 &message=5 Minute Video&color=red" height=22.5></a>  
<a href="https://replicate.ai/yuval-alaluf/restyle_encoder"><img src="https://img.shields.io/static/v1?label=Replicate&message=Demo and Docker Image&color=darkgreen" height=22.5></a>

Inference Notebook: <a href="http://colab.research.google.com/github/yuval-alaluf/restyle-encoder/blob/master/notebooks/inference_playground.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=20></a>  
Animation Notebook: <a href="http://colab.research.google.com/github/yuval-alaluf/restyle-encoder/blob/master/notebooks/animations_playground.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=20></a>  


<p align="center">
<img src="docs/teaser.jpg" width="800px"/>  
<br>
Different from conventional encoder-based inversion techniques, our residual-based ReStyle scheme incorporates an iterative refinement mechanism to progressively converge to an accurate inversion of real images. For each domain, we show the input image on the left followed by intermediate inversion outputs.
</p>

## Description   
Official Implementation of our ReStyle paper for both training and evaluation. ReStyle introduces an iterative
refinement mechanism which can be applied over different StyleGAN encoders for solving the StyleGAN inversion task.


## Getting Started
### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN (CPU may be possible with some modifications, but is not inherently supported)
- Python 3

### Installation
- Dependencies:  
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). 
All dependencies for defining the environment are provided in `environment/restyle_env.yaml`.

## Pretrained Models
In this repository, we provide pretrained ReStyle encoders applied over the 
[pSp](https://github.com/eladrich/pixel2style2pixel) and [e4e](https://github.com/omertov/encoder4editing) encoders 
across various domains.

Please download the pretrained models from the following links.

### ReStyle + pSp
| Path | Description
| :--- | :----------
|[FFHQ - ReStyle + pSp](https://drive.google.com/file/d/1sw6I2lRIB0MpuJkpc8F5BJiSZrc0hjfE/view?usp=sharing)  | ReStyle applied over pSp trained on the [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset.
|[Stanford Cars - ReStyle + pSp](https://drive.google.com/file/d/1zJHqHRQ8NOnVohVVCGbeYMMr6PDhRpPR/view?usp=sharing)  | ReStyle applied over pSp trained on the [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) dataset.
|[LSUN Church - ReStyle + pSp](https://drive.google.com/file/d/1bcxx7mw-1z7dzbJI_z7oGpWG1oQAvMaD/view?usp=sharing)  | ReStyle applied over pSp trained on the [LSUN](https://www.yf.io/p/lsun) Church dataset.
|[AFHQ Wild - ReStyle + pSp](https://drive.google.com/file/d/1GyFXVTNDUw3IIGHmGS71ChhJ1Rmslhk7/view?usp=sharing)  | ReStyle applied over pSp trained on the [AFHQ Wild](https://github.com/clovaai/stargan-v2) dataset.

### ReStyle + e4e
| Path | Description
| :--- | :----------
|[FFHQ - ReStyle + e4e](https://drive.google.com/file/d/1e2oXVeBPXMQoUoC_4TNwAWpOPpSEhE_e/view?usp=sharing)  | ReStyle applied over e4e trained on the [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset.
|[Stanford Cars - ReStyle + e4e](https://drive.google.com/file/d/1v6Yi-Yne6JX0Pyas6XPS83IzKKUvCe8r/view?usp=sharing)  | ReStyle applied over e4e trained on the [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) dataset.
|[LSUN Church - ReStyle + e4e](https://drive.google.com/file/d/1sCRzbM1fGTgTbWO87jP_rxApw-tS0CH_/view?usp=sharing)  | ReStyle applied over e4e trained on the [LSUN](https://www.yf.io/p/lsun) Church dataset.
|[AFHQ Wild - ReStyle + e4e](https://drive.google.com/file/d/1HQlUPKirShHyL3kGnwVRjhfcb-j9s4V3/view?usp=sharing)  | ReStyle applied over e4e trained on the [AFHQ Wild](https://github.com/clovaai/stargan-v2) dataset.
|[LSUN Horse - ReStyle + e4e](https://drive.google.com/file/d/19_sUpTYtJmhSAolKLm3VgI-ptYqd-hgY/view?usp=sharing)  | ReStyle applied over e4e trained on the [LSUN](https://www.yf.io/p/lsun) Horse dataset.

### Auxiliary Models
In addition, we provide various auxiliary models needed for training your own ReStyle models from scratch.  
This includes the StyleGAN generators and pre-trained models used for loss computation.

| Path | Description
| :--- | :----------
|[FFHQ StyleGAN](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing) | StyleGAN2 model trained on FFHQ with 1024x1024 output resolution.
|[LSUN Car StyleGAN](https://drive.google.com/file/d/1UmMHHB3DU1trTB8_9Fjkck5ZwArnD81B/view?usp=sharing) | StyleGAN2 model trained on LSUN Car with 512x384 output resolution.
|[LSUN Church StyleGAN](https://drive.google.com/file/d/1ZSZOFtw5wZfPhWEzHFB9LI__bikaHsn6/view?usp=sharing) | StyleGAN2 model trained on LSUN Church with 256x256 output resolution.
|[LSUN Horse StyleGAN](https://drive.google.com/file/d/1bM7ms1tWCucnySjiS6-NNrPJGDkqQQWi/view?usp=sharing) | StyleGAN2 model trained on LSUN Horse with 256x256 output resolution.
|[AFHQ Wild StyleGAN](https://drive.google.com/file/d/1z6IVVaCJuFTksKwp1CM3emWOVHbrBip-/view?usp=sharing) | StyleGAN-ADA model trained on AFHQ Wild with 512x512 output resolution.
|[IR-SE50 Model](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing) | Pretrained IR-SE50 model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in our ID loss and encoder backbone on human facial domain.
|[ResNet-34 Model](https://download.pytorch.org/models/resnet34-333f7ec4.pth) | ResNet-34 model trained on ImageNet taken from [torchvision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) for initializing our encoder backbone.
|[MoCov2 Model](https://drive.google.com/file/d/18rLcNGdteX5LwT7sv_F7HWr12HpVEzVe/view) | Pretrained ResNet-50 model trained using MOCOv2 for computing MoCo-based loss on non-facial domains. The model is taken from the [official implementation](https://github.com/facebookresearch/moco).
|[CurricularFace Backbone](https://drive.google.com/file/d/1f4IwVa2-Bn9vWLwB-bUwm53U_MlvinAj/view?usp=sharing) | Pretrained CurricularFace model taken from [HuangYG123](https://github.com/HuangYG123/CurricularFace) for use in ID similarity metric computation.
|[MTCNN](https://drive.google.com/file/d/1tJ7ih-wbCO6zc3JhI_1ZGjmwXKKaPlja/view?usp=sharing) | Weights for MTCNN model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in ID similarity metric computation. (Unpack the tar.gz to extract the 3 model weights.)

Note: all StyleGAN models are converted from the official TensorFlow models to PyTorch using the conversion script from [rosinality](https://github.com/rosinality/stylegan2-pytorch).

By default, we assume that all auxiliary models are downloaded and saved to the directory `pretrained_models`. 
However, you may use your own paths by changing the necessary values in `configs/path_configs.py`. 

## Training
### Preparing your Data
In order to train ReStyle on your own data, you should perform the following steps: 
1. Update `configs/paths_config.py` with the necessary data paths and model paths for training and inference.
```
dataset_paths = {
    'train_data': '/path/to/train/data'
    'test_data': '/path/to/test/data',
}
```
2. Configure a new dataset under the `DATASETS` variable defined in `configs/data_configs.py`. There, you should define 
the source/target data paths for the train and test sets as well as the transforms to be used for training and inference.
```
DATASETS = {
	'my_data_encode': {
		'transforms': transforms_config.EncodeTransforms,   # can define a custom transform, if desired
		'train_source_root': dataset_paths['train_data'],
		'train_target_root': dataset_paths['train_data'],
		'test_source_root': dataset_paths['test_data'],
		'test_target_root': dataset_paths['test_data'],
	}
}
```
3. To train with your newly defined dataset, simply use the flag `--dataset_type my_data_encode`.

### Preparing your Generator
In this work, we use rosinality's [StyleGAN2 implementation](https://github.com/rosinality/stylegan2-pytorch). 
If you wish to use your own generator trained using NVIDIA's implementation there are a few options we recommend:
1. Using NVIDIA's StyleGAN2 / StyleGAN-ADA TensorFlow implementation.  
   You can then convert the TensorFlow `.pkl` checkpoints to the supported format using the conversion script found in [rosinality's implementation](https://github.com/rosinality/stylegan2-pytorch#convert-weight-from-official-checkpoints).
2. Using NVIDIA's StyleGAN-ADA PyTorch implementation.  
   You can then convert the PyTorch `.pkl` checkpoints to the supported format using the conversion script created by [Justin Pinkney](https://github.com/justinpinkney) found in [dvschultz's fork](https://github.com/dvschultz/stylegan2-ada-pytorch/blob/main/SG2_ADA_PT_to_Rosinality.ipynb).  


Once you have the converted `.pt` files, you should be ready to use them in this repository.  


### Training ReStyle
The main training scripts can be found in `scripts/train_restyle_psp.py` and `scripts/train_restyle_e4e.py`. Each of the 
two scripts will run ReStyle applied over the corresponding base inversion method.  
Intermediate training results are saved to `opts.exp_dir`. This includes checkpoints, train outputs, and test outputs.  
Additionally, if you have tensorboard installed, you can visualize tensorboard logs in `opts.exp_dir/logs`.

We currently support applying ReStyle on the pSp encoder from [Richardson et al. [2020]](https://arxiv.org/abs/2008.00951) 
and the e4e encoder from [Tov et al. [2021]](https://arxiv.org/abs/2102.02766). 

Training ReStyle with the settings used in the paper can be done by running the following commands.

- ReStyle applied over pSp:
```
python scripts/train_restyle_psp.py \
--dataset_type=ffhq_encode \
--encoder_type=BackboneEncoder \
--exp_dir=experiment/restyle_psp_ffhq_encode \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=5000 \
--save_interval=10000 \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--w_norm_lambda=0 \
--id_lambda=0.1 \
--input_nc=6 \
--n_iters_per_batch=5 \
--output_size=1024 \
--stylegan_weights=pretrained_models/stylegan2-ffhq-config-f.pt
```
  
- ReStyle applied over e4e:
```
python scripts/train_restyle_e4e.py \
--dataset_type ffhq_encode \
--encoder_type ProgressiveBackboneEncoder \
--exp_dir=experiment/restyle_e4e_ffhq_encode \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--delta_norm_lambda 0.0002 \
--id_lambda 0.1 \
--use_w_pool \
--w_discriminator_lambda 0.1 \
--progressive_start 20000 \
--progressive_step_every 2000 \
--input_nc 6 \
--n_iters_per_batch=5 \
--output_size 1024 \
--stylegan_weights=pretrained_models/stylegan2-ffhq-config-f.pt
```

#### Additional Notes:
- Encoder backbones:
    - For the human facial domain (`ffhq_encode`), we use an IRSE-50 backbone using the flags:
        - `--encoder_type=BackboneEncoder` for pSp
        - `--encoder_type=ProgressiveBackboneEncoder` for e4e
    - For all other domains, we use a ResNet34 encoder backbone using the flags:
        - `--encoder_type=ResNetBackboneEncoder` for pSp
        - `--encoder_type=ResNetProgressiveBackboneEncoder` for e4e
- ID/similarity losses: 
    - For the human facial domain we also use a specialized ID loss which is set using the flag `--id_lambda=0.1`.
    - For all other domains, please set `--id_lambda=0` and `--moco_lambda=0.5` to use the MoCo-based similarity loss from Tov et al. 
        - Note, you __cannot__ set both `id_lambda` and `moco_lambda` to be active simultaneously.
- You should also adjust the `--output_size` and `--stylegan_weights` flags according to your StyleGAN generator. 
- See `options/train_options.py` and `options/e4e_train_options.py` for all training-specific flags. 

## Inference Notebooks
To help visualize the results of ReStyle we provide a Jupyter notebook found in `notebooks/inference_playground.ipynb`.   
The notebook will download the pretrained models and run inference on the images found in `notebooks/images` or 
on images of your choosing. It is recommended to run this in [Google Colab](http://colab.research.google.com/github/yuval-alaluf/restyle-encoder/blob/master/notebooks/inference_playground.ipynb).

We have also provided a notebook for generating interpolation videos such as those found in the project page. This 
notebook can be run using Google Colab [here](http://colab.research.google.com/github/yuval-alaluf/restyle-encoder/blob/master/notebooks/animations_playground.ipynb).

## Testing
### Inference
You can use `scripts/inference_iterative.py` to apply a trained model on a set of images:
```
python scripts/inference_iterative.py \
--exp_dir=/path/to/experiment \
--checkpoint_path=experiment/checkpoints/best_model.pt \
--data_path=/path/to/test_data \
--test_batch_size=4 \
--test_workers=4 \
--n_iters_per_batch=5
```
This script will save each step's outputs in a separate sub-directory (e.g., the outputs of step `i` will
be saved in `/path/to/experiment/inference_results/i`). 

Notes: 
- By default, the images will be saved at their original output resolutions (e.g., `1024x1024` for faces, `512x384` for cars).
If you wish to save outputs resized to resolutions of `256x256` (or `256x192` for cars), you can do so by adding the flag `--resize_outputs`.
- This script will also save all the latents as an `.npy` file in a dictionary format as follows: 
```
{
    "0.jpg": [latent_step_1, latent_step_2, ..., latent_step_N],
    "1.jpg": [latent_step_1, latent_step_2, ..., latent_step_N],
    ...
}
```
That is, the keys of the dictionary are the image file names and the values are lists of length `N` containing the output latent of each 
step where `N` is the number of inference steps. Each element in the list is of shape `(Kx512)` where `K` is the number
of style inputs of the generator. 

You can use the saved latents to perform latent space manipulations, for example.


### Step-by-Step Inference 
<p align="center">
<img src="docs/2441.jpg" width="800px"/>
<img src="docs/02530.jpg" width="800px"/>
<br>
Visualizing the intermediate outputs. Here, the intermediate outputs are saved from left to right with the input image shown on the right-hand side.
</p>

Sometimes, you may wish to save each step's outputs side-by-side instead of in separate sub-folders. This would allow one 
to easily see the progression in the reconstruction with each step. To save the step-by-step outputs as a single image, 
you can run the following: 
```
python scripts/inference_iterative_save_coupled.py \
--exp_dir=/path/to/experiment \
--checkpoint_path=experiment/checkpoints/best_model.pt \
--data_path=/path/to/test_data \
--test_batch_size=4 \
--test_workers=4 \
--n_iters_per_batch=5
```

### Computing Metrics
Given a trained model and generated outputs, we can compute the loss metrics on a given dataset.  
These scripts receive the inference output directory and ground truth directory.
- Calculating LPIPS loss:
```
python scripts/calc_losses_on_images.py \
--mode lpips \
--output_path=/path/to/experiment/inference_results \
--gt_path=/path/to/test_images
```
- Calculating L2 loss:
```
python scripts/calc_losses_on_images.py \
--mode l2 \
--output_path=/path/to/experiment/inference_results \
--gt_path=/path/to/test_images
```
- Calculating the identity loss for the human facial domain: 
```
python scripts/calc_id_loss_parallel.py \
--output_path=/path/to/experiment/inference_results \
--gt_path=/path/to/test_images
```

These scripts will traverse through each sub-directory of `output_path` to compute the metrics on each step's output images.

## Editing
<p align="center">
<img src="docs/ardern.jpg" width="800px"/>
<img src="docs/macron.jpg" width="800px"/>
<img src="docs/merkel.jpg" width="800px"/>
<br>
Editing results using InterFaceGAN on inversions obtained using ReStyle-e4e.
</p>
For performing edits using ReStyle-e4e, you can run the script found in `editing/inference_editing.py`, as follows:  

```
python editing/inference_editing.py \
--exp_dir=/path/to/experiment \
--checkpoint_path=/path/to/e4e_ffhq_encoder.pt \
--data_path=/path/to/test_data \
--test_batch_size=4 \
--test_workers=4 \
--n_iters_per_batch=5 \
--edit_directions=age,pose,smile \
--factor_ranges=5,5,5
```

This script will perform the inversion immediately followed by the latent space edit.  
The results for each edit will be saved to different sub-directories in the specified experiment directory. For each image, 
we save the original image followed by the inversion and the resulting edits.  
We support running inference using ReStyle-e4e models on the faces domain using edit several directions 
obtained via InterFaceGAN (`age`, `pose`, and `smile`). 


## Encoder Bootstrapping
<p align="center">
<img src="docs/346.jpg" width="800px"/>
<img src="docs/2598.jpg" width="800px"/>
<br>
Image toonification results using our proposed encoder bootstrapping technique.
</p>

In the paper, we introduce an encoder bootstrapping technique that can be used to solve the image toonification task by 
pairing an FFHQ-based encoder with a Toon-based encoder.  
Below we provide the models used to generate the results in the paper: 
| Path | Description
| :--- | :----------
|[FFHQ - ReStyle + pSp](https://drive.google.com/file/d/1sw6I2lRIB0MpuJkpc8F5BJiSZrc0hjfE/view?usp=sharing)  | Same FFHQ encoder as linked above.
|[Toonify - ReStyle + pSp](https://drive.google.com/file/d/1GtudVDig59d4HJ_8bGEniz5huaTSGO_0/view?usp=sharing)  | ReStyle applied over pSp trained for the image toonification task.
|[Toonify Generator](https://drive.google.com/file/d/1r3XVCt_WYUKFZFxhNH-xO2dTtF6B5szu/view?usp=sharing)  | Toonify generator from [Doron Adler](https://github.com/Norod) and [Justin Pinkney](https://github.com/justinpinkney) converted to Pytorch using rosinality's conversion script.

Note that the ReStyle toonify model is trained using only real images with __no__ paired data. 
More details regarding the training parameters and settings of the toonify encoder can be found [here](https://github.com/eladrich/pixel2style2pixel#additional-applications).

If you wish to run inference using these two models and the bootstrapping technique you may run the following: 
```
python scripts/encoder_bootstrapping_inference.py \
--exp_dir=/path/to/experiment \
--model_1_checkpoint_path=/path/to/restyle_psp_ffhq_encode.pt \
--model_2_checkpoint_path=/path/to/restyle_psp_toonify.pt \
--data_path=/path/to/test_data \
--test_batch_size=4 \
--test_workers=4 \
--n_iters_per_batch=1  # one step for each encoder is typically good
```
Here, we output the per-step outputs side-by-side with the inverted initialization real-image on the left and the original 
input image on the right.  



## Repository structure
| Path | Description <img width=200>
| :--- | :---
| restyle-encoder | Repository root folder
| &boxvr;&nbsp; configs | Folder containing configs defining model/data paths and data transforms
| &boxvr;&nbsp; criteria | Folder containing various loss criterias for training
| &boxvr;&nbsp; datasets | Folder with various dataset objects
| &boxvr;&nbsp; docs | Folder containing images displayed in the README
| &boxvr;&nbsp; environment | Folder containing Anaconda environment used in our experiments
| &boxvr;&nbsp; licenses | Folder containing licenses of the open source projects used in this repository
| &boxvr; models | Folder containing all the models and training objects
| &boxv;&nbsp; &boxvr;&nbsp; e4e_modules | Folder containing the latent discriminator implementation from [encoder4editing](https://github.com/omertov/encoder4editing)
| &boxv;&nbsp; &boxvr;&nbsp; encoders | Folder containing various architecture implementations including our simplified encoder architectures
| &boxv;&nbsp; &boxvr;&nbsp; mtcnn | MTCNN implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
| &boxv;&nbsp; &boxvr;&nbsp; stylegan2 | StyleGAN2 model from [rosinality](https://github.com/rosinality/stylegan2-pytorch)
| &boxv;&nbsp; &boxvr;&nbsp; psp.py | Implementation of pSp encoder extended to work with ReStyle
| &boxv;&nbsp; &boxur;&nbsp; e4e.py | Implementation of e4e encoder extended to work with ReStyle
| &boxvr;&nbsp; notebooks | Folder with jupyter notebook containing ReStyle inference playground
| &boxvr;&nbsp; options | Folder with training and test command-line options
| &boxvr;&nbsp; scripts | Folder with running scripts for training, inference, and metric computations
| &boxvr;&nbsp; training | Folder with main training logic and Ranger implementation from [lessw2020](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
| &boxvr;&nbsp; utils | Folder with various utility functions
| <img width=300> | <img>


## Credits
**StyleGAN2 model and implementation:**  
https://github.com/rosinality/stylegan2-pytorch  
Copyright (c) 2019 Kim Seonghyeon  
License (MIT) https://github.com/rosinality/stylegan2-pytorch/blob/master/LICENSE  

**IR-SE50 model and implementations:**  
https://github.com/TreB1eN/InsightFace_Pytorch  
Copyright (c) 2018 TreB1eN  
License (MIT) https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/LICENSE  

**Ranger optimizer implementation:**  
https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer   
License (Apache License 2.0) https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer/blob/master/LICENSE  

**LPIPS model and implementation:**  
https://github.com/S-aiueo32/lpips-pytorch  
Copyright (c) 2020, Sou Uchida  
License (BSD 2-Clause) https://github.com/S-aiueo32/lpips-pytorch/blob/master/LICENSE  

**pSp model and implementation:**   
https://github.com/eladrich/pixel2style2pixel  
Copyright (c) 2020 Elad Richardson, Yuval Alaluf  
License (MIT) https://github.com/eladrich/pixel2style2pixel/blob/master/LICENSE

**e4e model and implementation:**   
https://github.com/omertov/encoder4editing
Copyright (c) 2021 omertov  
License (MIT) https://github.com/omertov/encoder4editing/blob/main/LICENSE

**Please Note**: The CUDA files under the [StyleGAN2 ops directory](https://github.com/eladrich/pixel2style2pixel/tree/master/models/stylegan2/op) are made available under the [Nvidia Source Code License-NC](https://nvlabs.github.io/stylegan2/license.html)

## Acknowledgments
This code borrows heavily from [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel) and 
[encoder4editing](https://github.com/omertov/encoder4editing).

## Citation
If you use this code for your research, please cite the following works:
```
@InProceedings{alaluf2021restyle,
      author = {Alaluf, Yuval and Patashnik, Or and Cohen-Or, Daniel},
      title = {ReStyle: A Residual-Based StyleGAN Encoder via Iterative Refinement}, 
      month = {October},
      booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},  
      year = {2021}
}
```
```
@InProceedings{richardson2021encoding,
      author = {Richardson, Elad and Alaluf, Yuval and Patashnik, Or and Nitzan, Yotam and Azar, Yaniv and Shapiro, Stav and Cohen-Or, Daniel},
      title = {Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation},
      booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month = {June},
      year = {2021}
}
@article{tov2021designing,
      title={Designing an Encoder for StyleGAN Image Manipulation},
      author={Tov, Omer and Alaluf, Yuval and Nitzan, Yotam and Patashnik, Or and Cohen-Or, Daniel},
      journal={arXiv preprint arXiv:2102.02766},
      year={2021}
}
```

