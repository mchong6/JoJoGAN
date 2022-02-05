# JoJoGAN: One Shot Face Stylization
[![arXiv](https://img.shields.io/badge/arXiv-2112.11641-b31b1b.svg)](https://arxiv.org/abs/2112.11641)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mchong6/JoJoGAN/blob/main/stylize.ipynb)
[![Replicate](https://replicate.com/mchong6/jojogan/badge)](https://replicate.com/mchong6/jojogan)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/JoJoGAN)
[![Wandb Report](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg)](https://wandb.ai/akhaliq/jojogan/reports/JoJoGAN-One-Shot-Face-Stylization-with-Wandb-and-Gradio---VmlldzoxNDMzNzgx)


![](teasers/teaser.jpg)

This is the PyTorch implementation of [JoJoGAN: One Shot Face Stylization](https://arxiv.org/abs/2112.11641).


>**Abstract:**<br>
While there have been recent advances in few-shot image stylization, these methods fail to capture stylistic details
that are obvious to humans. Details such as the shape of the eyes, the boldness of the lines, are especially difficult
for a model to learn, especially so under a limited data setting. In this work, we aim to perform one-shot image stylization that gets the details right. Given
a reference style image, we approximate paired real data using GAN inversion and finetune a pretrained StyleGAN using
that approximate paired data. We then encourage the StyleGAN to generalize so that the learned style can be applied
to all other images.

## Updates

* `2021-12-22` Integrated into [Replicate](https://replicate.com) using [cog](https://github.com/replicate/cog). Try it out [![Replicate](https://replicate.com/mchong6/jojogan/badge)](https://replicate.com/mchong6/jojogan)

* `2022-02-03` Updated the paper. Improved stylization quality using discriminator perceptual loss. Added sketch model
<br><img src="teasers/sketch.gif" width="50%" height="50%"/>
* `2021-12-26` Added wandb logging. Fixed finetuning bug which begins finetuning from previously loaded checkpoint instead of the base face model. Added art model <details><br><img src="teasers/art.gif" width="50%" height="50%"/></details>

* `2021-12-25` Added arcane_multi model which is trained on 4 arcane faces instead of 1 (if anyone has more clean data, let me know!). Better preserves features <details><img src="teasers/arcane.gif" width="50%" height="50%"/></details>

* `2021-12-23` Paper is uploaded to [arxiv](https://arxiv.org/abs/2112.11641).
* `2021-12-22` Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try it out [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/JoJoGAN)
* `2021-12-22` Added pydrive authentication to avoid download limits from gdrive! Fixed running on cpu on colab.



## How to use
Everything to get started is in the [colab notebook](https://colab.research.google.com/github/mchong6/JoJoGAN/blob/main/stylize.ipynb).

## Citation
If you use this code or ideas from our paper, please cite our paper:
```
@article{chong2021jojogan,
  title={JoJoGAN: One Shot Face Stylization},
  author={Chong, Min Jin and Forsyth, David},
  journal={arXiv preprint arXiv:2112.11641},
  year={2021}
}
```

## Acknowledgments
This code borrows from [StyleGAN2 by rosalinity](https://github.com/rosinality/stylegan2-pytorch), [e4e](https://github.com/omertov/encoder4editing). Some snippets of colab code from [StyleGAN-NADA](https://github.com/rinongal/StyleGAN-nada)
