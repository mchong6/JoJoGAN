# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import os
import tempfile
from copy import deepcopy
from pathlib import Path

import cog
import numpy as np
import torch
from PIL import Image
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

from e4e_projection import projection as e4e_projection
from model import Discriminator, Generator
from util import align_face


class Predictor(cog.Predictor):
    def setup(self):
        pass

    @cog.input("input_face", type=Path, help="Photo of human face")
    @cog.input(
        "pretrained",
        type=str,
        default=None,
        help="Identifier of pretrained style",
        options=[
            "art",
            "arcane_multi",
            "sketch_multi",
            "arcane_jinx",
            "arcane_caitlyn",
            "jojo_yasuho",
            "jojo",
            "disney",
        ],
    )
    @cog.input("style_img_0", default=None, type=Path, help="Face style image (unused if pretrained style is set)")
    @cog.input("style_img_1", default=None, type=Path, help="Face style image (optional)")
    @cog.input("style_img_2", default=None, type=Path, help="Face style image (optional)")
    @cog.input("style_img_3", default=None, type=Path, help="Face style image (optional)")
    @cog.input(
        "preserve_color",
        default=False,
        type=bool,
        help="Preserve the colors of the original image",
    )
    @cog.input(
        "num_iter", default=200, type=int, min=0, help="Number of finetuning steps (unused if pretrained style is set)"
    )
    @cog.input(
        "alpha", default=1, type=float, min=0, max=1, help="Strength of finetuned style"
    )
    def predict(
        self,
        input_face,
        pretrained,
        style_img_0,
        style_img_1,
        style_img_2,
        style_img_3,
        preserve_color,
        num_iter,
        alpha,
    ):

        device = "cuda"  # 'cuda' or 'cpu'

        latent_dim = 512

        # Load original generator
        original_generator = Generator(1024, latent_dim, 8, 2).to(device)
        ckpt = torch.load(
            "models/stylegan2-ffhq-config-f.pt",
            map_location=lambda storage, loc: storage,
        )
        original_generator.load_state_dict(ckpt["g_ema"], strict=False)

        # to be finetuned generator
        generator = deepcopy(original_generator)

        transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # aligns and crops face
        aligned_face = align_face(str(input_face))

        my_w = e4e_projection(aligned_face, "input_face.pt", device).unsqueeze(0)

        if pretrained is not None:
            if (
                preserve_color
                and not (pretrained == "art")
                and not (pretrained == "sketch_multi")
            ):
                ckpt = f"{pretrained}_preserve_color.pt"
            else:
                ckpt = f"{pretrained}.pt"

            ckpt = torch.load(
                os.path.join("models", ckpt), map_location=lambda storage, loc: storage
            )
            generator.load_state_dict(ckpt["g"], strict=False)

            with torch.no_grad():
                generator.eval()
                stylized_face = generator(my_w, input_is_latent=True)

        else:
            # finetune with new style images
            targets = []
            latents = []

            style_imgs = [style_img_0, style_img_1, style_img_2, style_img_3]

            # Remove None values
            style_imgs = [i for i in style_imgs if i]

            for ind, style_img in enumerate(style_imgs):

                # crop and align the face
                style_aligned = align_face(str(style_img))

                out_path = f"style_aligned_{ind}.jpg"
                style_aligned.save(str(out_path))

                # GAN invert
                latent = e4e_projection(style_aligned, f"style_img_{ind}.pt", device)

                targets.append(transform(style_aligned).to(device))
                latents.append(latent.to(device))

            targets = torch.stack(targets, 0)
            latents = torch.stack(latents, 0)

            alpha = 1 - alpha
            # load discriminator for perceptual loss
            discriminator = Discriminator(1024, 2).eval().to(device)
            ckpt = torch.load(
                "models/stylegan2-ffhq-config-f.pt",
                map_location=lambda storage, loc: storage,
            )
            discriminator.load_state_dict(ckpt["d"], strict=False)

            g_optim = optim.Adam(generator.parameters(), lr=2e-3, betas=(0, 0.99))

            # Which layers to swap for generating a family of plausible real images -> fake image
            if preserve_color:
                id_swap = [9, 11, 15, 16, 17]
            else:
                id_swap = list(range(7, generator.n_latent))

            for idx in tqdm(range(num_iter)):
                mean_w = (
                    generator.get_latent(
                        torch.randn([latents.size(0), latent_dim]).to(device)
                    )
                    .unsqueeze(1)
                    .repeat(1, generator.n_latent, 1)
                )
                in_latent = latents.clone()
                in_latent[:, id_swap] = (
                    alpha * latents[:, id_swap] + (1 - alpha) * mean_w[:, id_swap]
                )

                img = generator(in_latent, input_is_latent=True)

                with torch.no_grad():
                    real_feat = discriminator(targets)
                fake_feat = discriminator(img)

                loss = sum(
                    [F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)]
                ) / len(fake_feat)

                g_optim.zero_grad()
                loss.backward()
                g_optim.step()

            with torch.no_grad():
                generator.eval()
                stylized_face = generator(my_w, input_is_latent=True)

        stylized_face = stylized_face.cpu()
        np.save("stylized_face.npy", stylized_face)

        stylized_face = 1 + stylized_face
        stylized_face /= 2

        stylized_face = stylized_face[0]
        stylized_face = 255 * torch.clip(stylized_face, min=0, max=1)
        stylized_face = stylized_face.byte()

        stylized_face = stylized_face.permute(1, 2, 0).detach().numpy()
        stylized_face = Image.fromarray(stylized_face, mode="RGB")
        out_path = Path(tempfile.mkdtemp()) / "out.jpg"
        stylized_face.save(str(out_path))

        return out_path
