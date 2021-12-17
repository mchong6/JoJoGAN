import torch
from utils.common import tensor2im


class LatentEditor(object):

    def __init__(self, stylegan_generator):
        self.generator = stylegan_generator
        self.interfacegan_directions = {
            'age': torch.load('editing/interfacegan_directions/age.pt').cuda(),
            'smile': torch.load('editing/interfacegan_directions/smile.pt').cuda(),
            'pose': torch.load('editing/interfacegan_directions/pose.pt').cuda()
        }

    def apply_interfacegan(self, latents, direction, factor=1, factor_range=None):
        edit_latents = []
        direction = self.interfacegan_directions[direction]
        if factor_range is not None:  # Apply a range of editing factors. for example, (-5, 5)
            for f in range(*factor_range):
                edit_latent = latents + f * direction
                edit_latents.append(edit_latent)
            edit_latents = torch.stack(edit_latents).transpose(0, 1)
        else:
            edit_latents = latents + factor * direction
        return self._latents_to_image(edit_latents)

    def _latents_to_image(self, all_latents):
        sample_results = {}
        with torch.no_grad():
            for idx, sample_latents in enumerate(all_latents):
                images, _ = self.generator([sample_latents], randomize_noise=False, input_is_latent=True)
                sample_results[idx] = [tensor2im(image) for image in images]
        return sample_results
