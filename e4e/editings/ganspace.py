import torch


def edit(latents, pca, edit_directions):
    edit_latents = []
    for latent in latents:
        for pca_idx, start, end, strength in edit_directions:
            delta = get_delta(pca, latent, pca_idx, strength)
            delta_padded = torch.zeros(latent.shape).to('cuda')
            delta_padded[start:end] += delta.repeat(end - start, 1)
            edit_latents.append(latent + delta_padded)
    return torch.stack(edit_latents)


def get_delta(pca, latent, idx, strength):
    # pca: ganspace checkpoint. latent: (16, 512) w+
    w_centered = latent - pca['mean'].to('cuda')
    lat_comp = pca['comp'].to('cuda')
    lat_std = pca['std'].to('cuda')
    w_coord = torch.sum(w_centered[0].reshape(-1)*lat_comp[idx].reshape(-1)) / lat_std[idx]
    delta = (strength - w_coord)*lat_comp[idx]*lat_std[idx]
    return delta
