import torch
import numpy as np
from tqdm import tqdm


def edit(generator, latents, indices, semantics=1, start_distance=-15.0, end_distance=15.0, num_samples=1, step=11):

    layers, boundaries, values = factorize_weight(generator, indices)
    codes = latents.detach().cpu().numpy()  # (1,18,512)

    # Generate visualization pages.
    distances = np.linspace(start_distance, end_distance, step)
    num_sam = num_samples
    num_sem = semantics

    edited_latents = []
    for sem_id in tqdm(range(num_sem), desc='Semantic ', leave=False):
        boundary = boundaries[sem_id:sem_id + 1]
        for sam_id in tqdm(range(num_sam), desc='Sample ', leave=False):
            code = codes[sam_id:sam_id + 1]
            for col_id, d in enumerate(distances, start=1):
                temp_code = code.copy()
                temp_code[:, layers, :] += boundary * d
                edited_latents.append(torch.from_numpy(temp_code).float().cuda())
    return torch.cat(edited_latents)


def factorize_weight(g_ema, layers='all'):

    weights = []
    if layers == 'all' or 0 in layers:
        weight = g_ema.conv1.conv.modulation.weight.T
        weights.append(weight.cpu().detach().numpy())

    if layers == 'all':
        layers = list(range(g_ema.num_layers - 1))
    else:
        layers = [l - 1 for l in layers if l != 0]

    for idx in layers:
        weight = g_ema.convs[idx].conv.modulation.weight.T
        weights.append(weight.cpu().detach().numpy())
    weight = np.concatenate(weights, axis=1).astype(np.float32)
    weight = weight / np.linalg.norm(weight, axis=0, keepdims=True)
    eigen_values, eigen_vectors = np.linalg.eig(weight.dot(weight.T))
    return layers, eigen_vectors.T, eigen_values
