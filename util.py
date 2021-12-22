from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import os
import cv2
import dlib
from PIL import Image
import numpy as np
import math
import torchvision
import scipy
import scipy.ndimage
import torchvision.transforms as transforms

google_drive_paths = {
    "models/stylegan2-ffhq-config-f.pt": "https://drive.google.com/uc?id=1Yr7KuD959btpmcKGAUsbAk5rPjX2MytK",
    "models/dlibshape_predictor_68_face_landmarks.dat": "https://drive.google.com/uc?id=11BDmNKS1zxSZxkgsEvQoKgFd8J264jKp",
    "models/e4e_ffhq_encode.pt": "https://drive.google.com/uc?id=1o6ijA3PkcewZvwJJ73dJ0fxhndn0nnh7",
    "models/restyle_psp_ffhq_encode.pt": "https://drive.google.com/uc?id=1nbxCIVw9H3YnQsoIPykNEFwWJnHVHlVd",
    "models/arcane_caitlyn.pt": "https://drive.google.com/uc?id=1gOsDTiTPcENiFOrhmkkxJcTURykW1dRc",
    "models/arcane_caitlyn_preserve_color.pt": "https://drive.google.com/uc?id=1cUTyjU-q98P75a8THCaO545RTwpVV-aH",
    "models/arcane_jinx_preserve_color.pt": "https://drive.google.com/uc?id=1jElwHxaYPod5Itdy18izJk49K1nl4ney",
    "models/arcane_jinx.pt": "https://drive.google.com/uc?id=1quQ8vPjYpUiXM4k1_KIwP4EccOefPpG_",
    "models/disney.pt": "https://drive.google.com/uc?id=1zbE2upakFUAx8ximYnLofFwfT8MilqJA",
    "models/disney_preserve_color.pt": "https://drive.google.com/uc?id=1Bnh02DjfvN_Wm8c4JdOiNV4q9J7Z_tsi",
    "models/jojo.pt": "https://drive.google.com/uc?id=13cR2xjIBj8Ga5jMO7gtxzIJj2PDsBYK4",
    "models/jojo_preserve_color.pt": "https://drive.google.com/uc?id=1ZRwYLRytCEKi__eT2Zxv1IlV6BGVQ_K2",
    "models/jojo_yasuho.pt": "https://drive.google.com/uc?id=1grZT3Gz1DLzFoJchAmoj3LoM9ew9ROX_",
    "models/jojo_yasuho_preserve_color.pt": "https://drive.google.com/uc?id=1SKBu1h0iRNyeKBnya_3BBmLr4pkPeg_L",
    "models/supergirl.pt": "https://drive.google.com/uc?id=1L0y9IYgzLNzB-33xTpXpecsKU-t9DpVC",
    "models/supergirl_preserve_color.pt": "https://drive.google.com/uc?id=1VmKGuvThWHym7YuayXxjv0fSn32lfDpE",
}

@torch.no_grad()
def load_model(generator, model_file_path):
    ensure_checkpoint_exists(model_file_path)
    ckpt = torch.load(model_file_path, map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"], strict=False)
    return generator.mean_latent(50000)

def ensure_checkpoint_exists(model_weights_filename):
    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename in google_drive_paths
    ):
        gdrive_url = google_drive_paths[model_weights_filename]
        try:
            from gdown import download as drive_download

            drive_download(gdrive_url, model_weights_filename, quiet=False)
        except ModuleNotFoundError:
            print(
                "gdown module not found.",
                "pip3 install gdown or, manually download the checkpoint file:",
                gdrive_url
            )

    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename not in google_drive_paths
    ):
        print(
            model_weights_filename,
            " not found, you may need to manually download the model weights."
        )

# given a list of filenames, load the inverted style code
@torch.no_grad()
def load_source(files, generator, device='cuda'):
    sources = []
    
    for file in files:
        source = torch.load(f'./inversion_codes/{file}.pt')['latent'].to(device)

        if source.size(0) != 1:
            source = source.unsqueeze(0)

        if source.ndim == 3:
            source = generator.get_latent(source, truncation=1, is_latent=True)
            source = list2style(source)
            
        sources.append(source)
        
    sources = torch.cat(sources, 0)
    if type(sources) is not list:
        sources = style2list(sources)
        
    return sources

def display_image(image, size=None, mode='nearest', unnorm=False, title=''):
    # image is [3,h,w] or [1,3,h,w] tensor [0,1]
    if not isinstance(image, torch.Tensor):
        image = transforms.ToTensor()(image).unsqueeze(0)
    if image.is_cuda:
        image = image.cpu()
    if size is not None and image.size(-1) != size:
        image = F.interpolate(image, size=(size,size), mode=mode)
    if image.dim() == 4:
        image = image[0]
    image = image.permute(1, 2, 0).detach().numpy()
    plt.figure()
    plt.title(title)
    plt.axis('off')
    plt.imshow(image)

def get_landmark(filepath, predictor):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()

    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)
    assert len(dets) > 0, "Face not detected, try another face image"

    for k, d in enumerate(dets):
        shape = predictor(img, d)

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    return lm


def align_face(filepath, output_size=1024, transform_size=4096, enable_padding=True):

    """
    :param filepath: str
    :return: PIL Image
    """
    ensure_checkpoint_exists("models/dlibshape_predictor_68_face_landmarks.dat")
    predictor = dlib.shape_predictor("models/dlibshape_predictor_68_face_landmarks.dat")
    lm = get_landmark(filepath, predictor)

    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    img = Image.open(filepath)

    transform_size = output_size
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)

    # Return aligned image.
    return img

def strip_path_extension(path):
   return  os.path.splitext(path)[0]
