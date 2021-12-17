import tempfile
from argparse import Namespace
from pathlib import Path
import numpy as np
import time
import cog
import dlib
import imageio
import torch
from PIL import Image
from torchvision import transforms

from models.e4e import e4e
from models.psp import pSp
from scripts.align_faces_parallel import align_face
from scripts import encoder_bootstrapping_inference
from utils.common import tensor2im
from utils.inference_utils import run_on_batch

DOMAINS = ["faces", "toonify"]


class Predictor(cog.Predictor):

    def setup(self):
        print("Starting setup!")
        self.model_paths = {
            "faces": "pretrained_models/restyle_psp_ffhq_encode.pt",
            "toonify": "pretrained_models/restyle_psp_toonify.pt"
        }
        print("Loading checkpoints...")
        self.checkpoints = {
            "faces": torch.load(self.model_paths["faces"], map_location="cpu"),
            "toonify": torch.load(self.model_paths["toonify"], map_location="cpu")
        }
        print("Done!")
        self.shape_predictor = dlib.shape_predictor("/content/shape_predictor_68_face_landmarks.dat")
        self.default_transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.cars_transforms = transforms.Compose([
                transforms.Resize((192, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        print("Setup complete!")

    @cog.input("input", type=Path, help="Path to input image")
    @cog.input("encoding_type",
               type=str,
               help=f"Which domain you wish to run on. Options are: {DOMAINS}",
               options=DOMAINS)
    @cog.input("num_iterations",
               type=int,
               default=5,
               min=1,
               max=10,
               help="Number of ReStyle iterations to run. "
                    "For `faces` we recommend 5 iterations and for `toonify` we recommend 1 to 2 iterations.")
    @cog.input("display_intermediate_results",
               type=bool,
               default=False,
               help="Whether to display all intermediate outputs. If unchecked, will display only the final result.")
    def predict(self, input, encoding_type, num_iterations, display_intermediate_results):
        if encoding_type == "toonify":
            return self.run_toonify_bootstrapping(input, num_iterations, display_intermediate_results)
        else:
            return self.run_default_encoding(input, encoding_type, num_iterations, display_intermediate_results)

    def run_default_encoding(self, input, encoding_type, num_iterations, display_intermediate_results):
        # load model
        print(f'Loading {encoding_type} model...')
        ckpt = self.checkpoints[encoding_type]
        opts = ckpt['opts']
        opts['checkpoint_path'] = self.model_paths[encoding_type]
        opts = Namespace(**opts)
        net = e4e(opts) if encoding_type == "horses" else pSp(opts)
        net.eval()
        net.cuda()
        print('Done!')

        # define some arguments
        opts.n_iters_per_batch = num_iterations
        opts.resize_outputs = False

        # define transforms
        image_transforms = self.cars_transforms if encoding_type == "cars" else self.default_transforms

        # if working on faces load and align the image
        if encoding_type == "faces":
            print('Aligning image...')
            input_image = self.run_alignment(str(input))
            print('Done!')
        # otherwise simply load the image
        else:
            input_image = Image.open(str(input)).convert("RGB")

        # preprocess image
        transformed_image = image_transforms(input_image)

        # run inference
        print("Running inference...")
        with torch.no_grad():
            start = time.time()
            avg_image = self.get_avg_image(net, encoding_type)
            result_batch, result_latents = run_on_batch(transformed_image.unsqueeze(0).cuda(), net, opts, avg_image)
            total_time = time.time() - start
        print(f"Finished inference in {total_time} seconds!")

        # post-processing
        print("Preparing result...")
        resize_amount = (512, 384) if encoding_type == "cars_encode" else (opts.output_size, opts.output_size)
        res = self.get_final_output(result_batch,
                                    resize_amount,
                                    display_intermediate_results,
                                    opts)

        # display output
        out_path = Path(tempfile.mkdtemp()) / "output.png"
        imageio.imwrite(str(out_path), res)
        return out_path

    def run_toonify_bootstrapping(self, input, num_iterations, display_intermediate_results):
        # load ffhq model
        print("Loading faces model...")
        ckpt = self.checkpoints["faces"]
        opts = ckpt['opts']
        opts['checkpoint_path'] = self.model_paths["faces"]
        opts = Namespace(**opts)
        net_ffhq = pSp(opts)
        net_ffhq.eval()
        net_ffhq.cuda()
        print("Done!")

        # load toonify model
        print("Loading toonify model...")
        ckpt = self.checkpoints["toonify"]
        opts = ckpt['opts']
        opts['checkpoint_path'] = self.model_paths["toonify"]
        opts = Namespace(**opts)
        net_toonify = pSp(opts)
        net_toonify.eval()
        net_toonify.cuda()
        print("Done!")

        # define some arguments
        opts.n_iters_per_batch = num_iterations
        opts.resize_outputs = False

        # load, align, and preprocess image
        print('Aligning image...')
        input_image = self.run_alignment(str(input))
        print('Done!')
        transformed_image = self.default_transforms(input_image)

        # run inference
        print("Running inference...")
        with torch.no_grad():
            start = time.time()
            avg_image = self.get_avg_image(net_ffhq, encoding_type="faces")
            result_batch = encoder_bootstrapping_inference.run_on_batch(transformed_image.unsqueeze(0).cuda(),
                                                                        net_ffhq,
                                                                        net_toonify,
                                                                        opts,
                                                                        avg_image)
            total_time = time.time() - start
        print(f"Finished inference in {total_time} seconds!")

        # post-processing
        print("Preparing result...")
        resize_amount = (1024, 1024)
        res = self.get_final_output(result_batch,
                                    resize_amount,
                                    display_intermediate_results,
                                    opts)

        # display output
        out_path = Path(tempfile.mkdtemp()) / "output.png"
        imageio.imwrite(str(out_path), res)
        return out_path

    def run_alignment(self, image_path):
        try:
            aligned_image = align_face(filepath=image_path, predictor=self.shape_predictor)
        except Exception:
            raise ValueError(f"Oh no! Could not align face! \nPlease try another image!")
        return aligned_image

    @staticmethod
    def get_avg_image(net, encoding_type):
        avg_image = net(net.latent_avg.unsqueeze(0),
                        input_code=True,
                        randomize_noise=False,
                        return_latents=False,
                        average_code=True)[0]
        avg_image = avg_image.to('cuda').float().detach()
        if encoding_type == "cars":
            avg_image = avg_image[:, 32:224, :]
        return avg_image

    @staticmethod
    def get_final_output(result_batch, resize_amount, display_intermediate_results, opts):
        result_tensors = result_batch[0]  # there's one image in our batch
        if display_intermediate_results:
            result_images = [tensor2im(result_tensors[iter_idx]) for iter_idx in range(opts.n_iters_per_batch)]
        else:
            result_images = [tensor2im(result_tensors[-1])]
        res = np.array(result_images[0].resize(resize_amount))
        for idx, result in enumerate(result_images[1:]):
            res = np.concatenate([res, np.array(result.resize(resize_amount))], axis=1)
        res = Image.fromarray(res)
        return res
