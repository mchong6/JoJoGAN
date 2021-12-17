import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.images_dataset import ImagesDataset
from utils.model_utils import setup_model


class LEC:
    def __init__(self, net, is_cars=False):
        """
        Latent Editing Consistency metric as proposed in the main paper.
        :param net: e4e model loaded over the pSp framework.
        :param is_cars: An indication as to whether or not to crop the middle of the StyleGAN's output images.
        """
        self.net = net
        self.is_cars = is_cars

    def _encode(self, images):
        """
        Encodes the given images into StyleGAN's latent space.
        :param images: Tensor of shape NxCxHxW representing the images to be encoded.
        :return: Tensor of shape NxKx512 representing the latent space embeddings of the given image (in W(K, *) space).
        """
        codes = self.net.encoder(images)
        assert codes.ndim == 3, f"Invalid latent codes shape, should be NxKx512 but is {codes.shape}"
        # normalize with respect to the center of an average face
        if self.net.opts.start_from_latent_avg:
            codes = codes + self.net.latent_avg.repeat(codes.shape[0], 1, 1)
        return codes

    def _generate(self, codes):
        """
        Generate the StyleGAN2 images of the given codes
        :param codes: Tensor of shape NxKx512 representing the StyleGAN's latent codes (in W(K, *) space).
        :return: Tensor of shape  NxCxHxW representing the generated images.
        """
        images, _ = self.net.decoder([codes], input_is_latent=True, randomize_noise=False, return_latents=True)
        images = self.net.face_pool(images)
        if self.is_cars:
            images = images[:, :, 32:224, :]
        return images

    @staticmethod
    def _filter_outliers(arr):
        arr = np.array(arr)

        lo = np.percentile(arr, 1, interpolation="lower")
        hi = np.percentile(arr, 99, interpolation="higher")
        return np.extract(
            np.logical_and(lo <= arr, arr <= hi), arr
        )

    def calculate_metric(self, data_loader, edit_function, inverse_edit_function):
        """
        Calculate the LEC metric score.
        :param data_loader: An iterable that returns a tuple of (images, _), similar to the training data loader.
        :param edit_function: A function that receives latent codes and performs a semantically meaningful edit in the
                              latent space.
        :param inverse_edit_function: A function that receives latent codes and performs the inverse edit of the
                                      `edit_function` parameter.
        :return: The LEC metric score.
        """
        distances = []
        with torch.no_grad():
            for batch in data_loader:
                x, _ = batch
                inputs = x.to(device).float()

                codes = self._encode(inputs)
                edited_codes = edit_function(codes)
                edited_image = self._generate(edited_codes)
                edited_image_inversion_codes = self._encode(edited_image)
                inverse_edit_codes = inverse_edit_function(edited_image_inversion_codes)

                dist = (codes - inverse_edit_codes).norm(2, dim=(1, 2)).mean()
                distances.append(dist.to("cpu").numpy())

        distances = self._filter_outliers(distances)
        return distances.mean()


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="LEC metric calculator")

    parser.add_argument("--batch", type=int, default=8, help="batch size for the models")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Path to the images directory on which we calculate the LEC score")
    parser.add_argument("ckpt", metavar="CHECKPOINT", help="path to the model checkpoints")

    args = parser.parse_args()
    print(args)

    net, opts = setup_model(args.ckpt, device)
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()

    images_directory = dataset_args['test_source_root'] if args.images_dir is None else args.images_dir
    test_dataset = ImagesDataset(source_root=images_directory,
                                 target_root=images_directory,
                                 source_transform=transforms_dict['transform_source'],
                                 target_transform=transforms_dict['transform_test'],
                                 opts=opts)

    data_loader = DataLoader(test_dataset,
                             batch_size=args.batch,
                             shuffle=False,
                             num_workers=2,
                             drop_last=True)

    print(f'dataset length: {len(test_dataset)}')

    # In the following example, we are using an InterfaceGAN based editing to calculate the LEC metric.
    # Change the provided example according to your domain and needs.
    direction = torch.load('../editings/interfacegan_directions/age.pt').to(device)

    def edit_func_example(codes):
        return codes + 3 * direction


    def inverse_edit_func_example(codes):
        return codes - 3 * direction

    lec = LEC(net, is_cars='car' in opts.dataset_type)
    result = lec.calculate_metric(data_loader, edit_func_example, inverse_edit_func_example)
    print(f"LEC: {result}")
