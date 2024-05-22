import torch
import os
import glob
import argparse

import pandas as pd
import numpy as np
import torch.nn as nn

from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from einops import rearrange

import cv2


def get_arg_parser():
    parser = argparse.ArgumentParser('Prepare image and density datasets', add_help=False)

    # Datasets path
    parser.add_argument('--dataset', default='shtech_A')
    parser.add_argument('--data_dir', default='primary_datasets/', type=str,
                        help='Path to the original dataset')
    parser.add_argument('--mode', default='train', type=str,
                        help='Indicate train or test folders')
    
    # Output path
    parser.add_argument('--output_dir', default='datasets/intermediate', type=str,
                        help='Path to save the results')
    
    # Gaussian kernel size and kernel variance
    parser.add_argument('--kernel_size', default='', type=str,
                        help='Size of the Gaussian kernel')
    parser.add_argument('--sigma', default='', type=str,
                        help='Variance of the Gaussian kernel')
    
    # Crop image parameters
    parser.add_argument('--image_size', default=256, type=int,
                        help='Size of the crop images')
    
    # Device parameter
    parser.add_argument('--ndevices', default=4, type=int)

    # Image output
    parser.add_argument('--with_density', action='store_true')

    # count bound
    parser.add_argument('--lower_bound', default=0, type=int)
    parser.add_argument('--upper_bound', default=np.Inf, type=int)

    return parser


def main(args):
    print("Starting preprocessing with arguments:", args)

    # dataset directiors
    data_dir = os.path.join(args.data_dir, args.dataset)
    mode = args.mode

    # output directory
    output_dir = os.path.join(args.output_dir, args.dataset)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    except FileExistsError:
        pass

    # density kernel parameters
    kernel_size_list, sigma_list = get_kernel_and_sigma_list(args)
    print(f"Kernel sizes: {kernel_size_list}")
    print(f"Sigma values: {sigma_list}")
    
    # normalization constants
    normalizer = 0.008

    # crop image parameters
    image_size = args.image_size

    # device parameter
    device = 'cpu'

    # distribution of crowd count
    crowd_bin = [0,0,0,0]

    img_list = sorted(glob.glob(os.path.join(data_dir, mode+'_data', 'images', '*.jpg')))
    print(f"Found {len(img_list)} images to process.")

    sub_list = setup_sub_folders(img_list, output_dir, ndevices=args.ndevices)

    kernel_list = []
    kernel_list = [create_density_kernel(kernel_size_list[index], sigma_list[index]) for index in range(len(sigma_list))]
    normalizer = [kernel.max() for kernel in kernel_list]

    kernel_list = [GaussianKernel(kernel, device) for kernel in kernel_list]

    count = 0

    for device, img_list in enumerate(sub_list):
        for file in img_list:
            count += 1
            if count % 10 == 0:
                print(f"Processing image {count}/{len(img_list)}: {file}")
            # load the images and locations
            image = Image.open(file).convert('RGB')

            file = file.replace('images', 'ground-truth').replace('IMG', 'GT_IMG').replace('jpg', 'mat')
            locations = loadmat(file)['image_info'][0][0]['location'][0][0]

            # resize the image and rescale locations
            if image_size == -1:
                image = np.asarray(image)
            else:
                if mode == 'train' or mode == 'test':
                    image, locations = resize_rescale_info(image, locations, image_size)
                else:
                    image = np.asarray(image)
            
            # create dot map
            density = create_dot_map(locations, image.shape)        
            density = torch.tensor(density)

            density = density.unsqueeze(0).unsqueeze(0)
            density_maps = [kernel(density) for kernel in kernel_list]
            density = torch.stack(density_maps).detach().numpy()
            density = density.transpose(1,2,0)

            # create image crops
            if image_size == -1:
                images, densities = np.expand_dims(image, 0), np.expand_dims(density, 0)
            else:
                if mode == 'train' or mode == 'test':
                    images = create_overlapping_crops(image, image_size, 0.5)
                    densities = create_overlapping_crops(density, image_size, 0.5)
                else:
                    images, densities = create_non_overlapping_crops(image, density, image_size)

            index = os.path.basename(file).split('.')[0].split('_')[-1]

            path = os.path.join(output_dir, f'part_{device+1}', mode)
            den_path = path.replace(os.path.basename(path), os.path.basename(path)+'_den')

            try:
                os.makedirs(path, exist_ok=True)
                os.makedirs(den_path, exist_ok=True)
                print(f"Created directories: {path}, {den_path}")
            except FileExistsError:
                pass
            
            for sub_index, (image, density) in enumerate(zip(images, densities)):
                file = os.path.join(path, str(index)+'-'+str(sub_index+1)+'.jpg')
                print(f"Saving image crop to {file}")
                
                if args.with_density:
                    req_image = [(density[:,:,index]/normalizer[index]*255.).clip(0,255).astype(np.uint8) for index in range(len(normalizer))]
                    req_image = torch.tensor(np.asarray(req_image))
                    req_image = rearrange(req_image, 'c h w -> h (c w)')
                    req_image = req_image.detach().numpy()
                    if len(req_image.shape) < 3:
                        req_image = req_image[:,:,np.newaxis]
                    req_image = np.repeat(req_image, 3, -1)
                    image = np.concatenate([image, req_image], axis=1)
                
                image = np.concatenate(np.split(image, 2, axis=1), axis=0) if args.with_density else image
                Image.fromarray(image, mode='RGB').save(file)
                density = rearrange(torch.tensor(density), 'h w c -> h (c w)').detach().numpy()
                file = os.path.join(den_path, str(index)+'-'+str(sub_index+1)+'.csv')
                print(f"Saving density map to {file}")
                density = pd.DataFrame(density.squeeze())
                density.to_csv(file, header=None, index=False)

    print(f"Total images processed: {count}")
    print(f"Normalization values: {normalizer}")


def get_kernel_and_sigma_list(args):
    kernel_list = [int(item) for item in args.kernel_size.split(' ')]
    sigma_list = [float(item) for item in args.sigma.split(' ')]
    return kernel_list, sigma_list


def get_circle_count(image, normalizer=1, threshold=0, draw=False):
    image = ((image / normalizer).clip(0,1)*255).astype(np.uint8)
    denoisedImg = cv2.fastNlMeansDenoising(image)
    th, threshedImg = cv2.threshold(denoisedImg, threshold, 255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    morphImg = cv2.morphologyEx(threshedImg, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(morphImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if draw:
        contoursImg = cv2.cvtColor(morphImg, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(contoursImg, contours, -1, (255,100,0), 3)
        Image.fromarray(contoursImg, mode='RGB').show()
    return len(contours)-1


def create_dot_map(locations, image_size):
    density = np.zeros(image_size[:-1])
    for x,y in locations:
        x, y = int(x), int(y)
        density[y,x] = 1.
    return density


def create_density_kernel(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    mid_point = kernel_size//2
    kernel[mid_point, mid_point] = 1
    kernel = gaussian_filter(kernel, sigma=sigma)
    return kernel


def resize_rescale_info(image, locations, image_size):
    w,h = image.size
    if h < image_size or w < image_size:
        scale = np.ceil(max(image_size/h, image_size/w))
        h, w = int(scale*h), int(scale*w)
        locations = locations*scale
    image = image.resize((w,h))
    return np.asarray(image), locations


def create_non_overlapping_crops(image, density, image_size):
    h, w = density.shape
    h, w = (h-1+image_size)//image_size, (w-1+image_size)//image_size
    h, w = h*image_size, w*image_size
    pad_density = np.zeros((h,w), dtype=density.dtype)
    pad_image = np.zeros((h,w,image.shape[-1]), dtype=image.dtype)
    start_h = (pad_density.shape[0] - density.shape[0])//2
    end_h = start_h + density.shape[0]
    start_w = (pad_density.shape[1] - density.shape[1])//2
    end_w = start_w + density.shape[1]
    pad_density[start_h:end_h, start_w:end_w] = density
    pad_image[start_h:end_h, start_w:end_w] = image
    pad_density = torch.tensor(pad_density)
    pad_image = torch.tensor(pad_image)
    pad_density = rearrange(pad_density, '(p1 h) (p2 w) -> (p1 p2) h w', h=image_size, w=image_size).numpy()
    pad_image = rearrange(pad_image, '(p1 h) (p2 w) c -> (p1 p2) h w c', h=image_size, w=image_size).numpy()
    return pad_image, pad_density


def create_overlapping_crops(image, crop_size, overlap):
    X_points = start_points(size=image.shape[1], split_size=crop_size, overlap=overlap)
    Y_points = start_points(size=image.shape[0], split_size=crop_size, overlap=overlap)
    image = arrange_crops(image=image, x_start=X_points, y_start=Y_points, crop_size=crop_size)
    return image


def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            if split_size == size:
                break
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


def arrange_crops(image, x_start, y_start, crop_size):
    crops = []
    for i in y_start:
        for j in x_start:
            split = image[i:i+crop_size, j:j+crop_size, :]
            crops.append(split)
    try:
        crops = np.stack(crops)
    except ValueError:
        print(image.shape)
        for crop in crops:
            print(crop.shape)
    return crops


def setup_sub_folders(img_list, output_dir, ndevices=4):
    per_device = len(img_list)//ndevices
    sub_list = []
    for device in range(ndevices-1):
        sub_list.append(img_list[device*per_device:(device+1)*per_device])
    sub_list.append(img_list[(ndevices-1)*per_device:])
    for device in range(ndevices):
        sub_path = os.path.join(output_dir, f'part_{device+1}')
        try:
            os.mkdir(sub_path)
        except FileExistsError:
            pass
    return sub_list


class GaussianKernel(nn.Module):
    def __init__(self, kernel_weights, device):
        super().__init__()
        self.kernel = nn.Conv2d(1,1,kernel_weights.shape, bias=False, padding=kernel_weights.shape[0]//2)
        kernel_weights = torch.tensor(kernel_weights).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            self.kernel.weight = nn.Parameter(kernel_weights)
    def forward(self, density):
        return self.kernel(density).squeeze()


if __name__=='__main__':
    parser = argparse.ArgumentParser('Prepare image and density dataset', parents=[get_arg_parser()])
    args = parser.parse_args()
    main(args)
