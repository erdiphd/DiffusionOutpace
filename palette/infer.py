"use demo_infer for testing"

import argparse

import core.praser as Praser
import torch
from core.util import set_device, tensor2img
from data.util.mask import get_irregular_mask, bbox2mask, random_bbox
from models.network import Network
from PIL import Image
from torchvision import transforms
import numpy as np
import os

#for debugging
def main():
    model_pth = os.path.join(os.path.abspath(os.path.dirname(__file__)),'../weights/which_env?/300_Network.pth')
    input_images_pth = "dir for test images/"
    diffusion_timesteps = 250

    # config arg
    model_args = {'init_type': 'kaiming',
                'module_name': 'guided_diffusion',
                'unet': {'in_channel': 6,
                    'out_channel': 3,
                    'inner_channel': 64,
                    'channel_mults': [1, 2, 4, 8],
                    'attn_res': [8],
                    'num_head_channels': 32,
                    'res_blocks': 2,
                    'dropout': 0.2,
                    'image_size': 64
                    },
                'beta_schedule': {'train': {'schedule': 'linear', 
                                        'n_timestep': 2000, 
                                        'linear_start': 1e-06,
                                        'linear_end': 0.01},
                                'test': {'schedule': 'linear',
                                        'n_timestep': diffusion_timesteps,
                                        'linear_start': 0.0001,
                                        'linear_end': 0.09
                                        }
                                }
                }

    # initializa model
    model = Network(**model_args)
    state_dict = torch.load(model_pth)
    model.load_state_dict(state_dict, strict=False)
    device = torch.device('cuda:0')
    model.to(device)
    model.set_new_noise_schedule(phase='test')
    model.eval()

    tfs = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    #Apply transformations
    tf_images = []
    for file in os.listdir(input_images_pth):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            img_pillow = Image.open(os.path.join(input_images_pth, filename)).convert('RGB')
            img = tfs(img_pillow)
            tf_images.append(img)

    masks = []
    cond_imgs = []

    #Generate masks
    for tf_image in tf_images:
        regular_mask = bbox2mask((64,64), random_bbox())
        irregular_mask = get_irregular_mask((64,64), brush_width = ((64,64)[0]//25,(64,64)[0]//15), length_range=(100, 150))

        black_pixels_mask= np.zeros_like(irregular_mask)
        for i in range((64,64)[0]):
            for j in range((64,64)[1]):
                #print(img[0][i][j])
                if img[0][i][j] < 0.9 or img[1][i][j] < 0.9 or img[2][i][j] < 0.9:
                    black_pixels_mask[i][j] = 1
                else:
                    black_pixels_mask[i][j] = 0
        mask = regular_mask | irregular_mask | black_pixels_mask

        mask = torch.from_numpy(mask).permute(2, 0, 1)
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        masks.append(mask)
        cond_imgs.append(cond_image)

    masks = torch.stack(masks, dim = 0)
    cond_imgs = torch.stack(cond_imgs, dim = 0)
    gt_images = torch.stack(tf_images, dim = 0)

    cond_imgs = set_device(cond_imgs)
    gt_images = set_device(gt_images)
    masks = set_device(masks)

    # inference
    with torch.no_grad():
        output, visuals = model.restoration(cond_imgs, y_t=cond_imgs,
                                            y_0=gt_images, mask=masks, sample_num=8)
        output_imgs = output.detach().float().cpu()

    id = 0
    episode = "debug"
    #Save results
    for output_img in output_imgs:
        img = tensor2img(output_img)
        if not os.path.exists(f"./inpaint_results/episode_{episode}/"):
            os.makedirs(f"./inpaint_results/episode_{episode}/")
        Image.fromarray(img).save(f"./inpaint_results/episode_{episode}/output{id}.png")
        id += 1

def single_img(img,save=False,save_path="./inpaint_results/",model_pth = "os.path.abspath(os.path.dirname(__file__)),'../weights/which_env?/300_Network.pth'",
             diffusion_n_steps = 1000, diffusion_schedule = "linear",
             diffusion_noise_min = 0.0001, diffusion_noise_max = 0.09,
             max_goal_candidates = 8):

    model_args = {'init_type': 'kaiming',
                'module_name': 'guided_diffusion',
                'unet': {'in_channel': 6,
                    'out_channel': 3,
                    'inner_channel': 64,
                    'channel_mults': [1, 2, 4, 8],
                    'attn_res': [8],
                    'num_head_channels': 32,
                    'res_blocks': 2,
                    'dropout': 0.2,
                    'image_size': 64
                    },
                'beta_schedule': {'train': {'schedule': 'linear', 
                                        'n_timestep': 2000, 
                                        'linear_start': 1e-06,
                                        'linear_end': 0.01},
                                'test': {'schedule': diffusion_schedule,
                                        'n_timestep': diffusion_n_steps,
                                        'linear_start': diffusion_noise_min,
                                        'linear_end': diffusion_noise_max
                                        }
                                }
                }

    # initializa model
    model = Network(**model_args)
    state_dict = torch.load(model_pth)
    model.load_state_dict(state_dict, strict=False)
    device = torch.device('cuda:0')
    model.to(device)
    model.set_new_noise_schedule(phase='test')
    model.eval()

    tfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    tf_images = []
    for i in range(max_goal_candidates):
        tf_img = tfs(img)
        tf_images.append(tf_img)
    masks = []
    cond_imgs = []

    for tf_image in tf_images:
        regular_mask = bbox2mask((64,64), random_bbox())
        irregular_mask = get_irregular_mask((64,64), brush_width = ((64,64)[0]//25,(64,64)[0]//15), length_range=(100, 150))

        black_pixels_mask= np.zeros_like(irregular_mask)
        for i in range((64,64)[0]):
            for j in range((64,64)[1]):
                if tf_image[0][i][j] < 0.9 or tf_image[1][i][j] < 0.9 or tf_image[2][i][j] < 0.9:
                    black_pixels_mask[i][j] = 1
                else:
                    black_pixels_mask[i][j] = 0
        mask = regular_mask | irregular_mask | black_pixels_mask

        mask = torch.from_numpy(mask).permute(2, 0, 1)
        cond_image = tf_image*(1. - mask) + mask*torch.randn_like(tf_image)
        mask_img = tf_image*(1. - mask) + mask

        masks.append(mask)
        cond_imgs.append(cond_image)


    masks = torch.stack(masks, dim = 0)
    cond_imgs = torch.stack(cond_imgs, dim = 0)
    gt_images = torch.stack(tf_images, dim = 0)

    cond_imgs = set_device(cond_imgs)
    gt_images = set_device(gt_images)
    masks = set_device(masks)

    # inference
    with torch.no_grad():
        output, visuals = model.restoration(cond_imgs, y_t=cond_imgs,
                                            y_0=gt_images, mask=masks, sample_num=8)
        output_imgs = output.detach().float().cpu()
    
    #Save images
    if save:
        id = 0
        for output_img in output_imgs:
            img = tensor2img(output_img)

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            Image.fromarray(img).save(save_path + f"output{id}.png")
            id += 1

    return output_imgs

if __name__ == "__main__":
    main()





