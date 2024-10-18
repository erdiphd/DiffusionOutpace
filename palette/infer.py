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

'''
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        default='/home/faruk/Palette-Image-to-Image-Diffusion-Models/config/inpainting_u_maze_64_goal_oriented_integration_demo.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str,
                        choices=['train', 'test'], help='Run train or test', default='test')
    parser.add_argument('-b', '--batch', type=int,
                        default=16, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)

    args = parser.parse_args()
    opt = Praser.parse(args)
    return opt
'''
'''
def inpaint_img(model, img, id, episode, save):
    regular_mask = bbox2mask((64,64), random_bbox())
    #irregular_mask = brush_stroke_mask((64,64))
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

    # save conditional image used a inference input
    cond_image_np = tensor2img(cond_image)
    #Image.fromarray(cond_image_np).save("./result/cond_image.png")

    # set device
    cond_image = set_device(cond_image)
    gt_image = set_device(img)
    mask = set_device(mask)

    # unsqueeze
    cond_image = cond_image.unsqueeze(0).to(device)
    gt_image = gt_image.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)

    # inference
    with torch.no_grad():
        output, visuals = model.restoration(cond_image, y_t=cond_image,
                                            y_0=gt_image, mask=mask, sample_num=8)

    # save intermediate processes
    output_img = output.detach().float().cpu()
    #for i in range(visuals.shape[0]):
    #    img = tensor2img(visuals[i].detach().float().cpu())
    #    Image.fromarray(img).save(f"./result/process_{i}.png")

    # save output (output should be the same as last process_{i}.png)
    img = tensor2img(output_img)
    if save:
        if not os.path.exists(f"./inpaint_results/episode_{episode}/"):
            os.makedirs(f"./inpaint_results/episode_{episode}/")
        Image.fromarray(img).save(f"./inpaint_results/episode_{episode}/output{id}.png")
    return img

def inpaint_list(model, tf_images, episode, save):
    inpainted_images = []
    counter = 0
    for tf_image in tf_images:
        print(tf_image.shape)
        inpainted_img = inpaint_img(model, tf_image, counter, episode, save)
        counter += 1
        inpainted_images.append(inpainted_img)
    return inpainted_images
'''

#for debugging
def main():
    model_pth = "/home/faruk/outpace_official/palette/experiments/inpainting_u_maze_64_path_oriented/checkpoint/300_Network.pth"
    input_images_pth = "/home/faruk/outpace_official/palette/test_images/" # if you want to test multiple images

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
                                        'n_timestep': 250,
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

    # read input and create random mask
    #img_pillow = Image.open(input_image_pth).convert('RGB')
    #img = tfs(img_pillow)

    tf_images = []
    for file in os.listdir(input_images_pth):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            img_pillow = Image.open(os.path.join(input_images_pth, filename)).convert('RGB')
            img = tfs(img_pillow)
            tf_images.append(img)
    #print(tf_images)

    masks = []
    cond_imgs = []

    for tf_image in tf_images:
        #print("Hello")
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
        # save conditional image used a inference input
        #cond_image_np = tensor2img(cond_image)
        #Image.fromarray(cond_image_np).save("./result/cond_image.png")

    masks = torch.stack(masks, dim = 0)
    cond_imgs = torch.stack(cond_imgs, dim = 0)
    gt_images = torch.stack(tf_images, dim = 0)

    cond_imgs = set_device(cond_imgs)
    gt_images = set_device(gt_images)
    masks = set_device(masks)


    print(masks.shape)
    print(cond_imgs.shape)
    print(gt_images.shape)
        # inference
    with torch.no_grad():
        output, visuals = model.restoration(cond_imgs, y_t=cond_imgs,
                                            y_0=gt_images, mask=masks, sample_num=8)

        # save intermediate processes
        output_imgs = output.detach().float().cpu()
        #for i in range(visuals.shape[0]):
        #    img = tensor2img(visuals[i].detach().float().cpu())
        #    Image.fromarray(img).save(f"./result/process_{i}.png")
    print(output_imgs.shape)
    id = 0
    episode = "test"
    for output_img in output_imgs:
        img = tensor2img(output_img)

        if not os.path.exists(f"./inpaint_results/episode_{episode}/"):
            os.makedirs(f"./inpaint_results/episode_{episode}/")
        Image.fromarray(img).save(f"./inpaint_results/episode_{episode}/output{id}.png")
        id += 1
        # save output (output should be the same as last process_{i}.png)

def single_img(img,save=False,save_path="./inpaint_results/",model_pth = "/home/faruk/Palette-Image-to-Image-Diffusion-Models/experiments/inpainting_u_maze_64_path_oriented/checkpoint/300_Network.pth", beta_schedule_n_steps = 1000):

    # config arg
    #opt = parse_config()
    #model_args = opt["model"]["which_networks"][0]["args"]

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
                                        'n_timestep': beta_schedule_n_steps,
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
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    tf_images = []
    for i in range(8):
        tf_img = tfs(img)
        tf_images.append(tf_img)
    #print(tf_images)
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


    #print(masks.shape)
    #print(cond_imgs.shape)
    #print(gt_images.shape)

    # inference
    with torch.no_grad():
        output, visuals = model.restoration(cond_imgs, y_t=cond_imgs,
                                            y_0=gt_images, mask=masks, sample_num=8)
        output_imgs = output.detach().float().cpu()
    #print(output_imgs.shape)
    
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





