import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, get_uniform_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str_, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask(index)
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask
        #print(img)

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self,index):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size, brush_width = (self.image_size[0]//25,self.image_size[0]//15), length_range=(100, 150))
        elif self.mask_mode == 'free_form': #acjusted the lenght range
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            #irregular_mask = brush_stroke_mask(self.image_size)
            irregular_mask = get_irregular_mask(self.image_size, brush_width = (self.image_size[0]//25,self.image_size[0]//15), length_range=(100, 150))
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'goal_oriented':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            #irregular_mask = brush_stroke_mask(self.image_size)
            irregular_mask = get_irregular_mask(self.image_size, brush_width = (self.image_size[0]//25,self.image_size[0]//15), length_range=(100, 150))

            path = self.imgs[index]
            img = self.tfs(self.loader(path))

            black_pixels_mask= np.zeros_like(irregular_mask)
            for i in range(self.image_size[0]):
                for j in range(self.image_size[1]):
                    #print(img[0][i][j])
                    if img[0][i][j] < 0 and img[1][i][j] < 0 and img[2][i][j] < 0:
                        black_pixels_mask[i][j] = 1
                    else:
                        black_pixels_mask[i][j] = 0
            mask = regular_mask | irregular_mask | black_pixels_mask
        elif self.mask_mode == 'path_oriented':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            #irregular_mask = brush_stroke_mask(self.image_size)
            irregular_mask = get_irregular_mask(self.image_size, brush_width = (self.image_size[0]//25,self.image_size[0]//15), length_range=(100, 150))

            path = self.imgs[index]
            img = self.tfs(self.loader(path))

            black_pixels_mask= np.zeros_like(irregular_mask)
            for i in range(self.image_size[0]):
                for j in range(self.image_size[1]):
                    #print(f"R:{img[0][i][j]} G: {img[1][i][j]} B: {img[2][i][j]}")
                    if img[2][i][j] < 1 and img[1][i][j] < 1: #and img[2][i][j] < 0.9: #This is in RGB not BGR
                        black_pixels_mask[i][j] = 1
                    else:
                        black_pixels_mask[i][j] = 0
            mask = regular_mask | irregular_mask | black_pixels_mask
        elif self.mask_mode == 'uniform_vertical_goal_oriented':
            regular_mask = get_uniform_mask(self.image_size, "vertical")

            black_pixels_mask= np.zeros_like(regular_mask)
            path = self.imgs[index]
            img = self.tfs(self.loader(path))
            for i in range(self.image_size[0]):
                for j in range(self.image_size[1]):
                    #print(img[0][i][j])
                    if img[0][i][j] < 0 and img[1][i][j] < 0 and img[2][i][j] < 0:
                        black_pixels_mask[i][j] = 1
                    else:
                        black_pixels_mask[i][j] = 0
            mask = regular_mask | black_pixels_mask
        elif self.mask_mode == 'uniform_horizontal_goal_oriented':
            regular_mask = get_uniform_mask(self.image_size, "horizontal")

            black_pixels_mask= np.zeros_like(regular_mask)
            path = self.imgs[index]
            img = self.tfs(self.loader(path))
            for i in range(self.image_size[0]):
                for j in range(self.image_size[1]):
                    #print(img[0][i][j])
                    if img[0][i][j] < 0 and img[1][i][j] < 0 and img[2][i][j] < 0:
                        black_pixels_mask[i][j] = 1
                    else:
                        black_pixels_mask[i][j] = 0
            mask = regular_mask | black_pixels_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)


