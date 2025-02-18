import random
import numpy as np
from pathlib import Path
from einops import rearrange
import random
import torch
import torchvision as thv
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from utils import util_sisr
from utils import util_image
from utils import util_common

from basicsr.data.realesrgan_dataset import RealESRGANDataset
from .ffhq_degradation_dataset import FFHQDegradationDataset
from .masks import MixedMaskGenerator, IrregularNvidiaMask
import os

def get_transforms(transform_type, kwargs):
    '''
    Accepted optins in kwargs.
        mean: scaler or sequence, for nornmalization
        std: scaler or sequence, for nornmalization
        crop_size: int or sequence, random or center cropping
        scale, out_shape: for Bicubic
        min_max: tuple or list with length 2, for cliping
    '''
    if transform_type == 'default':
        transform = thv.transforms.Compose([
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'bicubic_norm':
        transform = thv.transforms.Compose([
            util_sisr.Bicubic(scale=kwargs.get('scale', None), out_shape=kwargs.get('out_shape', None)),
            util_image.Clamper(min_max=kwargs.get('min_max', (0.0, 1.0))),
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'bicubic_back_norm':
        transform = thv.transforms.Compose([
            util_sisr.Bicubic(scale=kwargs.get('scale', None)),
            util_sisr.Bicubic(scale=1/kwargs.get('scale', None)),
            util_image.Clamper(min_max=kwargs.get('min_max', (0.0, 1.0))),
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'aug_crop_norm':
        transform = thv.transforms.Compose([
            util_image.SpatialAug(),
            thv.transforms.ToTensor(),
            thv.transforms.RandomCrop(
                crop_size=kwargs.get('crop_size', None),
                pad_if_needed=True,
                padding_mode='reflect',
                ),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'crop_norm_train':
        transform = thv.transforms.Compose([
            thv.transforms.ToTensor(),
            # thv.transforms.Resize((kwargs.get('img_resize',None),kwargs.get('img_resize',None))),
            thv.transforms.Resize((kwargs.get('img_resize',None))),
            thv.transforms.RandomCrop(
                size=kwargs.get('crop_size', None),
                ),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'crop_norm_val_test':
        transform = thv.transforms.Compose([
            thv.transforms.ToTensor(),
            # thv.transforms.Resize((kwargs.get('img_resize',None),kwargs.get('img_resize',None))),
            thv.transforms.Resize((kwargs.get('img_resize',None))),
            thv.transforms.CenterCrop(
                size=kwargs.get('crop_size', None)
                ),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    else:
        raise ValueError(f'Unexpected transform_variant {transform_variant}')
    return transform

def create_dataset(dataset_config):
    if dataset_config['type'] == 'gfpgan':
        dataset = FFHQDegradationDataset(dataset_config['params'])
    elif dataset_config['type'] == 'bicubic':
        dataset = DatasetBicubic(**dataset_config['params'])
    elif dataset_config['type'] == 'folder':
        dataset = BaseDataFolder(**dataset_config['params'])
    elif dataset_config['type'] == 'realesrgan':
        dataset = RealESRGANDataset(dataset_config['params'])
    elif dataset_config['type'] == 'inpainting':
        dataset = InpaintingDataSet(**dataset_config['params'])
    elif dataset_config['type'] == 'MaskTraining':
        dataset = MaskTrainingDataset(**dataset_config['params'])
    elif dataset_config['type'] == 'MaskTrainingModal':
        dataset = MaskTrainingModalDataset(**dataset_config['params'])
    elif dataset_config['type'] == 'MaskTrainingModal':
        dataset = MaskTrainingModalDataset(**dataset_config['params'])
    elif dataset_config['type'] == 'PriorTraining':
        dataset = MaskTrainingDataset(**dataset_config['params'])
    elif dataset_config['type'] == 'PriorTraining':
        dataset = PriorTrainingDataset(**dataset_config['params'])
    elif dataset_config['type'] == "DiffusionTraining":
        dataset = DiffusionTrainingDataset(**dataset_config['params'])
    else:
        raise NotImplementedError(dataset_config['type'])

    return dataset

class DatasetBicubic(Dataset):
    def __init__(self,
            files_txt=None,
            val_dir=None,
            ext='png',
            sf=None,
            up_back=False,
            need_gt_path=False,
            length=None):
        super().__init__()
        if val_dir is None:
            self.files_names = util_common.readline_txt(files_txt)
        else:
            self.files_names = [str(x) for x in Path(val_dir).glob(f"*.{ext}")]
        self.sf = sf
        self.up_back = up_back
        self.need_gt_path = need_gt_path

        if length is None:
            self.length = len(self.files_names)
        else:
            self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        im_path = self.files_names[index]
        im_gt = util_image.imread(im_path, chn='rgb', dtype='float32')
        im_lq = util_image.imresize_np(im_gt, scale=1/self.sf)
        if self.up_back:
            im_lq = util_image.imresize_np(im_lq, scale=self.sf)

        im_lq = rearrange(im_lq, 'h w c -> c h w')
        im_lq = torch.from_numpy(im_lq).type(torch.float32)

        im_gt = rearrange(im_gt, 'h w c -> c h w')
        im_gt = torch.from_numpy(im_gt).type(torch.float32)

        if self.need_gt_path:
            return {'lq':im_lq, 'gt':im_gt, 'gt_path':im_path}
        else:
            return {'lq':im_lq, 'gt':im_gt}

class BaseDataFolder(Dataset):
    def __init__(
            self,
            dir_path,
            transform_type,
            transform_kwargs=None,
            dir_path_extra=None,
            length=None,
            need_path=False,
            im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
            recursive=False,
            ):
        super(BaseDataFolder, self).__init__()

        file_paths_all = util_common.scan_files_from_folder(dir_path, im_exts, recursive)
        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all

        self.length = length
        self.need_path = need_path
        self.dir_path_extra = dir_path_extra
        self.transform = get_transforms(transform_type, transform_kwargs)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im = util_image.imread(im_path, chn='rgb', dtype='float32')
        im = self.transform(im)
        out_dict = {'image':im, 'lq':im}

        if self.dir_path_extra is not None:
            im_path_extra = Path(self.dir_path_extra) / Path(im_path).name
            im_extra = util_image.imread(im_path_extra, chn='rgb', dtype='float32')
            im_extra = self.transform(im_extra)
            out_dict['gt'] = im_extra

        if self.need_path:
            out_dict['path'] = im_path

        return out_dict

    def reset_dataset(self):
        self.file_paths = random.sample(self.file_paths_all, self.length)

class BaseDataTxt(BaseDataFolder):
    def __init__(
            self,
            txt_path,
            transform_type,
            transform_kwargs=None,
            dir_path_extra=None,
            length=None,
            need_path=False,
            ):

        file_paths_all = util_common.readline_txt(txt_path)
        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all

        self.length = length
        self.need_path = need_path
        self.dir_path_extra = dir_path_extra
        self.transform = get_transforms(transform_type, transform_kwargs)

class InpaintingDataSet(Dataset):
    def __init__(
            self,
            dir_path,
            transform_type,
            transform_kwargs,
            mask_kwargs,
            length=None,
            need_path=False,
            im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
            recursive=False,
            ):
        super().__init__()

        file_paths_all = util_common.scan_files_from_folder(dir_path, im_exts, recursive)
        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all

        self.length = length
        self.need_path = need_path
        self.transform = get_transforms(transform_type, transform_kwargs)
        self.mask_generator = MixedMaskGenerator(**mask_kwargs)
        self.iter_i = 0

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im = util_image.imread(im_path, chn='rgb', dtype='float32')
        im = self.transform(im)        # c x h x w
        out_dict = {'gt':im, }

        mask = self.mask_generator(im, iter_i=self.iter_i)   # c x h x w
        self.iter_i += 1
        im_masked = im *  (1 - mask)
        out_dict['lq'] = im_masked
        out_dict['mask'] = mask

        if self.need_path:
            out_dict['path'] = im_path

        return out_dict

    def reset_dataset(self):
        self.file_paths = random.sample(self.file_paths_all, self.length)


class MaskTrainingDataset(Dataset):
    def __init__(
            self,
            dataset_type,
            dir_path,
            noise_path1,
            noise_path2,
            transform_type,
            transform_kwargs,
            transform_noise_type,
            transform_noise_kwargs,
            mask_kwargs,
            folder_mask_path,
            length=None,
            need_path=False,
            im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
            recursive=False,
            img_size = 256,
            type_prior=None,
            kernel_gaussian_size=3
            ):
        super().__init__()
        file_paths_all=[]
        file_paths_all += util_common.scan_files_from_folder(dir_path, im_exts, recursive)
        print('len_file_path_all',len(file_paths_all))
        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all
        self.dataset_type = dataset_type
        self.type_prior = type_prior
        self.length = length
        self.need_path = need_path
        self.transform = get_transforms(transform_type, transform_kwargs)
        print(self.transform)
        self.transform_noise = get_transforms(transform_noise_type,transform_noise_kwargs)
        print(self.transform_noise)
        self.kernel_gaussian = thv.transforms.GaussianBlur(kernel_size=kernel_gaussian_size)
        self.mask_generator = MixedMaskGenerator(**mask_kwargs)
        # self.mask_generator = IrregularNvidiaMask(folder_mask_path)
        self.iter_i = 0
        self.noise_path1 = []
        self.noise_path2 = []
        self.img_size = img_size
        if not noise_path1 is None and not noise_path2 is None:
            self.noise_path1=util_common.scan_files_from_folder(noise_path1, im_exts, recursive)
            self.noise_path2=util_common.scan_files_from_folder(noise_path2, im_exts, recursive)
        elif not noise_path1 is None:
            self.noise_path1=util_common.scan_files_from_folder(noise_path1, im_exts, recursive)
        elif not noise_path2 is None:
            self.noise_path2=util_common.scan_files_from_folder(noise_path2, im_exts, recursive)
        print(len(self.noise_path1))
        print(len(self.noise_path2))
        self.file_paths_noise=self.noise_path1+self.noise_path2
        print('len_file_paths_noise',len(self.file_paths_noise))

    def __len__(self):
        return len(self.file_paths)

    def sameple_noise(self):
        noise = util_image.imread(self.file_paths_noise[random.randint(0,len(self.file_paths_noise)-1)], chn='rgb', dtype='float32')        
        # print(noise.shape)
        # print(self.transform_noise)
        noise = self.transform_noise(noise)
        return noise 
    
    def get_prior(self):
        pass
    
    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im = util_image.imread(im_path, chn='rgb', dtype='float32')
        im = self.transform(im)        # c x h x w
        out_dict = {'gt':im, }
        self.iter_i+=1
        
        if not (self.type_prior is None):
            if self.type_prior == 'edgeCanny':
                edge_img=util_image.getpriorcanny(im_path,100,200)
                edge_img = torch.tensor(edge_img)
                out_dict['prior']= edge_img
        mask = self.mask_generator(im, iter_i=self.iter_i)   # c x h x w
        self.iter_i += 1
        mask = torch.tensor(mask)
        mask = 1-mask #Convert mask to 1 (keep) and 0 (noise)
        # mask = self.kernel_gaussian(mask)
        mask_reshape=self.kernel_gaussian(mask)
        noise = self.sameple_noise()
        if mask.shape[0] == 1:
            mask_reshape = mask.expand(3, -1, -1)  # Expand along the channel dimension
        mask_reshape = mask_reshape.to(im.device, dtype=im.dtype)
        #Low quality = high quality *(mask_reshape) + (1-mask_reshape)*noise
        im_masked = im *  (mask_reshape) + (1-mask_reshape)*noise
        # print(im.shape, end=' ')
        # print(mask.shape)
        # print(im_masked.shape)
        out_dict['lq'] = im_masked
        out_dict['mask'] = mask
        
        if self.need_path:
            out_dict['path'] = im_path
        # print(out_dict['lq'].shape, out_dict['lq'].shape,  out_dict['mask'] .shape)
        return out_dict

    def reset_dataset(self):
        self.file_paths = random.sample(self.file_paths_all, self.length)

class MaskTrainingModalDataset(Dataset):
    def __init__(
            self,
            dataset_type,
            dir_path,
            noise_path1,
            noise_path2,
            transform_type,
            transform_kwargs,
            transform_noise_type,
            transform_noise_kwargs,
            mask_kwargs,
            folder_mask_path,
            length=None,
            need_path=False,
            im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
            recursive=False,
            img_size = 256,
            type_prior=None,
            kernel_gaussian_size=3
            ):
        super().__init__()
        file_paths_all=[]
        file_paths_all += util_common.scan_files_from_folder(dir_path, im_exts, recursive)
        print('len_file_path_all',len(file_paths_all))
        self.file_paths_all = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.cache_img_gt = {}
        self.cache_img_noise = {}
        self.dataset_type = dataset_type
        self.type_prior = type_prior
        self.length = length
        self.need_path = need_path
        self.transform = get_transforms(transform_type, transform_kwargs)
        self.transform_noise = get_transforms(transform_noise_type,transform_noise_kwargs)
        self.kernel_gaussian = thv.transforms.GaussianBlur(kernel_size=kernel_gaussian_size)
        self.mask_generator = MixedMaskGenerator(**mask_kwargs)
        # self.mask_generator = IrregularNvidiaMask(folder_mask_path)
        self.iter_i = 0
        self.noise_path1 = []
        self.noise_path2 = []
        self.img_size = img_size
        if not self.noise_path1 is None and not self.noise_path2 is None:
            self.noise_path1=util_common.scan_files_from_folder(noise_path1, im_exts, recursive)
            self.noise_path2=util_common.scan_files_from_folder(noise_path2, im_exts, recursive)
        elif not self.noise_path1 is None:
            self.noise_path1=util_common.scan_files_from_folder(noise_path1, im_exts, recursive)
        elif not self.noise_path2 is None:
            self.noise_path2=util_common.scan_files_from_folder(noise_path2, im_exts, recursive)
        print(len(self.noise_path1))
        print(len(self.noise_path2))
        self.file_paths_noise=self.noise_path1+self.noise_path2

        print('len_file_paths_noise',len(self.file_paths_noise))

    def __len__(self):
        return len(self.file_paths_all)

    def sameple_noise(self):
        im_noise_path = self.file_paths_noise[random.randint(0,len(self.file_paths_noise)-1)]  
        if im_noise_path in self.cache_img_noise:
            noise = self.cache_img_noise[im_noise_path]
        else:
            noise = util_image.imread(im_noise_path, chn='rgb', dtype='float32')
            self.cache_img_noise[im_noise_path] = noise

        noise = self.transform_noise(noise)
        return noise 
    
    def get_prior(self):
        pass
    
    def __getitem__(self, index):
        im_path = self.file_paths_all[index]
        if im_path in self.cache_img_gt:
            im = self.cache_img_gt[im_path] 
        else:
            im = util_image.imread(im_path, chn='rgb', dtype='float32')
            self.cache_img_gt[im_path] = im
        
        im = self.transform(im)        # c x h x w
        out_dict = {'gt':im, }
        self.iter_i+=1
        mask = self.mask_generator(im, iter_i=self.iter_i)   # c x h x w
        self.iter_i += 1
        mask = torch.tensor(mask)
        mask = 1-mask #Convert mask to 1 (keep) and 0 (noise)
        # mask = self.kernel_gaussian(mask)
        mask_reshape=self.kernel_gaussian(mask)
        noise = self.sameple_noise()
        if mask.shape[0] == 1:
            mask_reshape = mask.expand(3, -1, -1)  # Expand along the channel dimension
        mask_reshape = mask_reshape.to(im.device, dtype=im.dtype)
        #Low quality = high quality *(mask_reshape) + (1-mask_reshape)*noise
        im_masked = im *  (mask_reshape) + (1-mask_reshape)*noise
        out_dict['lq'] = im_masked
        out_dict['mask'] = mask
        
        if self.need_path:
            out_dict['path'] = im_path
        # print(out_dict['lq'].shape, out_dict['lq'].shape,  out_dict['mask'] .shape)
        return out_dict

    def reset_dataset(self):
        self.file_paths = random.sample(self.file_paths_all, self.length)


class PriorTrainingDataset(Dataset):
    def __init__(
            self,
            dataset_type,
            dir_path,
            noise_path1,
            noise_path2,
            transform_type,
            transform_kwargs,
            transform_noise_type,
            transform_noise_kwargs,
            mask_kwargs,
            folder_mask_path,
            length=None,
            need_path=False,
            im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
            recursive=False,
            img_size = 256,
            type_prior=None,
            kernel_gaussian_size=3
            ):
        super().__init__()
        file_paths_all=[]
        file_paths_all += util_common.scan_files_from_folder(dir_path, im_exts, recursive)
        print('len_file_path_all',len(file_paths_all))
        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all
        self.dataset_type = dataset_type
        self.type_prior = type_prior
        self.length = length
        self.need_path = need_path
        self.transform = get_transforms(transform_type, transform_kwargs)
        self.transform_noise = get_transforms(transform_noise_type,transform_noise_kwargs)
        self.kernel_gaussian = thv.transforms.GaussianBlur(kernel_size=kernel_gaussian_size)
        self.mask_generator = MixedMaskGenerator(**mask_kwargs)
        # self.mask_generator = IrregularNvidiaMask(folder_mask_path)
        self.iter_i = 0
        self.noise_path1 = []
        self.noise_path2 = []
        self.img_size = img_size
        if not self.noise_path1 is None and not self.noise_path2 is None:
            self.noise_path1=util_common.scan_files_from_folder(noise_path1, im_exts, recursive)
            self.noise_path2=util_common.scan_files_from_folder(noise_path2, im_exts, recursive)
        elif not self.noise_path1 is None:
            self.noise_path1=util_common.scan_files_from_folder(noise_path1, im_exts, recursive)
        elif not self.noise_path2 is None:
            self.noise_path2=util_common.scan_files_from_folder(noise_path2, im_exts, recursive)
        print(len(self.noise_path1))
        print(len(self.noise_path2))
        self.file_paths_noise=self.noise_path1+self.noise_path2
        print('len_file_paths_noise',len(self.file_paths_noise))

    def __len__(self):
        return len(self.file_paths)

    def sameple_noise(self):
        noise = util_image.imread(self.file_paths_noise[random.randint(0,len(self.file_paths_noise)-1)], chn='rgb', dtype='float32')        
        # print(noise.shape)
        # print(self.transform_noise)
        noise = self.transform_noise(noise)
        return noise 
    
    def get_prior(self):
        pass
    
    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im = util_image.imread(im_path, chn='rgb', dtype='float32')
        im = self.transform(im)        # c x h x w
        out_dict = {'gt':im, }
        self.iter_i+=1
        
        if not (self.type_prior is None):
            if self.type_prior == 'edgeCanny':
                edge_img=util_image.getpriorcanny(im_path,100,200)
                edge_img = torch.tensor(edge_img)
                out_dict['prior']= edge_img
        mask = self.mask_generator(im, iter_i=self.iter_i)   # c x h x w
        self.iter_i += 1
        mask = torch.tensor(mask)
        mask = 1-mask #Convert mask to 1 (keep) and 0 (noise)
        # mask = self.kernel_gaussian(mask)
        mask_reshape=self.kernel_gaussian(mask)
        noise = self.sameple_noise()
        if mask.shape[0] == 1:
            mask_reshape = mask.expand(3, -1, -1)  # Expand along the channel dimension
        mask_reshape = mask_reshape.to(im.device, dtype=im.dtype)
        #Low quality = high quality *(mask_reshape) + (1-mask_reshape)*noise
        im_masked = im *  (mask_reshape) + (1-mask_reshape)*noise
        out_dict['lq'] = im_masked
        
        if self.need_path:
            out_dict['path'] = im_path
        # print(out_dict['lq'].shape, out_dict['lq'].shape,  out_dict['mask'] .shape)
        return out_dict

    def reset_dataset(self):
        self.file_paths = random.sample(self.file_paths_all, self.length)


class DiffusionTrainingDataset(Dataset):
    def __init__(
            self,
            dataset_type,
            dir_path,
            noise_path1,
            noise_path2,
            transform_type,
            transform_kwargs,
            transform_noise_type,
            transform_noise_kwargs,
            mask_kwargs,
            model_mask_target,
            model_mask_ckpt,
            model_mask_params,
            model_prior_target,
            model_prior_ckpt,
            model_prior_params,
            length=None,
            need_path=False,
            im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
            recursive=False,
            img_size = 256,
            type_prior=None,
            kernel_gaussian_size=3,
            ):
        super().__init__()
        file_paths_all=[]
        file_paths_all += util_common.scan_files_from_folder(dir_path, im_exts, recursive)
        print('len_file_path_all',len(file_paths_all))
        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all
        self.dataset_type = dataset_type
        self.type_prior = type_prior
        self.length = length
        self.need_path = need_path
        self.transform = get_transforms(transform_type, transform_kwargs)
        self.transform_noise = get_transforms(transform_noise_type,transform_noise_kwargs)
        self.kernel_gaussian = thv.transforms.GaussianBlur(kernel_size=kernel_gaussian_size)
        self.mask_generator = MixedMaskGenerator(**mask_kwargs)
        # self.mask_generator = IrregularNvidiaMask(folder_mask_path)
        self.iter_i = 0
        self.noise_path1 = []
        self.noise_path2 = []
        self.img_size = img_size
        if not self.noise_path1 is None and not self.noise_path2 is None:
            self.noise_path1=util_common.scan_files_from_folder(noise_path1, im_exts, recursive)
            self.noise_path2=util_common.scan_files_from_folder(noise_path2, im_exts, recursive)
        elif not self.noise_path1 is None:
            self.noise_path1=util_common.scan_files_from_folder(noise_path1, im_exts, recursive)
        elif not self.noise_path2 is None:
            self.noise_path2=util_common.scan_files_from_folder(noise_path2, im_exts, recursive)
        print(len(self.noise_path1))
        print(len(self.noise_path2))
        self.file_paths_noise=self.noise_path1+self.noise_path2
        print('len_file_paths_noise',len(self.file_paths_noise))
        checkpoint = torch.load('path/to/your_checkpoint.pth', map_location=torch.device('cpu'))
        self.model_initial_mask = util_common.get_obj_from_str(string=model_mask_target)(**model_mask_params)
        self.model_initial_prior = util_common.get_obj_from_str(string=model_mask_target)(**model_mask_params)

    def __len__(self):
        return len(self.file_paths)

    def sameple_noise(self):
        noise = util_image.imread(self.file_paths_noise[random.randint(0,len(self.file_paths_noise)-1)], chn='rgb', dtype='float32')        
        # print(noise.shape)
        # print(self.transform_noise)
        noise = self.transform_noise(noise)
        return noise 
    
    def get_prior(self):
        pass

    def sample_initial_mask(self):
        pass

    def sample_initial_prior(self):
        pass

    
    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im = util_image.imread(im_path, chn='rgb', dtype='float32')
        im = self.transform(im)        # c x h x w
        out_dict = {'gt':im, }
        self.iter_i+=1
        
        if not (self.type_prior is None):
            if self.type_prior == 'edgeCanny':
                edge_img=util_image.getpriorcanny(im_path,100,200)
                edge_img = torch.tensor(edge_img)
                out_dict['prior']= edge_img
        mask = self.mask_generator(im, iter_i=self.iter_i)   # c x h x w
        self.iter_i += 1
        mask = torch.tensor(mask)
        mask = 1-mask #Convert mask to 1 (keep) and 0 (noise)
        # mask = self.kernel_gaussian(mask)
        mask_reshape=self.kernel_gaussian(mask)
        noise = self.sameple_noise()
        if mask.shape[0] == 1:
            mask_reshape = mask.expand(3, -1, -1)  # Expand along the channel dimension
        mask_reshape = mask_reshape.to(im.device, dtype=im.dtype)
        #Low quality = high quality *(mask_reshape) + (1-mask_reshape)*noise
        im_masked = im *  (mask_reshape) + (1-mask_reshape)*noise
        out_dict['lq'] = im_masked
        
        if self.need_path:
            out_dict['path'] = im_path
        # print(out_dict['lq'].shape, out_dict['lq'].shape,  out_dict['mask'] .shape)
        return out_dict

    def reset_dataset(self):
        self.file_paths = random.sample(self.file_paths_all, self.length)


# class DiffusionTrainingDataset(Dataset):
#     def __init__(
#             self,
#             dir_path,
#             noise_path1,
#             noise_path2,
#             mask_path,
#             initial_mask_path,
#             initial_prior_path,
#             transform_type,
#             transform_kwargs,
#             mask_kwargs,
#             length=None,
#             need_path=False,
#             im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
#             recursive=False,
#             recursive_noise1=False,
#             recursive_noise2=False,
#             type_prior=None
#             ):
#         super().__init__()

#         file_paths_all = util_common.scan_files_from_folder(dir_path, im_exts, recursive)
#         print('len_file_path_all',len(file_paths_all))
#         self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
#         self.file_paths_all = file_paths_all

#         self.initial_mask_path = initial_mask_path,
#         self.initial_prior_path = initial_prior_path,
#         self.prior_initial_list = []
#         self.prior_groundtruth_list = []
#         for original_path in self.file_paths_all:
#             dir_name, file_name = os.path.split(original_path)
#             file_root, file_ext = os.path.splitext(file_name)
#             file_name_initial = file_root + "_initial" + file_ext
#             initial_prior_path = os.path.join(self.initial_prior_path, file_name_initial)
#             self.prior_initial_list.append(initial_prior_path)

#         self.type_prior = type_prior
#         self.length = length
#         self.need_path = need_path
#         self.transform = get_transforms(transform_type, transform_kwargs)
#         self.iter_i = 0
#         self.noise_path1 = []
#         self.noise_path2 = []

#         if not self.noise_path1 is None and not self.noise_path2 is None:
#             self.noise_path1=util_common.scan_files_from_folder(noise_path1, im_exts, recursive_noise1)
#             self.noise_path2=util_common.scan_files_from_folder(noise_path2, im_exts, recursive_noise2)
#         elif not self.noise_path1 is None:
#             self.noise_path1=util_common.scan_files_from_folder(noise_path1, im_exts, recursive_noise1)
#         elif not self.noise_path2 is None:
#             self.noise_path2=util_common.scan_files_from_folder(noise_path2, im_exts, recursive_noise2)
#         print(len(self.noise_path1))
#         print(len(self.noise_path2))
#         self.file_paths_noise=self.noise_path1+self.noise_path2
#         print('len_file_paths_noise',len(self.file_paths_noise))
#         self.transform_noise_edge = thv.transforms.Compose([
#                 thv.transforms.Resize((256, 256)), 
#             ])

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, index):
#         im_path = self.file_paths[index]
#         noise_path = self.file_paths_noise[random.randint(0,len(self.file_paths_noise)-1)]
#         im = util_image.imread(im_path, chn='rgb', dtype='float32')
#         im = self.transform(im)        # c x h x w
#         out_dict = {'gt':im, }

#         noise = util_image.imread(noise_path, chn='rgb', dtype='float32')
#         noise = self.transform(noise)
#         noise = self.transform_noise_edge(noise) 

#         initial_mask = util_image.imread(,chn='gray',dtype='float32')
#         initial_mask=initial_mask/255
#         initial_mask = initial_mask.reshape(1,256,256)
#         initial_mask=initial_mask.astype(np.float32)
#         inital_mask = torch.tensor(inital_mask)
#         out_dict['initial_mask'] = inital_mask




#         if not (self.type_prior is None):
#             if self.type_prior == 'edgeCanny':
#                 edge_img=util_image.getpriorcanny(im_path,100,200)
#                 edge_img = torch.tensor(edge_img)
#                 out_dict['prior']= edge_img

#                 initial_prior = util_image.imread(,chn='gray',dtype='float32')
#                 initial_mask =torch.tensor(initial_prior)
#                 out_dict['initial_prior'] = initial_prior
#             elif self.type_prior == 'edgeModel':
#                 initial_prior = util_image.imread(,chn='gray',dtype='float32')
#                 initial_mask =torch.tensor(initial_prior)
#                 out_dict['initial_prior'] = initial_prior

 
#         mask = torch.tensor(mask)
#         mask_reshape=mask
#         if mask.shape[0] == 1:
#             mask_reshape = mask.expand(3, -1, -1)  # Expand along the channel dimension
#         mask_reshape = mask_reshape.to(im.device, dtype=im.dtype)
        
#         im_masked = im *  (1 - mask_reshape) +mask_reshape*noise
#         out_dict['lq'] = im_masked
#         out_dict['mask'] = mask

        
#         if self.need_path:
#             out_dict['path'] = im_path
#         # print(out_dict['lq'].shape, out_dict['lq'].shape,  out_dict['mask'] .shape)
#         return out_dict

#     def reset_dataset(self):
#         self.file_paths = random.sample(self.file_paths_all, self.length)
