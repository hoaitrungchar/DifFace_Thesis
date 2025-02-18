#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-05-18 13:04:06

import os
import sys
import math
import time
import lpips
import random
import copy
import datetime
import functools
import numpy as np
from pathlib import Path
from loguru import logger
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
from einops import rearrange
import wandb
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.utils.data as udata
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import util_net
from utils import util_common
from utils import util_image

from basicsr.utils import DiffJPEG
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from datapipe.datasets import create_dataset
from models.resample import UniformSampler
class TrainerBase:
    def __init__(self, configs):
        self.configs = configs
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        if self.configs.wandb_id is None:
            wandb.init(project=configs.project_name, group= configs.group_name, name=configs.name)
        else:
            wandb.init(project=configs.project_name,id = self.configs.wandb_id, resume = "must")
        # setup distributed training: self.num_gpus, self.rank
        self.setup_dist()

        # setup seed
        self.setup_seed()

        # setup logger: self.logger
        self.init_logger()

        # logging the configurations
        if self.rank == 0: self.logger.info(OmegaConf.to_yaml(self.configs))

        # build model: self.model, self.loss
        self.build_model()

        # setup optimization: self.optimzer, self.sheduler
        self.setup_optimizaton()

        # resume
        self.resume_from_ckpt()
        self.epoch = 0 


    def setup_dist(self):
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    )

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def setup_seed(self, seed=None):
        seed = self.configs.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def init_logger(self):
        if self.configs.resume:
            assert self.configs.resume.endswith(".pth")
            save_dir = Path(self.configs.resume).parents[1]
            project_id = save_dir.name
        else:
            project_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            save_dir = Path(self.configs.save_dir) / project_id
            if not save_dir.exists() and self.rank == 0:
                save_dir.mkdir(parents=True)

        # setting log counter
        if self.rank == 0:
            self.log_step = {phase: 1 for phase in ['train', 'val']}
            self.log_step_img = {phase: 1 for phase in ['train', 'val']}

        # text logging
        logtxet_path = save_dir / 'training.log'
        if self.rank == 0:
            if logtxet_path.exists():
                assert self.configs.resume
            self.logger = logger
            self.logger.remove()
            self.logger.add(logtxet_path, format="{message}", mode='a', level='INFO')
            self.logger.add(sys.stdout, format="{message}")

        # tensorboard logging
        log_dir = save_dir / 'tf_logs'
        self.tf_logging = self.configs.train.tf_logging
        if self.rank == 0 and self.tf_logging:
            if not log_dir.exists():
                log_dir.mkdir()
            self.writer = SummaryWriter(str(log_dir))

        # checkpoint saving
        ckpt_dir = save_dir / 'ckpts'
        self.ckpt_dir = ckpt_dir
        if self.rank == 0 and (not ckpt_dir.exists()):
            ckpt_dir.mkdir()
        if 'ema_rate' in self.configs.train:
            self.ema_rate = self.configs.train.ema_rate
            assert isinstance(self.ema_rate, float), "Ema rate must be a float number"
            ema_ckpt_dir = save_dir / 'ema_ckpts'
            self.ema_ckpt_dir = ema_ckpt_dir
            if self.rank == 0 and (not ema_ckpt_dir.exists()):
                ema_ckpt_dir.mkdir()

        # save images into local disk
        self.local_logging = self.configs.train.local_logging
        if self.rank == 0 and self.local_logging:
            image_dir = save_dir / 'images'
            if not image_dir.exists():
                (image_dir / 'train').mkdir(parents=True)
                (image_dir / 'val').mkdir(parents=True)
            self.image_dir = image_dir

        # logging the configurations
        if self.rank == 0:
            self.logger.info(OmegaConf.to_yaml(self.configs))

    def close_logger(self):
        if self.rank == 0 and self.tf_logging:
            self.writer.close()

    def resume_from_ckpt(self):
        def _load_ema_state(ema_state, ckpt):
            for key in ema_state.keys():
                if key not in ckpt and key.startswith('module'):
                    ema_state[key] = deepcopy(ckpt[7:].detach().data)
                elif key not in ckpt and (not key.startswith('module')):
                    ema_state[key] = deepcopy(ckpt['module.'+key].detach().data)
                else:
                    ema_state[key] = deepcopy(ckpt[key].detach().data)
        if self.configs.pretrained:
            ckpt_path = self.configs.pretrained
            assert os.path.isfile(self.configs.pretrained)
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            util_net.reload_model(self.model,ckpt)
        if self.configs.resume:
            if type(self.configs.resume) == bool:
                ckpt_index = max([int(x.stem.split('_')[1]) for x in Path(self.ckpt_dir).glob('*.pth')])
                ckpt_path = str(Path(self.ckpt_dir) / f"model_{ckpt_index}.pth")
            else:
                ckpt_path = self.configs.resume
            assert os.path.isfile(ckpt_path)
            if self.rank == 0:
                self.logger.info(f"=> Loading checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            util_net.reload_model(self.model, ckpt['state_dict'])

            # EMA model
            if self.rank == 0 and hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / ("ema_"+Path(ckpt_path).name)
                self.logger.info(f"=> Loading EMA checkpoint from {str(ema_ckpt_path)}")
                ema_ckpt = torch.load(ema_ckpt_path, map_location=f"cuda:{self.rank}")
                _load_ema_state(self.ema_state, ema_ckpt)

            torch.cuda.empty_cache()

            # starting iterations
            self.iters_start = ckpt['iters_start']

            # learning rate scheduler
            for ii in range(self.iters_start):
                self.adjust_lr(ii)

            # logging counter
            if self.rank == 0:
                self.log_step = ckpt['log_step']
                self.log_step_img = ckpt['log_step_img']

            # reset the seed
            self.setup_seed(self.iters_start)
        else:
            self.iters_start = 0

    def setup_optimizaton(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.configs.train.lr,
                                           weight_decay=self.configs.train.weight_decay)

    def build_model(self):
        params = self.configs.model.get('params', dict)
        model = util_common.get_obj_from_str(self.configs.model.target)(**params)
        if self.num_gpus > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.model = DDP(model.cuda(), device_ids=[self.rank,])  # wrap the network
        else:
            self.model = model.cuda()
        if hasattr(self.configs.model, 'ckpt_path') and self.configs.model.ckpt_path is not None:
            ckpt_path = self.configs.model.ckpt_path
            if self.rank == 0:
                self.logger.info(f"Initializing model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            util_net.reload_model(self.model, ckpt)

        # EMA
        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_model = deepcopy(model).cuda()
            self.ema_state = OrderedDict(
                {key:deepcopy(value.data) for key, value in self.model.state_dict().items()}
                )
            self.ema_ignore_keys = [x for x in self.ema_state.keys() if ('running_' in x or 'num_batches_tracked' in x)]

        # model information
        self.print_model_info()

    def build_dataloader(self):
        def _wrap_loader(loader):
            while True: yield from loader

        datasets = {}
        phases = ['train', ]
        if 'val' in self.configs.data:
            phases.append('val')

        if 'test' in self.configs.data:
            phases.append('test')

        for current_phase in phases:
            dataset_config = self.configs.data.get(current_phase, dict)
            datasets[current_phase] = create_dataset(dataset_config)

        dataloaders = {}
        # train dataloader
        if self.rank == 0:
            for current_phase in phases:
                length = len(datasets[current_phase])
                self.logger.info('Number of images in {:s} data set: {:d}'.format(current_phase, length))
        if self.num_gpus > 1:
            shuffle = False
            sampler = udata.distributed.DistributedSampler(datasets['train'],
                                                           num_replicas=self.num_gpus,
                                                           rank=self.rank)
        else:
            shuffle = True
            sampler = None
        dataloaders['train'] = _wrap_loader(udata.DataLoader(
                                    datasets['train'],
                                    batch_size=self.configs.train.batch[0] // self.num_gpus,
                                    shuffle=shuffle,
                                    drop_last=False,
                                    num_workers=self.configs.train.num_workers,
                                    pin_memory=True,
                                    prefetch_factor=self.configs.train.prefetch_factor,
                                    worker_init_fn=my_worker_init_fn,
                                    sampler=sampler))
        if 'val' in phases and self.rank == 0:
            dataloaders['val'] = udata.DataLoader(
                    datasets['val'],
                    batch_size=self.configs.train.batch[1],
                    shuffle=False,
                    drop_last=False,
                    num_workers=self.configs.train.num_workers,
                    pin_memory=True,
                    )

        self.datasets = datasets
        self.dataloaders = dataloaders
        self.sampler = sampler

    def print_model_info(self):
        if self.rank == 0:
            num_params = util_net.calculate_parameters(self.model) / 1000**2
            # self.logger.info("Detailed network architecture:")
            # self.logger.info(self.model.__repr__())
            self.logger.info(f"Number of parameters: {num_params:.2f}M")

    def prepare_data(self, data, phase='train'):
        return {key:value.cuda() for key, value in data.items()}

    def validation(self):
        pass

    def train(self):
        self.build_dataloader() # prepare data: self.dataloaders, self.datasets, self.sampler

        self.model.train()
        num_iters_epoch = math.ceil(len(self.datasets['train']) / self.configs.train.batch[0])
        for ii in range(self.iters_start, self.configs.train.iterations):
            self.current_iters = ii + 1

            # prepare data
            data = self.prepare_data(next(self.dataloaders['train']), phase='train')

            # training phase
            self.training_step(data)

            # validation phase
            if (ii+1) % self.configs.train.val_freq == 0 and 'val' in self.dataloaders and self.rank==0:
                self.validation()

            #update learning rate
            self.adjust_lr()

            # save checkpoint
            if (ii+1) % self.configs.train.save_freq == 0 and self.rank == 0:
                self.save_ckpt()

            if (ii+1) % num_iters_epoch == 0 and not self.sampler is None:
                self.sampler.set_epoch(ii+1)

        # close the tensorboard
        if self.rank == 0:
            self.close_logger()
        wandb.finish()
    def training_step(self, data):
        pass

    def adjust_lr(self, current_iters=None):
        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()

    def save_ckpt(self):
        if self.rank == 0:
            ckpt_path = self.ckpt_dir / 'model_{:d}.pth'.format(self.current_iters)
        
            torch.save({'iters_start': self.current_iters,
                        'log_step': {phase:self.log_step[phase] for phase in ['train', 'val']},
                        'log_step_img': {phase:self.log_step_img[phase] for phase in ['train', 'val']},
                        'state_dict': self.model.state_dict()}, ckpt_path)
            wandb.save(ckpt_path,base_path=self.ckpt_dir)
            if hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / 'ema_model_{:d}.pth'.format(self.current_iters)
                torch.save(self.ema_state, ema_ckpt_path)
                wandb.save(ema_ckpt_path,base_path= self.ema_ckpt_dir)

    def logging_image(self, im_tensor, tag, phase, add_global_step=False, nrow=8):
        """
        Args:
            im_tensor: b x c x h x w tensor
            im_tag: str
            phase: 'train' or 'val'
            nrow: number of displays in each row
        """
        assert self.tf_logging or self.local_logging
        im_tensor = vutils.make_grid(im_tensor, nrow=nrow, normalize=True, scale_each=True) # c x H x W
        if self.local_logging:
            im_path = str(self.image_dir / phase / f"{tag}-{self.log_step_img[phase]}.png")
            im_np = im_tensor.cpu().permute(1,2,0).numpy()
            util_image.imwrite(im_np, im_path)
        if self.tf_logging:
            self.writer.add_image(
                    f"{phase}-{tag}-{self.log_step_img[phase]}",
                    im_tensor,
                    self.log_step_img[phase],
                    )
        if add_global_step:
            self.log_step_img[phase] += 1

    def logging_metric(self, metrics, tag, phase, add_global_step=False):
        """
        Args:
            metrics: dict
            tag: str
            phase: 'train' or 'val' or 'test'
        """
        if self.tf_logging:
            tag = f"{phase}-{tag}"
            if isinstance(metrics, dict):
                self.writer.add_scalars(tag, metrics, self.log_step[phase])
            else:
                self.writer.add_scalar(tag, metrics, self.log_step[phase])
            if add_global_step:
                self.log_step[phase] += 1
        else:
            pass

    def calculate_L2_gradient(self):
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                # Compute the L2 norm of this parameter's gradient
                param_norm = param.grad.data.norm(2)  # L2 norm for this parameter
                total_norm += param_norm.item() ** 2

        # Compute the overall L2 norm across all gradients
        total_norm = total_norm ** 0.5
        return total_norm

    def update_ema_model(self):
        if self.num_gpus > 1:
            dist.barrier()
        if self.rank == 0:
            source_state = self.model.state_dict()
            rate = self.ema_rate
            for key, value in self.ema_state.items():
                if key in self.ema_ignore_keys:
                    self.ema_state[key] = source_state[key]
                else:
                    # print(f"Key: {key}, ema_state dtype: {self.ema_state[key].dtype}, source_state dtype: {source_state[key].dtype}, 1-rate dtype: {(1-rate.dtype)}")
                    # self.ema_state[key] = self.ema_state[key].to(dtype=source_state[key].dtype)
                    self.ema_state[key] = self.ema_state[key].to(dtype=torch.float32)
                    source_value = source_state[key].detach().to(dtype=torch.float32)
                    self.ema_state[key].mul_(rate).add_(source_state[key].detach().data, alpha=1-rate)

    def reload_ema_model(self):
        if self.rank == 0:
            if self.num_gpus > 1:
                model_state = {key[7:]:value for key, value in self.ema_state.items()}
            else:
                model_state = self.ema_state
            self.ema_model.load_state_dict(model_state)

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

class TrainerSR(TrainerBase):
    def build_model(self):
        super().build_model()

        # LPIPS metric
        # lpips_loss = lpips.LPIPS(net='alex').cuda()
        # self.freeze_model(lpips_loss)
        # self.lpips_loss = lpips_loss.eval()

    def feed_data(self, data, phase='train'):
        if phase == 'train':
            pred = self.model(data['lq'])
        elif phase == 'val':
            with torch.no_grad():
                if hasattr(self.configs.train, 'ema_rate'):
                    pred = self.ema_model(data['lq'])
                else:
                    pred = self.model(data['lq'])
        else:
            raise ValueError(f"Phase must be 'train' or 'val', now phase={phase}")

        return pred

    def get_loss(self, pred, data):
        target = data['gt']
        if self.configs.train.loss_type == "L1":
            return F.l1_loss(pred, target, reduction='mean')
        elif self.configs.train.loss_type == "L2":
            return F.mse_loss(pred, target, reduction='mean')
        else:
            raise ValueError(f"Not supported loss type: {self.configs.train.loss_type}")

    def setup_optimizaton(self):
        super().setup_optimizaton()   # self.optimizer
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max = self.configs.train.iterations,
                eta_min=self.configs.train.lr_min,
                )

    def training_step(self, data):
        current_batchsize = data['lq'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        self.optimizer.zero_grad()
        for jj in range(0, current_batchsize, micro_batchsize):
            micro_data = {key:value[jj:jj+micro_batchsize,] for key, value in data.items()}
            last_batch = (jj+micro_batchsize >= current_batchsize)
            hq_pred = self.feed_data(data, phase='train')
            if last_batch or self.num_gpus <= 1:
                loss = self.get_loss(hq_pred, micro_data)
            else:
                with self.model.no_sync():
                    loss = self.get_loss(hq_pred, micro_data)
            loss /= num_grad_accumulate
            loss.backward()

            # make logging
            self.log_step_train(hq_pred, loss, micro_data, flag=last_batch)

        self.optimizer.step()
        if hasattr(self.configs.train, 'ema_rate'):
            self.update_ema_model()

    def log_step_train(self, hq_pred, loss, batch, flag=False, phase='train'):
        '''
        param loss: loss value
        '''
        if self.rank == 0:
            chn = batch['lq'].shape[1]
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = 0

            self.loss_mean += loss.item()

            if self.current_iters % self.configs.train.log_freq[0] == 0 and flag:
                self.loss_mean /= self.configs.train.log_freq[0]
                log_str = 'Train:{:06d}/{:06d}, Loss:{:.2e}, lr:{:.2e}'.format(
                        self.current_iters,
                        self.configs.train.iterations,
                        self.loss_mean,
                        self.optimizer.param_groups[0]['lr']
                        )
                self.logger.info(log_str)
                self.logging_metric(self.loss_mean, 'Loss', phase, add_global_step=True)
            if self.current_iters % self.configs.train.log_freq[1] == 0 and flag:
                self.logging_image(batch['lq'], tag="lq", phase=phase, add_global_step=False)
                self.logging_image(batch['gt'], tag="hq", phase=phase, add_global_step=False)
                self.logging_image(batch['mask'], tag="mask", phase=phase, add_global_step=False)
                self.logging_image(hq_pred.detach(), tag="pred", phase=phase, add_global_step=True)

            if self.current_iters % self.configs.train.save_freq == 1 and flag:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0 and flag:
                self.toc = time.time()
                elaplsed = (self.toc - self.tic)
                self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                self.logger.info("="*60)

    def validation(self, phase='val'):
        if hasattr(self.configs.train, 'ema_rate'):
            self.reload_ema_model()
            self.ema_model.eval()
        else:
            self.model.eval()

        psnr_mean = lpips_mean = 0
        total_iters = math.ceil(len(self.datasets[phase]) / self.configs.train.batch[1])
        for ii, data in enumerate(self.dataloaders[phase]):
            data = self.prepare_data(data, phase='val')
            hq_pred = self.feed_data(data, phase='val')
            hq_pred.clamp_(0.0, 1.0)
            lpips = self.lpips_loss((hq_pred-0.5)*2, (data['gt']-0.5)*2).sum().item()
            psnr = util_image.batch_PSNR(hq_pred, data['gt'], ycbcr=True)

            psnr_mean += psnr
            lpips_mean += lpips

            if (ii+1) % self.configs.train.log_freq[2] == 0:
                log_str = '{:s}:{:03d}/{:03d}, PSNR={:5.2f}, LPIPS={:6.4f}'.format(
                        phase,
                        ii+1,
                        total_iters,
                        psnr / hq_pred.shape[0],
                        lpips / hq_pred.shape[0]
                        )
                self.logger.info(log_str)
                self.logging_image(data['lq'], tag="lq", phase=phase, add_global_step=False)
                self.logging_image(data['gt'], tag="hq", phase=phase, add_global_step=False)
                self.logging_image(data['mask'], tag="mask", phase=phase, add_global_step=False)
                self.logging_image(hq_pred.detach(), tag="pred", phase=phase, add_global_step=True)

        psnr_mean /= len(self.datasets[phase])
        lpips_mean /= len(self.datasets[phase])
        self.logging_metric(
                {"PSRN": psnr_mean, "lpips": lpips_mean},
                tag='Metrics',
                phase=phase,
                add_global_step=True,
                )
        # logging
        self.logger.info(f'PSNR={psnr_mean:5.2f}, LPIPS={lpips_mean:6.4f}')
        self.logger.info("="*60)

        if not hasattr(self.configs.train, 'ema_rate'):
            self.model.train()

class TrainerInpainting(TrainerSR):
    def get_loss(self, pred, data, weight_known=1, weight_missing=10):
        if self.configs.train.loss_type == "L1":
            mask, target = data['mask'], data['gt']
            per_pixel_loss = F.l1_loss(pred, target, reduction='none')
            pixel_weights = mask * weight_missing + (1 - mask) * weight_known
            loss = (pixel_weights * per_pixel_loss).sum() / pixel_weights.sum()
        elif self.configs.train.loss_type == "L2":
            mask, target = data['mask'], data['gt']
            per_pixel_loss = F.mse_loss(pred, target, reduction='none')
            pixel_weights = mask * weight_missing + (1 - mask) * weight_known
            loss = (pixel_weights * per_pixel_loss).sum() / pixel_weights.sum()
        else:
            raise ValueError(f"Not supported loss type: {self.configs.train.loss_type}")

        return loss

    def feed_data(self, data, phase='train'):
        if not 'mask' in data:
            ysum = torch.sum(data['lq'], dim=1, keepdim=True)
            mask = torch.where(
                    ysum==0,
                    torch.ones_like(ysum),
                    torch.zeros_like(ysum),
                    ).to(dtype=torch.float32, device=data['lq'].device)
        else:
            mask = data['mask']

        inputs = torch.cat([data['lq'], mask], dim=1)

        if phase == 'train':
            pred = self.model(inputs)
        elif phase == 'val':
            with torch.no_grad():
                if hasattr(self.configs.train, 'ema_rate'):
                    pred = self.ema_model(inputs)
                else:
                    pred = self.model(inputs)
        else:
            raise ValueError(f"Phase must be 'train' or 'val', now phase={phase}")

        return pred

class TrainerDiffusionFace(TrainerBase):
    def build_model(self):
        super().build_model()
        params = self.configs.diffusion.get('params', dict)
        self.base_diffusion = util_common.get_obj_from_str(self.configs.diffusion.target)(**params)
        self.sample_scheduler_diffusion = UniformSampler(self.base_diffusion.num_timesteps)

    def training_step(self, data):
        current_batchsize = data['image'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        if self.configs.train.use_fp16:
            scaler = amp.GradScaler()

        self.optimizer.zero_grad()
        for jj in range(0, current_batchsize, micro_batchsize):
            micro_data = {key:value[jj:jj+micro_batchsize,] for key, value in data.items()}
            last_batch = (jj+micro_batchsize >= current_batchsize)
            tt, weights = self.sample_scheduler_diffusion.sample(
                    micro_data['image'].shape[0],
                    device=f"cuda:{self.rank}",
                    use_fp16=self.configs.train.use_fp16
                    )
            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.model,
                micro_data['image'],
                tt,
                model_kwargs={'y':micro_data['label']} if 'label' in micro_data else None,
            )
            if self.configs.train.use_fp16:
                with amp.autocast():
                    if last_batch or self.num_gpus <= 1:
                        losses = compute_losses()
                    else:
                        with self.model.no_sync():
                            losses = compute_losses()
                    loss = (losses["loss"] * weights).mean() / num_grad_accumulate
                scaler.scale(loss).backward()
            else:
                if last_batch or self.num_gpus <= 1:
                    losses = compute_losses()
                else:
                    with self.model.no_sync():
                        losses = compute_losses()
                loss = (losses["loss"] * weights).mean() / num_grad_accumulate
                loss.backward()

            # make logging
            self.log_step_train(losses, tt, micro_data, last_batch)

        if self.configs.train.use_fp16:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()

        self.update_ema_model()

    def adjust_lr(self, current_iters=None):
        current_iters = self.current_iters if current_iters is None else current_iters
        base_lr = self.configs.train.lr
        linear_steps = self.configs.train.milestones[0]
        if current_iters <= linear_steps:
            for params_group in self.optimizer.param_groups:
                params_group['lr'] = (current_iters / linear_steps) * base_lr

    def log_step_train(self, loss, tt, batch, flag=False, phase='train'):
        '''
        param loss: a dict recording the loss informations
        param tt: 1-D tensor, time steps
        '''
        if self.rank == 0:
            chn = batch['image'].shape[1]
            num_timesteps = self.base_diffusion.num_timesteps
            record_steps = [1, (num_timesteps // 2) + 1, num_timesteps]
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = {key:torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                                  for key in loss.keys()}
                self.loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64)
            for jj in range(len(record_steps)):
                for key, value in loss.items():
                    index = record_steps[jj] - 1
                    mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt))
                    current_loss = torch.sum(value.detach() * mask)
                    self.loss_mean[key][jj] += current_loss.item()
                self.loss_count[jj] += mask.sum().item()

            if self.current_iters % self.configs.train.log_freq[0] == 0 and flag:
                if torch.any(self.loss_count == 0):
                    self.loss_count += 1e-4
                for key in loss.keys():
                    self.loss_mean[key] /= self.loss_count
                log_str = 'Train: {:06d}/{:06d}, Loss: '.format(
                        self.current_iters,
                        self.configs.train.iterations)
                for jj, current_record in enumerate(record_steps):
                    if 'vb' in self.loss_mean:
                        log_str += 't({:d}):{:.2e}/{:.2e}/{:.2e}, '.format(
                                current_record,
                                self.loss_mean['loss'][jj].item(),
                                self.loss_mean['mse'][jj].item(),
                                self.loss_mean['vb'][jj].item(),
                                )
                    else:
                        log_str += 't({:d}):{:.2e}, '.format(
                                current_record,
                                self.loss_mean['loss'][jj].item(),
                                )
                log_str += 'lr:{:.2e}'.format(self.optimizer.param_groups[0]['lr'])
                self.logger.info(log_str)
            if self.current_iters % self.configs.train.log_freq[1] == 0 and flag:
                self.logging_image(batch['image'], tag='image', phase=phase, add_global_step=True)

            if self.current_iters % self.configs.train.save_freq == 1 and flag:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0 and flag:
                self.toc = time.time()
                elaplsed = (self.toc - self.tic) * num_timesteps  / (num_timesteps - 1)
                self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                self.logger.info("="*130)

    def validation(self, phase='val'):
        self.reload_ema_model()
        self.ema_model.eval()

        indices = [int(self.base_diffusion.num_timesteps * x) for x in [0.25, 0.5, 0.75, 1]]
        chn = 3
        batch_size = self.configs.train.batch[1]
        shape = (batch_size, chn,) + (self.configs.data.train.params.img_size,) * 2
        num_iters = 0
        for sample in self.base_diffusion.p_sample_loop_progressive(
                model = self.ema_model,
                shape = shape,
                noise = None,
                clip_denoised = True,
                model_kwargs = None,
                device = f"cuda:{self.rank}",
                progress=False
                ):
            num_iters += 1
            img = util_image.normalize_th(sample['sample'], reverse=True)
            if num_iters == 1:
                im_recover = img
            elif num_iters in indices:
                im_recover_last = img
                im_recover = torch.cat((im_recover, im_recover_last), dim=1)
        im_recover = rearrange(im_recover, 'b (k c) h w -> (b k) c h w', c=chn)
        self.logging_image(
                im_recover,
                tag='progress',
                phase=phase,
                add_global_step=True,
                nrow=len(indices),
                )

class TrainerDiffusion(TrainerBase):
    def build_model(self):
        super().build_model()
        params = self.configs.diffusion.get('params', dict)
        self.base_diffusion = util_common.get_obj_from_str(self.configs.diffusion.target)(**params)
        self.sample_scheduler_diffusion = UniformSampler(self.base_diffusion.num_timesteps)
        self.load_initial_model()
        lpips_loss = lpips.LPIPS(net='vgg').cuda()
        self.freeze_model(lpips_loss)
        self.lpips_loss = lpips_loss.eval()
        self.cal_psnr = PeakSignalNoiseRatio()
        self.cal_ssim = StructuralSimilarityIndexMeasure()


    def load_initial_model(self):
        params_mask = self.configs.get('model_mask_params', dict)
        self.model_mask = util_common.get_obj_from_str(self.configs.model_mask_target)(**params_mask)
        if self.num_gpus >1:
            model_mask = nn.DataParallel(model_mask)
        self.model_mask = self.model_mask.to('cuda')
        if hasattr(self.configs, 'model_prior_ckpt') and self.configs.model_mask_ckpt is not None:
            ckpt_path = self.configs.model_mask_ckpt
            if self.rank == 0:
                self.logger.info(f"Initializing model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            util_net.reload_model(self.model_mask, ckpt)
        self.model_mask.eval()
        self.freeze_model(self.model_mask)


        params_prior = self.configs.get('model_prior_params', dict)
        self.model_prior = util_common.get_obj_from_str(self.configs.model_prior_target)(**params_prior)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            # Wrap the model with DataParallel
            model = nn.DataParallel(model)
        if self.num_gpus >1:
            model_prior = nn.DataParallel(model_prior)
        self.model_prior = self.model_prior.to('cuda')
        if hasattr(self.configs, 'model_prior_ckpt') and self.configs.model_prior_ckpt is not None:
            ckpt_path = self.configs.model_prior_ckpt
            if self.rank == 0:
                self.logger.info(f"Initializing model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            util_net.reload_model(self.model_prior, ckpt)
        self.model_prior.eval()
        self.freeze_model(self.model_prior)


    def training_step(self, data):
        current_batchsize = data['gt'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        if self.configs.train.use_fp16:
            scaler = amp.GradScaler()

        self.optimizer.zero_grad()
        sigmoid_layer = torch.nn.Sigmoid()
        for jj in range(0, current_batchsize, micro_batchsize):
            micro_data = {key:value[jj:jj+micro_batchsize,] for key, value in data.items()}
            last_batch = (jj+micro_batchsize >= current_batchsize)
            tt, weights = self.sample_scheduler_diffusion.sample(
                    micro_data['gt'].shape[0],
                    device=f"cuda:{self.rank}",
                    use_fp16=self.configs.train.use_fp16
                    )
            inital_mask= sigmoid_layer(self.model_mask(micro_data['lq']))
            inital_prior= sigmoid_layer(self.model_prior(micro_data['lq']))
            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.model,
                micro_data['lq'],
                tt,
                model_kwargs={
                    'mask': inital_mask,
                    'prior': inital_prior
                },
            )
            if self.configs.train.use_fp16:
                with amp.autocast():
                    if last_batch or self.num_gpus <= 1:
                        losses = compute_losses()
                    else:
                        with self.model.no_sync():
                            losses = compute_losses()

                #compute loss for mask and prior
                bce_loss = torch.nn.functional.binary_cross_entropy_with_logits
                mse_loss = torch.nn.functional.mse_loss
                loss_mask = bce_loss(losses['mask'],micro_data['mask'])
                if self.configs.train.prior=='edge':
                    loss_prior = bce_loss(losses['prior'],micro_data['prior'])
                elif self.configs.train.prior in ['segmentation','gradient']:
                    loss_prior = mse_loss(losses['prior'],micro_data['prior'])
                total_loss = losses['loss'] + self.configs.train.lambda_loss_mask*loss_mask + self.configs.train.lambda_loss_prior*loss_prior
                losses['mask_loss'] = loss_mask
                losses['prior_loss'] = loss_prior
                losses['total_loss'] = total_loss
                loss = (losses["total_loss"] * weights).mean() / num_grad_accumulate
                scaler.scale(loss).backward()
            else:
                if last_batch or self.num_gpus <= 1:
                    losses = compute_losses()
                else:
                    with self.model.no_sync():
                        losses = compute_losses()

                #compute loss for mask and prior
                bce_loss = torch.nn.functional.binary_cross_entropy_with_logits
                mse_loss = torch.nn.functional.mse_loss
                loss_mask = bce_loss(losses['mask'],micro_data['mask'])

                if self.configs.train.prior=='edge':
                    loss_prior = bce_loss(losses['prior'],micro_data['prior'])
                elif self.configs.train.prior in ['segmentation','gradient']:
                    loss_prior = mse_loss(losses['prior'],micro_data['prior'])
                total_loss = losses['loss']  + self.configs.train.lambda_loss_mask*loss_mask + self.configs.train.lambda_loss_prior*loss_prior
                losses['mask_loss'] = loss_mask
                losses['prior_loss'] = loss_prior
                losses['total_loss'] = total_loss

                loss = ((losses["loss"] * weights).mean() + self.configs.train.lambda_loss_mask*loss_mask + self.configs.train.lambda_loss_prior*loss_prior) / num_grad_accumulate
                loss.backward()
            

            # make logging
            self.log_step_train(losses, tt, micro_data, last_batch)

        if self.configs.train.use_fp16:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()

        self.update_ema_model()

    def adjust_lr(self, current_iters=None):
        current_iters = self.current_iters if current_iters is None else current_iters
        base_lr = self.configs.train.lr
        linear_steps = self.configs.train.milestones[0]
        if current_iters <= linear_steps:
            for params_group in self.optimizer.param_groups:
                params_group['lr'] = (current_iters / linear_steps) * base_lr

    def log_step_train(self, loss, tt, batch, flag=False, phase='train'):
        '''
        param loss: a dict recording the loss informations
        param tt: 1-D tensor, time steps
        '''
        if self.rank == 0:
            chn = batch['lq'].shape[1]
            num_timesteps = self.base_diffusion.num_timesteps
            record_steps = [1, (num_timesteps // 2) + 1, num_timesteps]
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = {key: 0 for key in loss.keys()}
            

            for key in ['mask_loss', 'prior_loss', 'total_loss']:
                if key in ['mask_loss', 'prior_loss']:
                    self.loss_mean[key] += loss[key].item()
                else:
                    self.loss_mean[key] += torch.mean(loss[key]).item()

            if self.current_iters % self.configs.train.log_freq[0] == 0 and flag:
                for key in loss.keys():
                    self.loss_mean[key] /= self.configs.train.log_freq[0] 
                log_str = 'Train: {:06d}/{:06d}, Loss: '.format(
                        self.current_iters,
                        self.configs.train.iterations)
               
                log_str += '{:.4e}/{:.4e}/{:.4e}, '.format(
                                self.loss_mean['total_loss'],
                                self.loss_mean['prior_loss'],
                                self.loss_mean['mask_loss']
                                )
                log_str += 'lr:{:.2e}'.format(self.optimizer.param_groups[0]['lr'])
                self.logger.info(log_str)
                wandb.log({"loss_mask":  self.loss_mean['mask_loss'], "loss_prior": self.loss_mean['prior_loss'], 'total_loss': self.loss_mean['total_loss'],  'epoch': self.epoch})
            if self.current_iters % self.configs.train.log_freq[0] == 0 and flag:
                self.logging_image(batch['gt'], tag='hq', phase=phase, add_global_step=True)
                self.logging_image(batch['lq'], tag='lq', phase=phase, add_global_step=True)
                self.logging_image(batch['mask'], tag='mask', phase=phase, add_global_step=True)
                self.logging_image(batch['prior'], tag='prior', phase=phase, add_global_step=True)

            if self.current_iters % self.configs.train.save_freq == 1 and flag:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0 and flag:
                self.toc = time.time()
                elaplsed = (self.toc - self.tic) * num_timesteps  / (num_timesteps - 1)
                self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                self.logger.info("="*130)

    def validation(self, phase='val'):
        self.reload_ema_model()
        self.ema_model.eval()
        self.freeze_model(self.ema_model)
        num_iters = 0
        total_iters = math.ceil(len(self.datasets[phase])/ self.configs.train.batch[1])
        sigmoid_layer = torch.nn.Sigmoid()
        if self.configs.train.only_log_image_val:
            for ii, data in enumerate(self.dataloaders[phase]):
                data = self.prepare_data(data)
                yt = self.base_diffusion.q_sample( 
                    x_start=util_image.normalize_th(data['lq'], mean=0.5, std=0.5, reverse=False),
                    t=torch.tensor([self.base_diffusion.num_timesteps-1,]*data['lq'].shape[0], device=f"cuda:{self.rank}")
                )
                
                initial_mask = sigmoid_layer(self.model_mask(data['lq']))
                initial_prior = sigmoid_layer(self.model_prior(data['lq']))

                final_sample = self.base_diffusion.p_sample_loop(
                    model = self.ema_model,
                    shape = yt.shape,
                    noise = yt,
                    device=f"cuda:{self.rank}",
                    clip_denoised = True,
                    denoised_fn=None,
                    progress = False,
                    model_kwargs={
                        'mask': initial_mask,
                        'prior': initial_prior
                    }
                )
                psnr_mean = 0
                lpips_mean = 0
                ssim_mean = 0
                mask_recover = torch.where(final_sample['mask']>0.5,1,0)
                mask_reshape = mask_recover.expand(data['lq'].shape[0],3, -1, -1)
                prior_recover = final_sample['prior']
                hq_pred = data['gt'] *  (1 - mask_reshape) +mask_reshape*final_sample['sample'] 
                self.logging_image(data['gt'], tag="hq", phase=phase, add_global_step=False)
                
                self.logging_image(hq_pred, tag="pred", phase=phase, add_global_step=False)
                        
                self.logging_image(mask_recover.float(),tag="pred_mask",phase=phase, add_global_step=False)

                self.logging_image( prior_recover,tag="pred_prior",phase=phase, add_global_step=False)
                self.logging_image(mask_recover.float(),tag="pred_mask",phase=phase, add_global_step=False)

                self.logging_image( initial_mask,tag="initial_mask",phase=phase, add_global_step=False)
                self.logging_image( initial_prior,tag="initial_prior",phase=phase, add_global_step=False)
                break
            return


        for ii, data in enumerate(self.dataloaders[phase]):
            data = self.prepare_data(data)
            yt = self.base_diffusion.q_sample( 
                x_start=util_image.normalize_th(data['lq'], mean=0.5, std=0.5, reverse=False),
                t=torch.tensor([self.base_diffusion.num_timesteps-1,]*data['lq'].shape[0], device=f"cuda:{self.rank}")
            )
            
            initial_mask = sigmoid_layer(self.model_mask(data['lq']))
            initial_prior = sigmoid_layer(self.model_prior(data['lq']))

            final_sample = self.base_diffusion.p_sample_loop(
                model = self.ema_model,
                shape = yt.shape,
                noise = yt,
                device=f"cuda:{self.rank}",
                clip_denoised = True,
                denoised_fn=None,
                progress = False,
                model_kwargs={
                    'mask': initial_mask,
                    'prior': initial_prior
                }
            )
            psnr_mean = 0
            lpips_mean = 0
            ssim_mean = 0
            # for sample in diffusion_progress:
            #     print(type(sample))
            #     num_iters += 1
            #     img = util_image.normalize_th(sample['sample'], reverse=True)
            #     prior = util_image.normalize_th(sample['prior'], reverse=True)
            #     mask = util_image.normalize_th(sample['mask'], reverse=True)
            #     if num_iters == 1:
            #         im_recover = img
            #         prior_recover = prior
            #         mask_recover = mask
            #     elif num_iters in indices:
            #         im_recover_last = img
            #         im_recover = torch.cat((im_recover, im_recover_last), dim=1)
            #         prior_recover_last = prior
            #         prior_recover = torch.cat((prior_recover, prior_recover_last), dim=1)
            #         mask_recover_last = mask
            #         mask_recover = torch.cat((mask_recover, mask_recover_last), dim=1)
            #     im_recover = rearrange(im_recover, 'b (k c) h w -> (b k) c h w', c=chn)
            #     mask_recover = rearrange(mask_recover, 'b (k c) h w -> (b k) c h w', c=1)
            #     prior_recover = rearrange(prior_recover, 'b (k c) h w -> (b k) c h w', c=data['prior'].shape[1])
            # mask_recover_last = torch.where(mask_recover_last>0.5,1,0)
            # mask_reshape = mask_recover_last.expand(batch_size,3, -1, -1)
            # hq_pred = data['gt'] *  (1 - mask_reshape) +mask_reshape*im_recover_last 
            mask_recover = torch.where(final_sample['mask']>0.5,1,0)
            mask_reshape = mask_recover.expand(data['lq'].shape[0],3, -1, -1)
            prior_recover = final_sample['prior']
            hq_pred = data['gt'] *  (1 - mask_reshape) +mask_reshape*final_sample['sample'] 
            lpips = self.lpips_loss((hq_pred-0.5)*2, (data['gt']-0.5)*2).sum().item()
            self.cal_psnr = self.cal_psnr.to(data['gt'].get_device())
            self.cal_ssim = self.cal_ssim.to(data['gt'].get_device())
            psnr  = self.cal_psnr(hq_pred, data['gt'])
            ssim  = self.cal_ssim(hq_pred, data['gt'])
            # psnr = util_image.batch_PSNR(hq_pred, data['gt'], ycbcr=True)
            # ssim = util_image.batch_SSIM(hq_pred, data['gt'], ycbcr=True)
            psnr_mean += psnr
            lpips_mean += lpips
            ssim_mean += ssim
            if (ii+1) % self.configs.train.log_freq[2] == 0:
                log_str = '{:s}:{:03d}/{:03d}, PSNR={:6.3f}, SSIM={:6.3f}, LPIPS={:7.5f}'.format(
                        phase,
                        ii+1,
                        total_iters,
                        psnr,
                        ssim,
                        lpips
                        )
                self.logger.info(log_str)
                self.logging_image(data['gt'], tag="hq", phase=phase, add_global_step=False)
                
                self.logging_image(hq_pred, tag="pred", phase=phase, add_global_step=False)
                        
                self.logging_image(mask_recover.float(),tag="pred_mask",phase=phase, add_global_step=False)

                self.logging_image( prior_recover,tag="pred_prior",phase=phase, add_global_step=False)
                self.logging_image(mask_recover.float(),tag="pred_mask",phase=phase, add_global_step=False)

                self.logging_image( initial_mask,tag="initial_mask",phase=phase, add_global_step=False)
                self.logging_image( initial_prior,tag="initial_prior",phase=phase, add_global_step=False)
            num_iters=ii+1
        psnr_mean /= num_iters
        lpips_mean /= num_iters
        ssim_mean /= num_iters
        self.logging_metric(
                {"PSRN": psnr_mean, "SSIM": ssim_mean, "lpips": lpips_mean},
                tag='Metrics',
                phase=phase,
                add_global_step=True,
                )
        # logging
        wandb.log({"PSNR": psnr_mean, "lpips": lpips_mean, 'SSIM': ssim_mean, 'epoch': self.epoch})
        self.logger.info(f'PSNR={psnr_mean:6.3f}, SSIM={ssim_mean:6.3f}, LPIPS={lpips_mean:7.5f}')
        self.logger.info("="*60)
        self.epoch +=1

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class TrainerPredictedMask(TrainerSR):
    def get_loss(self, pred, data, weight_known=1, weight_missing=10):
        if self.configs.train.loss_type == "BCE":
            target = data['mask']
            loss = F.binary_cross_entropy_with_logits(pred,target)
        else:
            raise ValueError(f"Not supported loss type: {self.configs.train.loss_type}")

        return loss

    def feed_data(self, data, phase='train'):
        inputs = data['lq']

        if phase == 'train':
            pred = self.model(inputs)
        elif phase == 'val':
            with torch.no_grad():
                if hasattr(self.configs.train, 'ema_rate'):
                    pred = self.ema_model(inputs)
                else:
                    pred = self.model(inputs)
        else:
            raise ValueError(f"Phase must be 'train' or 'val', now phase={phase}")

        return pred
    
    def log_step_train(self, hq_pred, loss, batch, flag=False, phase='train'):
        '''
        param loss: loss value
        '''
        if self.rank == 0:
            chn = batch['lq'].shape[1]
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = 0
                self.L2_gradient = 0 

            self.loss_mean += loss.item()
            self.L2_gradient += self.calculate_L2_gradient()
            if self.current_iters % self.configs.train.log_freq[0] == 0 and flag:
                self.loss_mean /= self.configs.train.log_freq[0]
                self.L2_gradient  /= self.configs.train.log_freq[0]
                log_str = 'Train:{:06d}/{:06d}, Loss:{:.2e}, lr:{:.2e}, gradient:{:4e}'.format(
                        self.current_iters,
                        self.configs.train.iterations,
                        self.loss_mean,
                        self.optimizer.param_groups[0]['lr'],
                        self.L2_gradient
                        )
                self.logger.info(log_str)
                wandb.log({"train/loss": self.loss_mean, 'train/learning_rate': self.optimizer.param_groups[0]['lr'], 'train/gradient': self.L2_gradient, "epoch": self.epoch })
                self.logging_metric(self.loss_mean, 'Loss', phase, add_global_step=True)
            if self.current_iters % self.configs.train.log_freq[1] == 0 and flag:
                self.logging_image(batch['lq'], tag="lq", phase=phase, add_global_step=False)
                self.logging_image(batch['mask'], tag="mask", phase=phase, add_global_step=False)
                self.logging_image(hq_pred.detach(), tag="pred", phase=phase, add_global_step=True)

            if self.current_iters % self.configs.train.save_freq == 1 and flag:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0 and flag:
                self.toc = time.time()
                elaplsed = (self.toc - self.tic)
                self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                self.logger.info("="*60)
    def validation(self, phase='val'):
        if hasattr(self.configs.train, 'ema_rate'):
            self.reload_ema_model()
            self.ema_model.eval()
        else:
            self.model.eval()

        total_iters = math.ceil(len(self.datasets[phase]) / self.configs.train.batch[1])
        loss_mean=0
        loss_avg=0
        print(total_iters, len(self.dataloaders[phase]) )
        sigmoid_layer = torch.nn.Sigmoid()
        for ii, data in enumerate(self.dataloaders[phase]):
            data = self.prepare_data(data, phase='val')
            hq_pred = self.feed_data(data, phase='val')
            loss = self.get_loss(hq_pred, data)
            loss_mean += loss.item()
            loss_avg += loss.item()
            hq_pred = sigmoid_layer(hq_pred)
            if (ii+1) % self.configs.train.log_freq[2] == 0:
                loss_avg/=self.configs.train.log_freq[2]
                log_str = '{:s}:{:03d}/{:03d}, loss={:5.8f}'.format(
                        phase,
                        ii+1,
                        total_iters,
                        loss_avg
                        )
                hq_pred = sigmoid_layer(hq_pred)
                self.logger.info(log_str)
                self.logging_image(data['lq'], tag="lq", phase=phase, add_global_step=False)
                self.logging_image(data['mask'], tag="mask", phase=phase, add_global_step=False)
                self.logging_image(hq_pred.detach(), tag="pred", phase=phase, add_global_step=True)
                loss_avg=0
        loss_mean /= total_iters
        self.logging_metric(
                {"loss": loss_mean},
                tag='Metrics',
                phase=phase,
                add_global_step=True,
                )
        wandb.log({"val/loss": loss_mean, "epoch": self.epoch})
        self.logger.info(f'loss={loss_mean:5.8f}')
        if not hasattr(self.configs.train, 'ema_rate'):
            self.model.train()
        self.epoch +=1

class TrainerPredictedPrior(TrainerSR):
    def get_loss(self, pred, data, weight_known=1, weight_missing=10):
        if self.configs.train.loss_type == "BCE":
            mask, target = data['mask'], data['prior']
            per_pixel_loss = F.binary_cross_entropy_with_logits(pred,target,reduction='none')
            pixel_weights = mask *weight_known  + (1 - mask) * weight_missing
            loss = (pixel_weights * per_pixel_loss).sum() / pixel_weights.sum()
        elif self.configs.train.loss_type == "WCE":
            mask, target = data['mask'], data['prior']
            
            num_negative = torch.sum(data['prior']==0).float()
            num_positive = torch.sum(data['prior']==1).float()

            weight = copy.deepcopy(mask)
            weight[(mask == 1) & (target==1)] = 1.0 * num_negative / (num_negative+num_positive)
            weight[(mask == 1) & (target==0)] = 1.0 * 1.1 * num_positive / (num_negative+num_positive)
            weight[(mask == 0) & (target==1)] = weight_missing *  num_negative / (num_negative+num_positive)
            weight[(mask == 0) & (target==0)] = weight_missing * 1.1 * num_positive / (num_negative+num_positive)
            loss = F.binary_cross_entropy_with_logits(pred,target,weight=weight)
        # elif self.configs.train.loss_type == "focal":
            
        else:
            raise ValueError(f"Not supported loss type: {self.configs.train.loss_type}")

        return loss
    
    def get_loss_val(self, pred, type_loss, data, weight_known=1, weight_missing=10):
        if type_loss == "BCE":
            mask, target = data['mask'], data['prior']
            per_pixel_loss = F.binary_cross_entropy_with_logits(pred,target,reduction='none')
            pixel_weights = mask *weight_known  + (1 - mask) * weight_missing
            loss = (pixel_weights * per_pixel_loss).sum() / pixel_weights.sum()
        else:
            raise ValueError(f"Not supported loss type: {self.configs.train.loss_type}")

        return loss

    def feed_data(self, data, phase='train'):

        inputs =data['lq']

        if phase == 'train':
            pred = self.model(inputs)
        elif phase == 'val':
            with torch.no_grad():
                if hasattr(self.configs.train, 'ema_rate'):
                    pred = self.ema_model(inputs)
                else:
                    pred = self.model(inputs)
        else:
            raise ValueError(f"Phase must be 'train' or 'val', now phase={phase}")

        return pred
    def training_step(self, data):
        current_batchsize = data['lq'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)
        
        self.optimizer.zero_grad()
        for jj in range(0, current_batchsize, micro_batchsize):
            micro_data = {key:value[jj:jj+micro_batchsize,] for key, value in data.items()}
            last_batch = (jj+micro_batchsize >= current_batchsize)
            hq_pred = self.feed_data(data, phase='train')
            if last_batch or self.num_gpus <= 1:
                loss = self.get_loss(hq_pred, micro_data)
            else:
                with self.mcodel.no_sync():
                    loss = self.get_loss(hq_pred, micro_data)
            loss /= num_grad_accumulate
            loss.backward()

            # make logging
            self.log_step_train(hq_pred, loss, micro_data, flag=last_batch)

        self.optimizer.step()
        if hasattr(self.configs.train, 'ema_rate'):
            self.update_ema_model()
    def log_step_train(self, hq_pred, loss, batch, flag=False, phase='train'):
        '''
        param loss: loss value
        '''
        if self.rank == 0:
            chn = batch['lq'].shape[1]
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = 0
                self.L2_gradient = 0 

            self.loss_mean += loss.item()
            self.L2_gradient += self.calculate_L2_gradient()
            if self.current_iters % self.configs.train.log_freq[0] == 0 and flag:
                self.loss_mean /= self.configs.train.log_freq[0]
                self.L2_gradient  /= self.configs.train.log_freq[0]
                log_str = 'Train:{:06d}/{:06d}, Loss:{:.2e}, lr:{:.2e}, gradient:{:4e}'.format(
                        self.current_iters,
                        self.configs.train.iterations,
                        self.loss_mean,
                        self.optimizer.param_groups[0]['lr'],
                        self.L2_gradient
                        )
                self.logger.info(log_str)
                wandb.log({"train/loss": self.loss_mean, 'train/learning_rate': self.optimizer.param_groups[0]['lr'], 'train/gradient': self.L2_gradient, "epoch": self.epoch})
                self.logging_metric(self.loss_mean, 'Loss', phase, add_global_step=True)
            if self.current_iters % self.configs.train.log_freq[1] == 0 and flag:
                self.logging_image(batch['lq'], tag="lq", phase=phase, add_global_step=False)
                self.logging_image(batch['mask'], tag="mask", phase=phase, add_global_step=False)
                self.logging_image(batch['prior'], tag="prior", phase=phase, add_global_step=False)
                self.logging_image(hq_pred.detach(), tag="pred", phase=phase, add_global_step=True)

            if self.current_iters % self.configs.train.save_freq == 1 and flag:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0 and flag:
                self.toc = time.time()
                elaplsed = (self.toc - self.tic)
                self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                self.logger.info("="*60)

    def validation(self, phase='val'):
        if hasattr(self.configs.train, 'ema_rate'):
            self.reload_ema_model()
            self.ema_model.eval()
        else:
            self.model.eval()

        total_iters = math.ceil(len(self.datasets[phase]) / self.configs.train.batch[1])
        loss_mean=0
        loss_avg=0
        print(total_iters, len(self.dataloaders[phase]) )
        sigmoid_layer=  torch.nn.Sigmoid()
        for ii, data in enumerate(self.dataloaders[phase]):
            data = self.prepare_data(data, phase='val')
            hq_pred = self.feed_data(data, phase='val')
            loss = self.get_loss_val(hq_pred, "BCE",data)
            
            loss_mean += loss.item()
            loss_avg += loss.item()

            if (ii+1) % self.configs.train.log_freq[2] == 0:
                loss_avg/=self.configs.train.log_freq[2]
                log_str = '{:s}:{:03d}/{:03d}, loss={:5.8f}'.format(
                        phase,
                        ii+1,
                        total_iters,
                        loss_avg
                        )
                hq_pred = sigmoid_layer(hq_pred)
                self.logger.info(log_str)
                self.logging_image(data['lq'], tag="lq", phase=phase, add_global_step=False)
                self.logging_image(data['prior'], tag="prior", phase=phase, add_global_step=False)
                self.logging_image(hq_pred.detach(), tag="pred", phase=phase, add_global_step=True)
                loss_avg=0
        loss_mean /= total_iters
        self.logging_metric(
                {"loss": loss_mean},
                tag='Metrics',
                phase=phase,
                add_global_step=True,
                )
        wandb.log({"val/loss": loss_mean, "epoch": self.epoch})
        self.logger.info(f'loss={loss_mean:5.8f}')
        if not hasattr(self.configs.train, 'ema_rate'):
            self.model.train()
        self.epoch +=1

if __name__ == '__main__':
    from utils import util_image
    from  einops import rearrange
    im1 = util_image.imread('./testdata/inpainting/val/places/Places365_val_00012685_crop000.png',
                            chn = 'rgb', dtype='float32')
    im2 = util_image.imread('./testdata/inpainting/val/places/Places365_val_00014886_crop000.png',
                            chn = 'rgb', dtype='float32')
    im = rearrange(np.stack((im1, im2), 3), 'h w c b -> b c h w')
    im_grid = im.copy()
    for alpha in [0.8, 0.4, 0.1, 0]:
        im_new = im * alpha + np.random.randn(*im.shape) * (1 - alpha)
        im_grid = np.concatenate((im_new, im_grid), 1)

    im_grid = np.clip(im_grid, 0.0, 1.0)
    im_grid = rearrange(im_grid, 'b (k c) h w -> (b k) c h w', k=5)
    xx = vutils.make_grid(torch.from_numpy(im_grid), nrow=5, normalize=True, scale_each=True).numpy()
    util_image.imshow(np.concatenate((im1, im2), 0))
    util_image.imshow(xx.transpose((1,2,0)))

