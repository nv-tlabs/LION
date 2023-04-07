# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

""" copied and modified from https://github.com/stevenygd/PointFlow/blob/master/datasets.py """
import os
import open3d as o3d
import time
import torch
import numpy as np
from loguru import logger
from torch.utils.data import Dataset
from torch.utils import data
import random
import tqdm
from datasets.data_path import get_path
from PIL import Image
OVERFIT = 0

# taken from https://github.com/optas/latent_3d_points/blob/
# 8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane',
    '02773838': 'bag',
    '02801938': 'basket',
    '02808440': 'bathtub',
    '02818832': 'bed',
    '02828884': 'bench',
    '02876657': 'bottle',
    '02880940': 'bowl',
    '02924116': 'bus',
    '02933112': 'cabinet',
    '02747177': 'can',
    '02942699': 'camera',
    '02954340': 'cap',
    '02958343': 'car',
    '03001627': 'chair',
    '03046257': 'clock',
    '03207941': 'dishwasher',
    '03211117': 'monitor',
    '04379243': 'table',
    '04401088': 'telephone',
    '02946921': 'tin_can',
    '04460130': 'tower',
    '04468005': 'train',
    '03085013': 'keyboard',
    '03261776': 'earphone',
    '03325088': 'faucet',
    '03337140': 'file',
    '03467517': 'guitar',
    '03513137': 'helmet',
    '03593526': 'jar',
    '03624134': 'knife',
    '03636649': 'lamp',
    '03642806': 'laptop',
    '03691459': 'speaker',
    '03710193': 'mailbox',
    '03759954': 'microphone',
    '03761084': 'microwave',
    '03790512': 'motorcycle',
    '03797390': 'mug',
    '03928116': 'piano',
    '03938244': 'pillow',
    '03948459': 'pistol',
    '03991062': 'pot',
    '04004475': 'printer',
    '04074963': 'remote_control',
    '04090263': 'rifle',
    '04099429': 'rocket',
    '04225987': 'skateboard',
    '04256520': 'sofa',
    '04330267': 'stove',
    '04530566': 'vessel',
    '04554684': 'washer',
    '02992529': 'cellphone',
    '02843684': 'birdhouse',
    '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class ShapeNet15kPointClouds(Dataset):
    def __init__(self,
                 categories=['airplane'],
                 tr_sample_size=10000,
                 te_sample_size=10000,
                 split='train',
                 scale=1.,
                 normalize_per_shape=False,
                 normalize_shape_box=False,
                 random_subsample=False,
                 sample_with_replacement=1,
                 normalize_std_per_axis=False,
                 normalize_global=False,
                 recenter_per_shape=False,
                 all_points_mean=None,
                 all_points_std=None,
                 input_dim=3, 
                 clip_forge_enable=0, clip_model=None
                 ):
        self.clip_forge_enable = clip_forge_enable 
        if clip_forge_enable:
            import clip
            _, self.clip_preprocess = clip.load(clip_model)
        if self.clip_forge_enable:
            self.img_path = []
            img_path = get_path('clip_forge_image') 

        self.normalize_shape_box = normalize_shape_box
        root_dir = get_path('pointflow')
        self.root_dir = root_dir
        logger.info('[DATA] cat: {}, split: {}, full path: {}; norm global={}, norm-box={}',
                    categories, split, self.root_dir, normalize_global, normalize_shape_box)

        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.tr_sample_size = tr_sample_size
        self.te_sample_size = te_sample_size
        if type(categories) is str:
            categories = [categories]
        self.cates = categories

        if 'all' in categories:
            self.synset_ids = list(cate_to_synsetid.values())
        else:
            self.synset_ids = [cate_to_synsetid[c] for c in self.cates]
        subdirs = self.synset_ids
        # assert 'v2' in root_dir, "Only supporting v2 right now."
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        self.root_dir = root_dir
        self.split = split
        self.in_tr_sample_size = tr_sample_size
        self.in_te_sample_size = te_sample_size
        self.subdirs = subdirs
        self.scale = scale
        self.random_subsample = random_subsample
        self.sample_with_replacement = sample_with_replacement
        self.input_dim = input_dim

        self.all_cate_mids = []
        self.cate_idx_lst = []
        self.all_points = []
        tic = time.time()
        for cate_idx, subd in enumerate(self.subdirs):
            # NOTE: [subd] here is synset id
            sub_path = os.path.join(root_dir, subd, self.split)
            if not os.path.isdir(sub_path):
                print("Directory missing : %s " % (sub_path))
                raise ValueError('check the data path')
                continue

            if True:
                all_mids = []
                assert(os.path.exists(sub_path)), f'path missing: {sub_path}'
                for x in os.listdir(sub_path):
                    if not x.endswith('.npy'):
                        continue
                    all_mids.append(os.path.join(self.split, x[:-len('.npy')]))

                logger.info('[DATA] number of file [{}] under: {} ',
                            len(os.listdir(sub_path)), sub_path)
                # NOTE: [mid] contains the split: i.e. "train/<mid>"
                # or "val/<mid>" or "test/<mid>"
                all_mids = sorted(all_mids)
                for mid in all_mids:
                    # obj_fname = os.path.join(sub_path, x)
                    if self.clip_forge_enable:
                        synset_id = subd
                        render_img_path = os.path.join(img_path, synset_id, mid.split('/')[-1], 'img_choy2016')
                        
                        #render_img_path = os.path.join(img_path, synset_id, mid.split('/')[-1])
                        #if not (os.path.exists(render_img_path)): continue
                        self.img_path.append(render_img_path)
                        assert(os.path.exists(render_img_path)), f'render img path not find: {render_img_path}'

                    obj_fname = os.path.join(root_dir, subd, mid + ".npy")
                    point_cloud = np.load(obj_fname)  # (15k, 3)
                    self.all_points.append(point_cloud[np.newaxis, ...])
                    self.cate_idx_lst.append(cate_idx)
                    self.all_cate_mids.append((subd, mid))

        logger.info('[DATA] Load data time: {:.1f}s | dir: {} | '
                    'sample_with_replacement: {}; num points: {}', time.time() - tic, self.subdirs,
                    self.sample_with_replacement, len(self.all_points))

        # Shuffle the index deterministically (based on the number of examples)
        self.shuffle_idx = list(range(len(self.all_points)))
        random.Random(38383).shuffle(self.shuffle_idx)
        self.cate_idx_lst = [self.cate_idx_lst[i] for i in self.shuffle_idx]
        self.all_points = [self.all_points[i] for i in self.shuffle_idx]
        self.all_cate_mids = [self.all_cate_mids[i] for i in self.shuffle_idx]
        if self.clip_forge_enable:
            self.img_path = [self.img_path[i] for i in self.shuffle_idx]

        # Normalization
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        self.recenter_per_shape = recenter_per_shape
        if self.normalize_shape_box:  # per shape normalization
            B, N = self.all_points.shape[:2]
            self.all_points_mean = (  # B,1,3
                (np.amax(self.all_points, axis=1)).reshape(B, 1, input_dim) +
                (np.amin(self.all_points, axis=1)).reshape(B, 1, input_dim)) / 2
            self.all_points_std = np.amax(  # B,1,1
                ((np.amax(self.all_points, axis=1)).reshape(B, 1, input_dim) -
                 (np.amin(self.all_points, axis=1)).reshape(B, 1, input_dim)),
                axis=-1).reshape(B, 1, 1) / 2
        elif self.normalize_per_shape:  # per shape normalization
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.mean(axis=1).reshape(
                B, 1, input_dim)
            logger.info('all_points shape: {}. mean over axis=1',
                        self.all_points.shape)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(
                    B, N, -1).std(axis=1).reshape(B, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(
                    B, -1).std(axis=1).reshape(B, 1, 1)
        elif all_points_mean is not None and all_points_std is not None and not self.recenter_per_shape:
            # using loaded dataset stats
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        elif self.recenter_per_shape:  # per shape center
            # TODO: bounding box scale at the large dim and center
            B, N = self.all_points.shape[:2]
            self.all_points_mean = (
                (np.amax(self.all_points, axis=1)).reshape(B, 1, input_dim) +
                (np.amin(self.all_points, axis=1)).reshape(B, 1,
                                                           input_dim)) / 2
            self.all_points_std = np.amax(
                ((np.amax(self.all_points, axis=1)).reshape(B, 1, input_dim) -
                 (np.amin(self.all_points, axis=1)).reshape(B, 1, input_dim)),
                axis=-1).reshape(B, 1, 1) / 2
        # else:  # normalize across the dataset
        elif normalize_global:  # normalize across the dataset
            self.all_points_mean = self.all_points.reshape(
                -1, input_dim).mean(axis=0).reshape(1, 1, input_dim)

            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(
                    -1, input_dim).std(axis=0).reshape(1, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(-1).std(
                    axis=0).reshape(1, 1, 1)

            logger.info('[DATA] normalize_global: mean={}, std={}',
                        self.all_points_mean.reshape(-1),
                        self.all_points_std.reshape(-1))
        else:
            raise NotImplementedError('No Normalization')
        self.all_points = (self.all_points - self.all_points_mean) / \
            self.all_points_std
        logger.info('[DATA] shape={}, all_points_mean:={}, std={}, max={:.3f}, min={:.3f}; num-pts={}',
                    self.all_points.shape,
                    self.all_points_mean.shape, self.all_points_std.shape,
                    self.all_points.max(), self.all_points.min(), tr_sample_size)

        if OVERFIT:
            self.all_points = self.all_points[:40]

        # TODO: why do we need this??
        self.train_points = self.all_points[:, :min(
            10000, self.all_points.shape[1])]  # subsample 15k points to 10k points per shape
        self.tr_sample_size = min(10000, tr_sample_size)
        self.te_sample_size = min(5000, te_sample_size)
        assert self.scale == 1, "Scale (!= 1) is deprecated"

        # Default display axis order
        self.display_axis_order = [0, 1, 2]

    def get_pc_stats(self, idx):
        if self.recenter_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s

        if self.normalize_per_shape or self.normalize_shape_box:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s

        return self.all_points_mean.reshape(1, -1), \
            self.all_points_std.reshape(1, -1)

    def renormalize(self, mean, std):
        self.all_points = self.all_points * self.all_points_std + \
            self.all_points_mean
        self.all_points_mean = mean
        self.all_points_std = std
        self.all_points = (self.all_points - self.all_points_mean) / \
            self.all_points_std
        self.train_points = self.all_points[:, :min(
            10000, self.all_points.shape[1])]
        ## self.test_points = self.all_points[:, 10000:]

    def __len__(self):
        return len(self.train_points)

    def __getitem__(self, idx):
        output = {}
        tr_out = self.train_points[idx]
        if self.random_subsample and self.sample_with_replacement:
            tr_idxs = np.random.choice(tr_out.shape[0], self.tr_sample_size)
        elif self.random_subsample and not self.sample_with_replacement:
            tr_idxs = np.random.permutation(
                np.arange(tr_out.shape[0]))[:self.tr_sample_size]
        else:
            tr_idxs = np.arange(self.tr_sample_size)
        tr_out = torch.from_numpy(tr_out[tr_idxs, :]).float()
        m, s = self.get_pc_stats(idx)

        cate_idx = self.cate_idx_lst[idx]
        sid, mid = self.all_cate_mids[idx]
        input_pts = tr_out
    
        output.update(
            {
                'idx': idx,
                'select_idx': tr_idxs,
                'tr_points': tr_out,
                'input_pts': input_pts,
                'mean': m,
                'std': s,
                'cate_idx': cate_idx,
                'sid': sid,
                'mid': mid,
                'display_axis_order': self.display_axis_order
            })

        # read image 
        if self.clip_forge_enable:
            img_path = self.img_path[idx]
            img_list = os.listdir(img_path) 
            img_list = [os.path.join(img_path, p) for p in img_list if 'jpg' in p or 'png' in p]
            assert(len(img_list) > 0), f'get empty list at {img_path}: {os.listdir(img_path)}'
            # subset 5 image
            img_idx = np.random.choice(len(img_list), 5) 
            img_list = [img_list[o] for o in img_idx]
            img_list = [Image.open(img).convert('RGB') for img in img_list] 
            img_list = [self.clip_preprocess(img) for img in img_list]
            img_list = torch.stack(img_list, dim=0) # B,3,H,W  
            all_img = img_list 
            output['tr_img'] = all_img

        return output


def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


def get_datasets(cfg, args):
    """
        cfg: config.data sub part 
    """
    if OVERFIT:
        random_subsample = 0
    else:
        random_subsample = cfg.random_subsample
    logger.info(f'get_datasets: tr_sample_size={cfg.tr_max_sample_points}, '
                f' te_sample_size={cfg.te_max_sample_points}; '
                f' random_subsample={random_subsample}'
                f' normalize_global={cfg.normalize_global}'
                f' normalize_std_per_axix={cfg.normalize_std_per_axis}'
                f' normalize_per_shape={cfg.normalize_per_shape}'
                f' recenter_per_shape={cfg.recenter_per_shape}'
                )
    kwargs = {}
    tr_dataset = ShapeNet15kPointClouds(
        categories=cfg.cates,
        split='train',
        tr_sample_size=cfg.tr_max_sample_points,
        te_sample_size=cfg.te_max_sample_points,
        sample_with_replacement=cfg.sample_with_replacement,
        scale=cfg.dataset_scale,  # root_dir=cfg.data_dir,
        normalize_shape_box=cfg.normalize_shape_box,
        normalize_per_shape=cfg.normalize_per_shape,
        normalize_std_per_axis=cfg.normalize_std_per_axis,
        normalize_global=cfg.normalize_global,
        recenter_per_shape=cfg.recenter_per_shape,
        random_subsample=random_subsample,
        clip_forge_enable=cfg.clip_forge_enable,
        clip_model=cfg.clip_model,
        **kwargs)

    eval_split = getattr(args, "eval_split", "val")
    # te_dataset has random_subsample as False, therefore not using sample_with_replacement
    te_dataset = ShapeNet15kPointClouds(
        categories=cfg.cates,
        split=eval_split,
        tr_sample_size=cfg.tr_max_sample_points,
        te_sample_size=cfg.te_max_sample_points,
        scale=cfg.dataset_scale,  # root_dir=cfg.data_dir,
        normalize_shape_box=cfg.normalize_shape_box,
        normalize_per_shape=cfg.normalize_per_shape,
        normalize_std_per_axis=cfg.normalize_std_per_axis,
        normalize_global=cfg.normalize_global,
        recenter_per_shape=cfg.recenter_per_shape,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
        clip_forge_enable=cfg.clip_forge_enable,
        clip_model=cfg.clip_model,
    )
    return tr_dataset, te_dataset


def get_data_loaders(cfg, args):
    tr_dataset, te_dataset = get_datasets(cfg, args)
    kwargs = {}
    if args.distributed:
        kwargs['sampler'] = data.distributed.DistributedSampler(
            tr_dataset, shuffle=True)
    else:
        kwargs['shuffle'] = True
    if args.eval_trainnll:
        kwargs['shuffle'] = False
    train_loader = data.DataLoader(dataset=tr_dataset,
                                   batch_size=cfg.batch_size,
                                   num_workers=cfg.num_workers,
                                   drop_last=cfg.train_drop_last == 1,
                                   pin_memory=False, **kwargs)
    test_loader = data.DataLoader(dataset=te_dataset,
                                  batch_size=cfg.batch_size_test,
                                  shuffle=False,
                                  num_workers=cfg.num_workers,
                                  pin_memory=False,
                                  drop_last=False,
                                  )
    logger.info(
        f'[Batch Size] train={cfg.batch_size}, test={cfg.batch_size_test}; drop-last={cfg.train_drop_last}')
    loaders = {
        "test_loader": test_loader,
        'train_loader': train_loader,
    }
    return loaders
