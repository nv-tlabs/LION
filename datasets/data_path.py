# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import os


def get_path(dataname=None):
    dataset_path = {}
    dataset_path['pointflow'] = [
        './data/ShapeNetCore.v2.PC15k/'

    ]
    dataset_path['clip_forge_image'] = [
            './data/shapenet_render/'
            ]

    if dataname is None:
        return dataset_path
    else:
        assert(
            dataname in dataset_path), f'not found {dataname}, only: {list(dataset_path.keys())}'
        for p in dataset_path[dataname]:
            print(f'searching: {dataname}, get: {p}')
            if os.path.exists(p):
                return p
        ValueError(
            f'all path not found for {dataname}, please double check: {dataset_path[dataname]}; or edit the datasets/data_path.py ')


def get_cache_path():
    cache_list = ['/workspace/data_cache_local/data_stat/',
                  '/workspace/data_cache/data_stat/']
    for p in cache_list:
        if os.path.exists(p):
            return p
    ValueError(
        f'all path not found for {cache_list}, please double check: or edit the datasets/data_path.py ')
