#!/bin/bash

# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
# ---------------------------------------------------------------

docker="nvcr.io/nvidian/lion_env:0"
echo $PWD
exit
code_path=$PWD
docker run --gpus all -p 8081:8081 --ipc=host -v $code_path:$code_path -it $docker bash

