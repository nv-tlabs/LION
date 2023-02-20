# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
# ---------------------------------------------------------------

FROM nvcr.io/nvidia/pytorch:21.12-py3
ARG PYTHON_VERSION=3.8
ARG WITH_TORCHVISION=1
RUN python -m pip uninstall torch -y
RUN python -m pip uninstall torchvision -y
RUN python -m pip uninstall torchaudio -y
RUN python -m pip uninstall torchtext -y
RUN python -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN apt-get update
RUN apt-get install -y build-essential zsh tmux wget
RUN apt-get install -y libgl1-mesa-glx libusb-1.0-0-dev
RUN apt purge --auto-remove cmake
RUN python -m pip install cmake
RUN python -m pip install scipy
#RUN python -m pip install pykeops
RUN python -m pip install h5py
RUN python -m pip install wandb
RUN python -m pip install --ignore-installed PyYAML
#RUN python -m pip install open3d
RUN python -m pip install cupy-cuda111
RUN python -m pip install nestargs
RUN python -m pip install flatten_dict
#RUN python -m pip uninstall point_cloud_utils
#RUN python -m pip install git+https://github.com/fwilliams/point-cloud-utils.git
RUN conda install -y scikit-image
RUN apt-get update
#RUN python -m pip uninstall torch-scatter
RUN export PATH=/usr/local/cuda-11.1/bin:$PATH
RUN export CPATH=/usr/local/cuda-11.1/include:$CPATH
RUN export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
RUN export DYLD_LIBRARY_PATH=/usr/local/cuda-11.1/lib:$DYLD_LIBRARY_PATH

RUN python -m pip install git+https://github.com/openai/CLIP.git
RUN python -m pip install loguru einops comet_ml calmsize diffusers

#COPY environment.yml .
#COPY activate.sh .
#
#RUN apt -y update && apt -y install wget curl vim libgcrypt20 coreutils libgl1-mesa-glx
#
#RUN wget https://repo.continuum.io/miniconda/Miniconda3-3.7.0-Linux-x86_64.sh -O ./miniconda.sh
#
#RUN chmod ouga+xw ./miniconda.sh
#RUN bash ./miniconda.sh -b -p ./miniconda
#
#ENV PATH /opt/pytorch/miniconda/bin:$PATH
#
#RUN conda update conda
#RUN conda env create -f environment.yml
## RUN /bin/bash -c "source activate pvcnn_pyt13" 
#RUN conda activate pvcnn_pyt13 
#RUN pip install einops 
#RUN pip install sklearn 
#RUN pip install tensorflow 
