# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import torch 

def CHECKDIM(tensor, dim, val): 
    if type(tensor) == list: 
        for t in tensor: 
            CHECKDIM(t, dim, val) 
    else: 
        assert(len(tensor.shape) >= dim), 'expect {} to have {} dim shape {}'.format(tensor.shape, dim, val)
        if type(val) is list: 
            assert(tensor.shape[dim] in val), 'expect {} to have {} dim shape {}'.format(
                    tensor.shape, dim, val)
        else:
            assert(tensor.shape[dim] == val), 'expect tensor with shape: {} having dim {} as {}'.format(
                    tensor.shape, dim, val)

    return True

def CHECK5D(tensor, *args):
    assert(len(tensor.shape) == 5), 'get {} {}'.format(tensor.shape, len(tensor.shape))
    for t in args:
        CHECK5D(t)
    return tensor.shape

def CHECK3D(tensor, *args):
    assert(len(tensor.shape) == 3), 'get {} {}'.format(tensor.shape, len(tensor.shape))
    for t in args:
        CHECK3D(t)
    return tensor.shape

def CHECK4D(tensor):
    assert(len(tensor.shape) == 4), 'get {} {}'.format(tensor.shape, len(tensor.shape))  
    return tensor.shape 
def CHECKND(tensor, N):
    assert(len(tensor.shape) == N), 'get tensor shape:{} DIM={}, expect:{}'.format(tensor.shape, len(tensor.shape), N)  
    return tensor.shape 

def CHECK2D(tensor):
    assert(len(tensor.shape) == 2), 'get {} {}'.format(tensor.shape, len(tensor.shape))  
    return tensor.shape 

def CHECK_N3or6(input):
    # expect input in shape (N,3) or (N,6)
    CHECK_TENSOR(input)
    CHECK2D(input)
    assert(input.shape[1] == 3 or input.shape[1] == 6), f'expect shape N,3 or N,6; get {input.shape}'
    return input.shape 

def CHECK_N3or6or9(input):
    # expect input in shape (N,3) or (N,6)
    CHECK_TENSOR(input)
    CHECK2D(input)
    assert(input.shape[1] == 3 or input.shape[1] == 6 or input.shape[1] == 9), f'expect shape N,3 or N,6; get {input.shape}'
    return input.shape 

def CHECK_N3(input):
    # expect input in shape (N,3)
    CHECK_TENSOR(input)
    CHECK2D(input)
    CHECKDIM(input, dim=1, val=3)
    return input.shape 

def CHECK_TENSOR(input):
    assert(torch.is_tensor(input)), f'expect tensor, get {type(input)}'

def CHECKEQ(a, b):
    assert(a == b), f'expect a=b, get a={a} and b={b}' 

def CHECKSIZE(t, values):
    CHECKND(t, len(values)) 
    for iv, vv in enumerate(values):
        CHECKDIM(t, iv, vv)
def CHECKSAMESIZE(t1, t2):
    CHECKSIZE(t1, t2.shape)
