# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import sys
sys.path.append('.')
from utils.eval_helper import compute_score 
# samples = sys.argv[1]
# ref = sys.argv[2]

samples = './lion_ckpt/unconditional/car/samples.pt'
ref = './datasets/test_data/ref_val_car.pt'
compute_score(samples, ref_name=ref, norm_box=False)
"""
will get: 
[Test] MinMatDis | CD 0.000913 | EMD 0.007523
[Test] Coverage | CD 0.500000 | EMD 0.565341
[Test] 1NN-Accur | CD 0.534091 | EMD 0.511364
[Test] JsnShnDis | 0.009229 
"""

samples = './lion_ckpt/unconditional/chair/samples.pt'
ref = './datasets/test_data/ref_val_chair.pt'
compute_score(samples, ref_name=ref, norm_box=False)
"""
[Test] MinMatDis | CD 0.002643 | EMD 0.015516
[Test] Coverage | CD 0.489426 | EMD 0.521148
[Test] 1NN-Accur | CD 0.537009 | EMD 0.523414
[Test] JsnShnDis | 0.013535
"""

samples = './lion_ckpt/unconditional/chair/samples.pt'
ref = './datasets/test_data/ref_val_chair.pt'
compute_score(samples, ref_name=ref, norm_box=False)
"""
[Test] MinMatDis | CD 0.000221 | EMD 0.003706
[Test] Coverage | CD 0.471605 | EMD 0.496296
[Test] 1NN-Accur | CD 0.674074 | EMD 0.612346
[Test] JsnShnDis | 0.060703 
"""
