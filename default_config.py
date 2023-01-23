# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
# ---------------------------------------------------------------


from third_party.yacs_config import CfgNode as CN

cfg = CN()
cfg.dpm_ckpt = ''
cfg.clipforge = CN()
cfg.clipforge.clip_model = "ViT-B/32"
cfg.clipforge.enable = 0
cfg.clipforge.feat_dim = 512
cfg.eval_trainnll = 0
cfg.exp_name = ''
cfg.cmt = ''
cfg.hash = ''
cfg.ngpu = 1
cfg.snapshot_min = 30  # snapshot every 30 min
cfg.bash_name = ''
cfg.set_detect_anomaly = 0
cfg.weight_recont = 1.0
# vae ckpt
# lns
cfg.use_checkpoint = 0
cfg.num_val_samples = 16  # 24 #12

# config for pointtransformer
cfg.eval = CN()
cfg.eval.need_denoise = 0
cfg.eval.load_other_vae_ckpt = 0
cfg.register_deprecated_key('eval.other_vae_ckpt_path')
cfg.vis_latent_point = 0
cfg.latent_pts = CN()
#cfg.latent_pts.class_embed_layer = ''
cfg.register_deprecated_key('latent_pts.class_embed_layer')
cfg.latent_pts.style_dim = 128  # dim of global style latent variable
cfg.register_deprecated_key('latent_pts.perturb_input')
cfg.register_deprecated_key('latent_pts.perturb_input_scale')
cfg.register_deprecated_key('latent_pts.outlier_input')

# scale of init weights for the mlp in adaGN layer
cfg.latent_pts.ada_mlp_init_scale = 1.0
# models.latent_points_ada.StyleMLP' # style mlp layers
cfg.latent_pts.style_mlp = ''
cfg.latent_pts.pts_sigma_offset = 0.0
cfg.latent_pts.skip_weight = 0.1
cfg.latent_pts.encoder_layer_out_dim = 32
cfg.latent_pts.decoder_layer_out_dim = 32
cfg.register_deprecated_key('latent_pts.encoder_nneighbor')
cfg.register_deprecated_key('latent_pts.decoder_nneighbor')
cfg.latent_pts.style_prior = 'models.score_sde.resnet.PriorSEDrop'
cfg.latent_pts.mask_out_extra_latent = 0  # use only latent coordinates
# latent coordinates directly same as input (not using the decoder and encoder)
cfg.register_deprecated_key('latent_pts.latent_as_pts')

cfg.latent_pts.normalization = 'bn'  # BatchNorm or LayerNorm
cfg.latent_pts.pvd_mse_loss = 0
cfg.latent_pts.hid = 64

cfg.register_deprecated_key('latent_pts.knn')
cfg.register_deprecated_key('latent_pts.n5layer')
cfg.register_deprecated_key('latent_pts.dgcnn_last_hid')

cfg.latent_pts.latent_dim_ext = [64]  # the global latent dim
cfg.latent_pts.weight_kl_pt = 1.0  # kl ratio of the pts
cfg.latent_pts.weight_kl_feat = 1.0  # kl ratio of the latent feat
cfg.latent_pts.weight_kl_glb = 1.0  # kl ratio of the latent feat
# kl ratio of the latent feat
cfg.latent_pts.style_encoder = 'models.shapelatent_modules.PointNetPlusEncoder'
cfg.latent_pts.use_linear_for_adagn = 0
# cfg.latent_pts.weight_kl_glb = 1.0 # kl ratio of the global latent

# shapelatent:
cfg.has_shapelatent = 1 
cfg.shapelatent = CN()
cfg.shapelatent.local_emb_agg = 'mean'
cfg.shapelatent.freeze_vae = 0  # learn vae
cfg.shapelatent.eps_z_global_only = 1
cfg.shapelatent.model = 'flow'
cfg.shapelatent.residual = 1
cfg.shapelatent.encoder_type = 'pointnet'
cfg.shapelatent.prior_type = 'flow'
cfg.shapelatent.decoder_type = 'PointwiseNet'
cfg.shapelatent.loss0_weight = 1.0
cfg.shapelatent.latent_dim = 256
cfg.shapelatent.kl_weight = 1e-3
cfg.shapelatent.decoder_num_points = -1
# offset the sigma towards zero for better init, will use the log_sigma - offset value, better to be positive s.t. - offset < 0 since we'd like to push it towards 0; exp(-0.1)=0.9, exp(-0.8)=0.44, exp(-1)=0.3, exp(-10)=4e-5
cfg.shapelatent.log_sigma_offset = 0.0

cfg.sde = CN()
cfg.sde.ode_sample = 0 #1
# train the prior or not, default is 1, only when we do voxel2pts, will freeze prior
cfg.sde.train_dae = 1
cfg.sde.init_t = 1.0  # start from time = 1.0
cfg.sde.nhead = 4  # number of head in transformder: multi-head attention layer
cfg.sde.local_prior = 'same_as_global'  # architecture for local prior
cfg.sde.drop_inactive_var = 0
cfg.sde.learn_mixing_logit = 1  # freeze it
cfg.sde.regularize_mlogit_margin = 0.0
cfg.sde.share_mlogit = 0  # use same mlogit for all latent variables
cfg.sde.hypara_mixing_logit = 0  # set as hyper-parameter and freeze it?
cfg.sde.bound_mlogit = 0  # clamp or not
cfg.sde.bound_mlogit_value = -5.42  # clamp the max value
cfg.sde.regularize_mlogit = 0  # set the sum of sigmoid(mlogit) as one loss
cfg.sde.attn_mhead = 0  # use multi-head attention in prior model
cfg.sde.attn_mhead_local = -1  # use multi-head attention in prior model
cfg.sde.pos_embed = 'none'
cfg.sde.hier_prior = 0
cfg.sde.is_continues = 0
cfg.sde.time_emb_scales = 1.0  # -> 1k?
cfg.sde.time_eps = 1e-2
cfg.sde.ode_eps = 1e-5  # cut off for ode sampling
cfg.sde.sde_type = 'vpsde'  # vada
cfg.sde.sigma2_0 = 0.0
cfg.sde.sigma2_max = 0.99
cfg.sde.sigma2_min = 1e-4
cfg.sde.beta_start = 0.1  # 1e-4 * 1e3
cfg.sde.beta_end = 20.0  # 1e-2 * 1e3
# sampling, always iw # ll: small times; 'll_uniform'  # -> ll_iw
cfg.sde.iw_sample_p = 'll_iw'
# drop_all_iw / drop_sigma2t_iw
cfg.sde.iw_subvp_like_vp_sde = False
cfg.sde.prior_model = 'models.latent_points_ada_localprior.PVCNN2Prior'

# -- to train diffusion in latent space -- #
cfg.sde.update_q_ema = False
cfg.sde.iw_sample_q = 'reweight_p_samples'
# ll_iw / reweight_p_samples
cfg.sde.kl_anneal_portion_vada = 0.1
cfg.sde.kl_const_portion_vada = 0.0
cfg.sde.kl_const_coeff_vada = 0.7
cfg.sde.kl_balance_vada = False
cfg.sde.grad_clip_max_norm = 0.0
cfg.sde.cont_kl_anneal = True
# False
cfg.sde.mixing_logit_init = -6
cfg.sde.weight_decay_norm_vae = 0.0 #1e-2
cfg.sde.weight_decay_norm_dae = 0.0 #1e-2
# -> 0, for sn calculator
cfg.sde.train_vae = True
cfg.sde.jac_reg_coeff = 0
cfg.sde.jac_reg_freq = 1
cfg.sde.kin_reg_coeff = 0
cfg.sde.learning_rate_mlogit = -1.0
cfg.sde.learning_rate_dae_local = 3e-4
cfg.sde.learning_rate_min_dae_local = 3e-4
cfg.sde.learning_rate_dae = 3e-4
cfg.sde.learning_rate_min_dae = 3e-4
cfg.sde.learning_rate_min_vae = 1e-5
cfg.sde.learning_rate_vae = 1e-4
cfg.sde.epochs = 800
cfg.sde.warmup_epochs = 20
cfg.sde.weight_decay = 3e-4
cfg.sde.use_adamax = False
cfg.sde.use_adam = True # False
cfg.sde.mixed_prediction = False # True
cfg.sde.vae_checkpoint = ''
cfg.sde.dae_checkpoint = ''
# will be used to multiply with the t value, if ode solver, use 1k, if discrete solver, use 1.0
cfg.sde.embedding_scale = 1.0 # 1000.0
cfg.sde.embedding_type = 'positional'
cfg.sde.train_ode_solver_tol = 1e-5
cfg.sde.num_scales_dae = 2
cfg.sde.autocast_train = False
cfg.sde.diffusion_steps = 1000
cfg.sde.embedding_dim = 128
cfg.sde.num_channels_dae = 256
cfg.sde.num_cell_per_scale_dae = 8
cfg.sde.num_cell_per_scale_dae_local = 0
cfg.sde.dropout = 0.2
cfg.sde.num_preprocess_blocks = 2
cfg.sde.num_latent_scales = 1
cfg.sde.fir = False
cfg.sde.progressive = 'none'
cfg.sde.progressive_input = 'none'
cfg.sde.progressive_combine = 'sum'
cfg.sde.dataset = 'shape'
cfg.sde.denoising_stddevs = 'beta'
cfg.sde.ema_decay = 0.9999
# cfg.sde.is_train_vae=True
cfg.register_deprecated_key("sde.is_train_vae")
cfg.sde.kl_max_coeff_vada = 1.0
# conditional prior input
cfg.sde.condition_add = 1
cfg.sde.condition_cat = 0
cfg.sde.global_prior_ckpt = ''  # checkpoint for global prior component
cfg.sde.pool_feat_cat = 0  # the local prior aggregate the feat as extra input channels

# hyperparameter of ddim sampling
cfg.sde.ddim_skip_type = 'uniform'
cfg.sde.ddim_kappa = 1.0  # 1.0: fully ddpm sampling; 0: ode style sampling

cfg.ddpm = CN()
cfg.ddpm.use_p2_weight = 0
cfg.ddpm.p2_k = 1.0
cfg.ddpm.p2_gamma = 1.0
cfg.ddpm.use_new_timeemb = 0
cfg.ddpm.input_dim = 3
cfg.ddpm.dropout = 0.1
cfg.ddpm.num_layers_classifier = 3
cfg.ddpm.use_bn = True
cfg.ddpm.add_point_feat = True
cfg.ddpm.use_gn = False
cfg.ddpm.time_dim = 64
cfg.ddpm.ema = 1
cfg.ddpm.with_se = 0
cfg.ddpm.use_global_attn = 0
cfg.ddpm.num_steps = 1000
cfg.ddpm.beta_1 = 1e-4
cfg.ddpm.beta_T = 2e-2
# ['linear', 'customer'] 'customer' for airplane in PVD
cfg.ddpm.sched_mode = 'linear'
cfg.ddpm.model_var_type = 'fixedlarge'
# define architecture:
cfg.register_deprecated_key("ddpm.pointnet_plus")
cfg.register_deprecated_key("ddpm.pointnet_pp")
cfg.register_deprecated_key("ddpm.pointnet_luo")
# end define architecture
#cfg.ddpm.use_pvc = 1
cfg.register_deprecated_key("ddpm.use_pvc")
cfg.ddpm.clip_denoised = 0
cfg.ddpm.model_mean_type = 'eps'
cfg.ddpm.loss_type = 'mse'
cfg.ddpm.loss_type_0 = ''
cfg.ddpm.loss_weight_emd = 0.02
cfg.ddpm.loss_weight_cdnorm = 1.0
cfg.ddpm.attn = [0, 1, 0, 0]
cfg.ddpm.ncenter = [1024, 256, 64, 16]

#cfg.ddpm.pvc = CN()
#cfg.ddpm.pvc.use_small_model = 0
#cfg.ddpm.pvc.mlp_after_pvc = 0
cfg.register_deprecated_key("ddpm.pvc")
cfg.register_deprecated_key("ddpm.pvc.use_small_model")
cfg.register_deprecated_key("ddpm.pvc.mlp_after_pvc")

cfg.ddpm.ddim_step = 200

cfg.data = CN()
cfg.data.nclass = 55
cfg.data.cond_on_cat = 0
cfg.data.cond_on_voxel = 0
cfg.data.eval_test_split = 0  # eval loader will be using test split
cfg.data.voxel_size = 0.1  # size of voxel for voxel_datasets.py
cfg.data.noise_std = 0.1  # std for the noise added to the input data
cfg.data.noise_type = 'normal'  # std for the noise added to the input data
cfg.data.noise_std_min = -1.0  # for range of noise std
cfg.data.clip_forge_enable = 0
cfg.data.clip_model = 'ViT-B/32'
cfg.data.type = "datasets.pointflow_datasets"
# datasets/neuralspline_datasets datasets/shape_curvature
cfg.data.dataset_type = "shapenet15k"
cfg.data.num_workers = 12  # 8
cfg.data.train_drop_last = 1  # drop_last for train data loader
cfg.data.cates = 'chair'  # data category
cfg.data.tr_max_sample_points = 2048
cfg.data.te_max_sample_points = 2048
cfg.data.data_dir = "data/ShapeNetCore.v2.PC15k"  # depreciated
cfg.data.batch_size = 12
cfg.data.batch_size_test = 10
cfg.data.dataset_scale = 1
# -- the following option in terms of normalization should turn into string -- #
cfg.data.normalize_per_shape = False
cfg.data.normalize_shape_box = False
cfg.data.normalize_global = False
cfg.data.normalize_std_per_axis = False
cfg.data.normalize_range = False  # not used
cfg.data.recenter_per_shape = True
# -- for the normal prediction model, used in folder_datasets
cfg.register_deprecated_key('data.load_point_stat')
cfg.register_deprecated_key('data.is_load_pointflow2NS')
cfg.register_deprecated_key('data.data_path')

#
cfg.data.sample_with_replacement = 1
# fixed the  data.tr_max_sample_points $np data.te_max_sample_points $np2048 points of the first 15k points
cfg.data.random_subsample = 1
# the data dim, used in dataset worker, if -1, it will be the same as ddpm.input_dim
cfg.data.input_dim = -1
cfg.data.is_encode_whole_dataset_trainer = 0
cfg.register_deprecated_key('data.augment')
cfg.register_deprecated_key('data.aug_translate')
cfg.register_deprecated_key('data.aug_scale')
cfg.register_deprecated_key('data.sub_train_set')

cfg.test_size = 660

cfg.viz = CN()
cfg.viz.log_freq = 10
cfg.viz.viz_freq = 400
cfg.viz.save_freq = 200
cfg.viz.val_freq = -1
cfg.viz.viz_order = [2, 0, 1]
cfg.viz.vis_sample_ddim_step = 0

cfg.trainer = CN()
# when loss 1 is weighted, also weight the kl terms
cfg.trainer.apply_loss_weight_1_kl = 0
cfg.trainer.kl_free = [0, 0]  # the value for the threshold
# not back ward kl loss if KL value is smaller than the threshold
cfg.trainer.use_kl_free = 0
cfg.trainer.type = "trainers.ddpm_trainer"  # it means dist trainer
cfg.trainer.epochs = 10000
cfg.trainer.warmup_epochs = 0
cfg.trainer.seed = 1
cfg.trainer.use_grad_scalar = 0
cfg.trainer.opt = CN()
cfg.trainer.opt.type = 'adam'
cfg.trainer.opt.lr = 1e-4  # use bs*1e-5/8
cfg.trainer.opt.lr_min = 1e-4  # use bs*1e-5/8
# lr start to anneal after ratio of epochs; used in cosine and lambda lr scheduler
cfg.trainer.opt.start_ratio = 0.6
cfg.trainer.opt.beta1 = 0.9
cfg.trainer.opt.beta2 = 0.999
cfg.trainer.opt.momentum = 0.9  # for SGD
cfg.trainer.opt.weight_decay = 0.
cfg.trainer.opt.ema_decay = 0.9999
cfg.trainer.opt.grad_clip = -1.
cfg.trainer.opt.scheduler = ''
cfg.trainer.opt.step_decay = 0.998
cfg.trainer.opt.vae_lr_warmup_epochs = 0
cfg.trainer.anneal_kl = 0
cfg.trainer.kl_balance = 0
cfg.trainer.rec_balance = 0
cfg.trainer.loss1_weight_anneal_v = 'quad'
cfg.trainer.kl_ratio = [1.0, 1.0]
cfg.trainer.kl_ratio_apply = 0  # apply the fixed kl ratio in the kl_ratio list
# using spectral norm regularization on vae training or not (used in hvae_trainer)
cfg.trainer.sn_reg_vae = 0
cfg.trainer.sn_reg_vae_weight = 0.0  # loss weight for the sn regulatrization

# [start] set in runtime
cfg.log_name = ''
cfg.save_dir = ''
cfg.log_dir = ''
cfg.comet_key = ''
# [end]

cfg.voxel2pts = CN()
cfg.voxel2pts.init_weight = ''
cfg.voxel2pts.diffusion_steps = [0]

cfg.dpm = CN()
cfg.dpm.train_encoder_only = 0
cfg.num_ref = 0  # manully set the number of reference
cfg.eval_ddim_step = 0  # ddim sampling for the model evaluation
cfg.model_config = ''  # used for model control, without ading new flag

## --- depreciated --- #
cfg.register_deprecated_key('cls')  # CN()
cfg.register_deprecated_key('cls.classifier_type')  # 'models.classifier.OneLayer'
cfg.register_deprecated_key('cls.train_on_eps')  # 1
cfg.register_deprecated_key('cond_prior')  # CN()
cfg.register_deprecated_key('cond_prior.grid_emb_resolution')  # 32
cfg.register_deprecated_key('cond_prior.emb_dim')  # 64
cfg.register_deprecated_key('cond_prior.use_voxel_feat')  # 1
cfg.register_deprecated_key('cond_encoder_prior')  # 'models.shapelatent_modules.VoxelGridEncoder'
cfg.register_deprecated_key('cond_prior.pvcconv_concat_3d_feat_input')  # 0
cfg.register_deprecated_key('generate_mode_global')  # 'interpolate'
cfg.register_deprecated_key('generate_mode_local')  # 'freeze'
cfg.register_deprecated_key('normals')  # CN()
cfg.register_deprecated_key('normals.model_type')  # ''
cfg.register_deprecated_key('save_sample_seq_and_quit')  # 0
cfg.register_deprecated_key('lns_loss_weight')  # 1.0
cfg.register_deprecated_key('normal_pred_checkpoint')  # ''
cfg.register_deprecated_key('lns')  # CN()
cfg.register_deprecated_key('lns.override_config')  # ''
cfg.register_deprecated_key('lns.wandb_checkpoint')  # 'nvidia-toronto/generative_chairs/3m3gc6sz/checkpoint-171.pth'
cfg.register_deprecated_key('lns.num_input_points')  # 1000
cfg.register_deprecated_key('lns.num_simulate')  # 20
cfg.register_deprecated_key('lns.split_simulate')  # 'train'
# use mesh-trainer or not
cfg.register_deprecated_key('with_lns')  # 0

cfg.register_deprecated_key('normal_predictor_yaml')  # ''

cfg.register_deprecated_key('pointtransformer')  # CN()
# number of attention layer in each block
cfg.register_deprecated_key('pointtransformer.blocks')  # [2, 3, 4, 6, 3]
cfg.register_deprecated_key('shapelatent.refiner_bp')  # 1  # bp gradient to the local-decoder or not
cfg.register_deprecated_key('shapelatent.loss_weight_refiner')  # 1.0  # weighted loss for the refiner
cfg.register_deprecated_key('shapelatent.refiner_type')  # 'models.pvcnn2.PVCNN2BaseAPI'  # mode for the refiner

cfg.register_deprecated_key('shapelatent.encoder_weight_std')  # 0.1
cfg.register_deprecated_key('shapelatent.encoder_weight_norm')  # 0
cfg.register_deprecated_key('shapelatent.encoder_weight_uniform')  # 1
cfg.register_deprecated_key('shapelatent.key_point_gen')  # 'mlps'
cfg.register_deprecated_key('shapelatent.add_sub_loss')  # 1  # not used
cfg.register_deprecated_key('shapelatent.local_decoder_type')  # ''
cfg.register_deprecated_key('shapelatent.local_decoder_type_1')  # ''
cfg.register_deprecated_key('shapelatent.local_encoder_ball_radius')  # 0.8
cfg.register_deprecated_key('shapelatent.local_encoder_ap_ball_radius')  # 1.0
cfg.register_deprecated_key('shapelatent.local_encoder_type')  # ''
cfg.register_deprecated_key('shapelatent.local_encoder_type_1')  # ''
cfg.register_deprecated_key('shapelatent.local_loss_weight_max')  # 50
cfg.register_deprecated_key('shapelatent.num_neighbors')  # 0
cfg.register_deprecated_key('shapelatent.extra_centers')  # []
# for latent model is flow
cfg.register_deprecated_key('shapelatent.latent_flow_depth')  # 14
cfg.register_deprecated_key('shapelatent.latent_flow_hidden_dim')  # 256
cfg.register_deprecated_key('shapelatent.bp_to_l0')  # True
cfg.register_deprecated_key('shapelatent.global_only_epochs')  # 0
cfg.register_deprecated_key('shapelatent.center_local_points')  # 1
cfg.register_deprecated_key('shapelatent.hvae')  # CN()
# alternatively way to compute the local loss
cfg.register_deprecated_key('shapelatent.hvae.loss_wrt_ori')  # 0
# add voxel feature to the latent space; the decoder require pvc conv or query
cfg.register_deprecated_key('shapelatent.add_voxel2z_global')  # 0
# reuse the encoder to get local latent
cfg.register_deprecated_key('shapelatent.query_output_local_from_enc')  # 0
# check models/shapelatent_modules where the feature will be saved as a dict
cfg.register_deprecated_key('shapelatent.query_local_feat_layer')  # 'inter_voxelfeat_0'
# need to check the sa_blocks of the global encoder
cfg.register_deprecated_key('shapelatent.query_local_feat_dim')  # 32
# reuse the encoder to get local latent
cfg.register_deprecated_key('shapelatent.query_center_emd_from_enc')  # 0  # reuse the encoder for center emd
cfg.register_deprecated_key('shapelatent.prog_dec_gf')  # 8  # grow_factor in VaniDecoderProg
cfg.register_deprecated_key('shapelatent.prog_dec_gf_list')  # [0, 0]  # grow_factor in VaniDecoderProg
cfg.register_deprecated_key('shapelatent.prog_dec_ne')  # 2  # num_expand in VaniDecoderProg
# increase number hirach, used by hvaemul model
cfg.register_deprecated_key('shapelatent.num_neighbors_per_level')  # [64]  # number of neighbors for each level
cfg.register_deprecated_key('shapelatent.num_level')  # 1  # number of hierarchi latent space (local)
cfg.register_deprecated_key('shapelatent.x0_target_fps')  # 0  # let the target of global output as the
cfg.register_deprecated_key('shapelatent.downsample_input_ratio')  # 1.0
# whether taking other tensor as input to local-encoder of not
cfg.register_deprecated_key('shapelatent.local_enc_input')  # 'sim'
# local encoder take z0 as input at which location
cfg.register_deprecated_key('shapelatent.local_encoder_condition_z0')  # ''
# output the absolution coordinates or the offset w.r.t centers
cfg.register_deprecated_key('shapelatent.local_decoder_output_offset')  # 0
# feed coords of keypoints to the local prior model
cfg.register_deprecated_key('shapelatent.local_prior_need_coords')  # 0

# add the time embedding tensor to each encoder layer instead of add to first layer only
cfg.register_deprecated_key('sde.transformer_temb2interlayer')  # 0
# normalization used in transformer encoder;
cfg.register_deprecated_key('sde.transformer_norm_type')  # 'layer_norm'
cfg.register_deprecated_key('data.has_normal')  # 0  # for datasets/pointflow_rgb.py only
cfg.register_deprecated_key('data.has_color')  # 0  # for datasets/pointflow_rgb.py only
cfg.register_deprecated_key('data.cls_data_ratio')  # 1.0  # ratio of the training data
cfg.register_deprecated_key('data.sample_curvature')  # 0  # only for datasets/shape_curvature
cfg.register_deprecated_key('data.ratio_c')  # 1.0  # only for datasets/shape_curvature
