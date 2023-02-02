""" to train VAE-encoder with two prior """
import os
import time 
import random 
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from loguru import logger
from torch import optim
from utils.ema import EMA
from utils import eval_helper, exp_helper 
from utils.sr_utils import SpectralNormCalculator
from utils.checker import *
from utils.utils import AvgrageMeter 
from utils import utils
from torch.optim import Adam as FusedAdam
from torch.cuda.amp import autocast, GradScaler
from trainers.train_2prior import Trainer as BaseTrainer 
from utils.data_helper import normalize_point_clouds 
from utils.vis_helper import visualize_point_clouds_3d
from utils.eval_helper import compute_NLL_metric 
CHECKPOINT = int(os.environ.get('CHECKPOINT', 0))
EVAL_LNS = int(os.environ.get('EVAL_LNS', 0))
def interpolate_noise(noise):
    noise_a = noise[0].contiguous() # 1,D,1,1   
    noise_b = noise[-1].contiguous() # 1,D,1,1   
    num_inter = noise.shape[0] - 2 
    for k in range(1, noise.shape[0]-1):
        p = float(k) / len(noise) # 1/8 to 7/8 
        ## logger.info('p={}; eps: {}', p, noise.shape)
        # noise[k] = p * noise_b + (1-p) * noise_a 
        noise[k] = np.sqrt(p) * noise_b + np.sqrt(1-p) * noise_a  
    return noise 
#def get_data(num_points=3072, num_selected=50):
#    import h5py 
#    import torch 
#    import sys 
#    import os 
#    from datasets.data_path import get_path 
#    
#    synsetid_to_cate = {
#        "02691156": "airplane",
#        "02828884": "bench",
#        "02958343": "car",
#        "03001627": "chair",
#        "03211117": "display",
#        "03636649": "lamp",
#        "04256520": "sofa",
#        "04379243": "table",
#        "04530566": "watercraft",
#    }
#
#
#    root_dir = get_path('pointflow') 
#    out_dict = [] 
#    out_name = [] 
#    for synset_id in synsetid_to_cate.keys():
#        filename = os.path.join(root_dir, f'{synset_id}', 'train')
#        tr_out_full = []
#        for file in sorted(os.listdir(filename))[:10]:
#            pts = np.load(filename + '/' + file)
#            tr_out_full.append(pts)
#        tr_out_full = np.stack(tr_out_full)
#        #with h5py.File(filename, "r") as h5f:
#        #    tr_out_full = h5f['surface_points/points'][5:5+num_selected]  
#        if tr_out_full.shape[1] > num_points:
#            pt_cur = []
#            for b in range(tr_out_full.shape[0]):
#                tr_idxs = np.random.choice(tr_out_full.shape[1], num_points)
#                pt_cur.append(tr_out_full[b, tr_idxs]) ## pt_cur[tr_idxs]
#            tr_out_full = torch.from_numpy(np.stack(pt_cur))
#        # out_dict[synsetid_to_cate[synset_id]] = tr_out_full 
#        out_dict.append(tr_out_full) 
#        out_name.extend([synsetid_to_cate[synset_id]] * num_selected)
#    out_dict = torch.cat(out_dict) 
#    logger.info('created data: {}, ', out_dict.shape)
#    return out_dict, out_name 
#
#def fun_hash(a, b):
#    if a < b:
#        return f'{a}-{b}'
#    else:
#        return f'{b}-{a}'
#
#def fun_select_count_pair(names, a, b):
#    select_all_pair = [] 
#    indexes = list( np.arange(len(names)) ) 
#    indexes_a = [i for ii, i in enumerate(indexes) if names[ii] == a] 
#    indexes_b = [i for ii, i in enumerate(indexes) if names[ii] == b]
#    hash_d = [] 
#    logger.info('select paits for: {}, {}', a, b) 
#    for ai in indexes_a:
#        for bi in indexes_b: 
#            if ai != bi and fun_hash(ai, bi) not in hash_d:
#                hash_d.append(fun_hash(ai, bi)) 
#                select_all_pair.append([ai, bi])
#    logger.info('get pairs: {}', len(select_all_pair))
#    return select_all_pair 
#
class Trainer(BaseTrainer):
    is_diffusion = 0 
    generate_mode_global = 'interpolate'
    generate_mode_local = 'interpolate'
    def __init__(self, cfg, args):
        """
        Args:
            cfg: training config 
            args: used for distributed training 
        """
        super().__init__(cfg, args)
        self.draw_sample_when_vis = 0 

    @torch.no_grad()
    def eval_sample(self, step=0):
        pass # do nothing 

    # -- shared method for all model with vae component -- # 
    @torch.no_grad() 
    def eval_nll(self, step, ntest=None, save_file=False):
        cfg = self.cfg 
        if cfg.ddpm.ema: self.swap_vae_param_if_need()
        self.dae_optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        args = self.args 
        device = torch.device('cuda:%d'%args.local_rank)
        tag = exp_helper.get_evalname(self.cfg) + 'D%d'%self.cfg.voxel2pts.diffusion_steps[0] 
        ode_sample = self.cfg.sde.ode_sample 
        diffusion = self.diffusion_cont if ode_sample else self.diffusion_disc 
        eval_trainnll = self.cfg.eval_trainnll  
        data_loader = self.train_loader 
        #data_loader = self.test_loader 
        gen_pcs, ref_pcs = [], []
        gen_x_lns_list = []
        tag_cur = self.cfg.voxel2pts.diffusion_steps[0]
        num_selected = 60 
        #input_pts, input_names = get_data(self.num_points, num_selected)

        #is_all_class_model = 'nsall' in self.cfg.save_dir
        #logger.info('is_all_class_model: {}', is_all_class_model)
        #if is_all_class_model: 
        #    select_count_pair = [] 
        #    class_pairs = [
        #        ['airplane', 'car'],
        #        #['car', 'watercraft'],
        #        #['watercraft', 'airplane']
        #        #['airplane', 'bench'],
        #        #['bench', 'table'],
        #        #['table', 'chair']
        #        #['chair', 'display'],
        #        ##['chair', 'lamp'],
        #        ##['lamp', 'watercraft']
        #            ]

        #    for class_a, class_b in class_pairs:
        #        select_count_pair_cur = fun_select_count_pair(input_names, class_a, class_b) 
        #        select_count_pair.extend(select_count_pair_cur) 

        #else:
        #    class_a = class_b = self.cfg.data.cates 
        #    select_count_pair = fun_select_count_pair(input_names, class_a, class_b) 

        #input_pts_list = [] 
        #input_pts_cates = [] 
        #print('number of pair', len(select_count_pair))
        #for select_count in select_count_pair:
        #    input_pts_list_pair = [] 
        #    input_pts_cates_pair = [] 
        #    for vi in select_count:
        #        input_pts_list_pair.append(input_pts[vi][None])
        #        input_pts_cates_pair.append(input_names[vi])
        #    input_pts_list.append(torch.cat(input_pts_list_pair, dim=0))
        #    input_pts_cates.append('-'.join(input_pts_cates_pair)) 

        ##shuffle_idx = list(range(len(input_pts_list)))
        ##random.Random(38383).shuffle(shuffle_idx)
        ##input_pts_list = [input_pts_list[i] for i in shuffle_idx][:50] # select first 50 pairs 
        ##input_pts_cates = [input_pts_cates[i] for i in shuffle_idx][:50] # select first 50 pairs 
        #logger.info('num of input pts: {}, cates: {}', 
        #        len(input_pts_list), input_pts_cates[:10])
        output_dir_template = self.cfg.save_dir + '/enc%d_%s_%s/'%(num_selected, self.generate_mode_global, self.generate_mode_local) 
        vis_output_dir = self.cfg.save_dir + '/vis_enc%d_%s_%s/'%(num_selected, self.generate_mode_global, self.generate_mode_local) 
        if not os.path.exists(vis_output_dir):
            os.makedirs(vis_output_dir)

        ode_eps = cfg.sde.ode_eps  
        ode_solver_tol = 1e-5  
        enable_autocast = False
        temp = 1.0  
        noise = None 
        condition_input = None 
        clip_feat = None

        ##for vid, val_batch in enumerate(data_loader):
        ##    m, s = val_batch['mean'][0:1], val_batch['std'][0:1] 
        ##    B = val_batch['mean'].shape[0] 
        ##    break 
        #m = torch.zeros(1,3)
        #m[:,0] = -0.0308 #-1.0504e-02 
        #m[:,1] = -0.0353 #-4.1844e-03 
        #m[:,2] = -0.0001 #-5.1331e-05 
        #s = torch.zeros(1,1)
        #s[0] = 0.1512 #0.1694 

        #logger.info('mean: {}, s={}', m, s)
        B = 4 
        for vid, val_batch in enumerate(data_loader):
        ## for vid, (pt_cur, cate_cur) in enumerate(zip(input_pts_list, input_pts_cates)):
            if vid % 30 == 1:
                logger.info('eval: {}/{}, BS={}', vid, len(data_loader), B)
            output_dir = output_dir_template + '/sph_B%d_%04d'%(B, vid) 
            if not os.path.exists(output_dir):
                os.makedirs(output_dir) 
            pt_cur = val_batch['tr_points']
            pt_cur = pt_cur[:2] # select two input here
            B0,N,C = pt_cur.shape 
            input_pts = pt_cur[None].expand(B//B0, -1, -1, -1).contiguous().view(B,N,C).contiguous()

            # input_pts = ( input_pts - m ) / s 
            input_pts = input_pts.cuda().float()
            file_name = ['%04d'%i for i in range(B)] 
            val_x = input_pts  
            logger.info('val_x: {}', val_x.shape)

            B,N,C = val_x.shape
            inputs = val_x 
            log_info = {'x_0': val_x, 'vis/input_noise': inputs.view(B,N,-1), 'x_t': val_x}

            ## dist = self.model.encode(inputs)  
            # -- global latent -- # 
            dae_index = 0
            dist = self.model.encode_global(inputs)

            B = inputs.shape[0]
            shape = self.model.latent_shape()[dae_index]  
            noise_shape_global = [B] + shape  
            eps = dist.sample()[0] 
            eps_shape_global = eps.shape 
            eps = eps.view(noise_shape_global)  
            eps_ori = eps 
            # get eps_0: 
            i = 0
            dae = self.dae
            eps_0 = eps 
            eps_global = eps_0.contiguous() 
            condition_input = eps_global

            # style 
            style = self.model.global2style(eps_global.view(eps_shape_global))
            dist_local = self.model.encode_local(inputs, style) 

            # -- local latent -- # 
            dae_index = 1
            shape = self.model.latent_shape()[dae_index]  
            noise_shape_local = [B] + shape  
            eps = dist_local.sample()[0] 
            eps_shape_local = eps.shape 
            eps = eps.view(noise_shape_local)  
            eps_ori = eps 

            # -------------------
            # start the interplot 
            # -------------------
            ## eps_local = eps_ori 
            eps_T_global_interp = diffusion.compute_ode_nll(
                dae[0], eps_0, ode_eps, ode_solver_tol,
                condition_input=None
                )
            eps_T_global_interp = interpolate_noise(eps_T_global_interp.contiguous()) 
            eps_0_global_interp, _, _= diffusion.sample_model_ode(
                dae[0], B, shape, ode_eps, ode_solver_tol, enable_autocast, temp, 
                noise=eps_T_global_interp,
                condition_input=None, clip_feat=clip_feat
                )

            eps_T_local_interp = diffusion.compute_ode_nll(
                dae[1], eps, ode_eps, ode_solver_tol,
                condition_input=eps_global 
                )

            eps_T_local_interp = interpolate_noise(eps_T_local_interp.contiguous())    
            # double check if the eps_0 can denoise to get eps: 
            eps_0_local_interp, _, _= diffusion.sample_model_ode(
                dae[1], B, shape, ode_eps, 
                ode_solver_tol, enable_autocast, temp, 
                noise=eps_T_local_interp,
                condition_input=eps_0_global_interp, 
                clip_feat=clip_feat
                )
            
            style = self.model.global2style(eps_0_global_interp.view(eps_shape_global))
            eps_local = eps_0_local_interp.view(eps_shape_local) 
            gen_x = self.model.decoder(None, beta=None, context=eps_local, style=style) # (B,ncenter,3) 
            ## gen_x = val_x 
            
            # start the interpretation: 


            log_info['x_0_pred'] = gen_x.detach().cpu()
            if True: 
                img_list = []
                for i in range(B): 
                    points = gen_x[i] # N,3  
                    points = normalize_point_clouds([points])[0] 
                    img = visualize_point_clouds_3d([points])
                    img_list.append(img)
                img = np.concatenate(img_list, axis=2)
                self.writer.add_image('%d'%vid, torch.as_tensor(img), 0)
                img_list = [torch.as_tensor(img) / 255.0 for img in img_list] 
                vis_file = os.path.join(vis_output_dir, '%04d.png'%(vid)) 
                torchvision.utils.save_image(img_list, vis_file)
                logger.info('save img as: {}', vis_file)

            # forward to get output 
            val_x = val_x.contiguous() 
            inputs = inputs.contiguous()
            output = self.model.get_loss(val_x, it=step, is_eval_nll=1, noisy_input=inputs)
            #gen_x = gen_x.cpu() * s + m 
            for i, file_name_i in  enumerate(file_name):
                torch.save(gen_x[i], os.path.join(output_dir, file_name_i)) 
            logger.info('save output at : {}', output_dir)
        return 0  
