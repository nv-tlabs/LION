# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import os
import time
from abc import ABC, abstractmethod
from comet_ml import Experiment
import torch
import importlib
import numpy as np
from PIL import Image
from loguru import logger
import torchvision
import torch.distributed as dist
from utils.evaluation_metrics_fast import print_results 
from utils.checker import *
from utils.vis_helper import visualize_point_clouds_3d
from utils.eval_helper import compute_score, get_ref_pt, get_ref_num
from utils import model_helper, exp_helper, data_helper
from utils.utils import infer_active_variables 
from utils.data_helper import normalize_point_clouds
from utils.eval_helper import compute_NLL_metric
from utils.utils import AvgrageMeter
import clip

class BaseTrainer(ABC):
    def __init__(self, cfg, args):
        self.cfg, self.args = cfg, args
        self.scheduler = None
        self.local_rank = args.local_rank
        self.cur_epoch = 0
        self.start_epoch = 0
        self.epoch = 0
        self.step = 0
        self.writer = None
        self.encoder = None
        self.num_val_samples = cfg.num_val_samples
        self.train_iter_kwargs = {}
        self.num_points = self.cfg.data.tr_max_sample_points
        self.best_eval_epoch = 0
        self.best_eval_score = -1
        self.use_grad_scalar = cfg.trainer.use_grad_scalar
        device = torch.device('cuda:%d' % args.local_rank)
        self.device_str = 'cuda:%d' % args.local_rank
        self.t2s_input = []
        if cfg.clipforge.enable:
            self.prepare_clip_model_data()
        else:
            self.clip_feat_list = None

    def set_writer(self, writer):
        self.writer = writer
        logger.info(
            '\n'+'-'*10 + f'\n[url]: {self.writer.url}\n{self.cfg.save_dir}\n' + '-'*10)

    @abstractmethod
    def train_iter(self, data, *args, **kwargs):
        pass

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass

    def log_val(self, val_info, writer=None, step=None, epoch=None, **kwargs):
        if writer is not None:
            for k, v in val_info.items():
                if step is not None:
                    writer.add_scalar(k, v, step)
                else:
                    writer.add_scalar(k, v, epoch)

    def epoch_start(self, epoch):
        pass

    def epoch_end(self, epoch, writer=None, **kwargs):
        # Signal now that the epoch ends....
        if self.scheduler is not None:
            self.scheduler.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'train/opt_lr', self.scheduler.get_lr()[0], epoch)
        if writer is not None:
            writer.upload_meter(epoch=epoch, step=kwargs.get('step', None))

    # --- util function --
    def save(self, save_name=None, epoch=None, step=None, appendix=None, save_dir=None, **kwargs):
        d = {
            'opt': self.optimizer.state_dict(),
            'model': self.model.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if appendix is not None:
            d.update(appendix)
        if self.use_grad_scalar:
            d.update({'grad_scalar': self.grad_scalar.state_dict()})
        save_name = "epoch_%s_iters_%s.pt" % (
            epoch, step) if save_name is None else save_name
        save_dir = self.cfg.save_dir if save_dir is None else save_dir
        path = os.path.join(save_dir, "checkpoints", save_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info('save model as : {}', path)
        torch.save(d, path)
        return path

    def filter_name(self, ckpt):
        ckpt_new = {}
        for k, v in ckpt.items():
            if k[:7] == 'module.':
                kn = k[7:]
            elif k[:13] == 'model.module.':
                kn = k[13:]
            else:
                kn = k
            ckpt_new[kn] = v
        return ckpt_new

    def resume(self, path, strict=True, **kwargs):
        ckpt = torch.load(path)
        strict = True
        model_weight = ckpt['model'] if 'model' in ckpt else ckpt['model_state']
        vae_weight = self.filter_name(model_weight)
        self.model.load_state_dict(vae_weight, strict=strict)
        if 'opt' in ckpt:
            self.optimizer.load_state_dict(ckpt['opt'])
        else:
            logger.info('no optimizer found in ckpt')

        start_epoch = ckpt['epoch']
        self.epoch = start_epoch
        self.cur_epoch = start_epoch
        self.step = ckpt.get('step', 0)
        logger.info('resume from : {}, epo={}', path, start_epoch)
        if self.use_grad_scalar:
            assert('grad_scalar' in ckpt), 'otherwise set it false'
            self.grad_scalar.load_state_dict(ckpt['grad_scalar'])
        return start_epoch

    def build_model(self):
        cfg, args = self.cfg, self.args
        if args.distributed:
            dist.barrier()
        model_lib = importlib.import_module(cfg.shapelatent.model)
        model = model_lib.Model(cfg)
        return model

    def build_data(self):
        logger.info('start build_data')
        cfg, args = self.cfg, self.args
        self.args.eval_trainnll = cfg.eval_trainnll
        data_lib = importlib.import_module(cfg.data.type)
        loaders = data_lib.get_data_loaders(cfg.data, args)
        train_loader = loaders['train_loader']
        test_loader = loaders['test_loader']
        return train_loader, test_loader

    def train_epochs(self):
        """ train for number of epochs; """
        # main training loop
        cfg, args = self.cfg, self.args
        train_loader = self.train_loader
        writer = self.writer

        if cfg.viz.log_freq <= -1:  # treat as per epoch
            cfg.viz.log_freq = int(- cfg.viz.log_freq * len(train_loader))
        if cfg.viz.viz_freq <= -1:
            cfg.viz.viz_freq = - cfg.viz.viz_freq * len(train_loader)

        logger.info("[rank=%d] Start epoch: %d End epoch: %d, batch-size=%d | "
                    "Niter/epo=%d | log freq=%d, viz freq %d, val freq %d " %
                    (args.local_rank,
                     self.start_epoch, cfg.trainer.epochs, cfg.data.batch_size,
                     len(train_loader),
                        cfg.viz.log_freq, cfg.viz.viz_freq, cfg.viz.val_freq))
        tic0 = time.time()
        step = 0
        if args.global_rank == 0:
            tic_log = time.time()
        self.num_total_iter = cfg.trainer.epochs * len(train_loader)
        self.model.num_total_iter = self.num_total_iter

        for epoch in range(self.start_epoch, cfg.trainer.epochs):
            self.cur_epoch = epoch
            if args.global_rank == 0:
                tic_epo = time.time()
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            if args.global_rank == 0 and cfg.trainer.type in ['trainers.voxel2pts', 'trainers.voxel2pts_ada'] and epoch == 0:
                self.eval_nll(step=step)
            epoch_loss = []
            self.epoch_start(epoch)

            # remove disabled latent variables by setting their mixing component to a small value
            if epoch == 0 and cfg.sde.mixed_prediction and cfg.sde.drop_inactive_var:
                raise NotImplementedError

            ## -- train for one epoch -- ##
            for bidx, data in enumerate(train_loader):
                # let step start from 0 instead of  1
                step = bidx + len(train_loader) * epoch

                if args.global_rank == 0 and self.writer is not None:
                    tic_iter = time.time()

                # -- train for one iter -- #
                logs_info = self.train_iter(data, step=step,
                                            **self.train_iter_kwargs)

                # -- log information within epoch -- #
                if self.args.global_rank == 0:
                    epoch_loss.append(logs_info['loss'])
                if self.args.global_rank == 0 and (
                        time.time() - tic_log > 60
                ):  # log per min
                    logger.info('[R%d] | E%d iter[%3d/%3d] | [Loss] %2.2f | '
                        '[exp] %s | [step] %5d | [url] %s ' % (
                        args.global_rank, epoch,  bidx, len(train_loader),
                        np.array(epoch_loss).mean(),
                        cfg.save_dir, step, writer.url
                        ))
                    tic_log = time.time()

                # -- visualize rec and samples -- #
                if step % int(cfg.viz.log_freq) == 0 and \
                        args.global_rank == 0 and not (
                        step == 0 and cfg.sde.ode_sample and
                    (cfg.trainer.type == 'trainers.train_prior' or cfg.trainer.type ==
                     'trainers.train_2prior') # this case, skip sampling at first step
                ):
                    avg_loss = np.array(epoch_loss).mean()
                    epo_loss = []  # clean up epoch loss
                    self.log_loss({'epo_loss': avg_loss},
                                  writer=writer, step=step)
                    visualize = int(cfg.viz.viz_freq) > 0 and \
                        (step) % int(cfg.viz.viz_freq) == 0
                    vis_recont = visualize
                    if vis_recont:
                        self.vis_recont(logs_info, writer, step)
                    if visualize:
                        self.model.eval()
                        self.vis_sample(writer, step=step,
                                        include_pred_x0=False)
                        self.model.train()

                # -- timer -- #
                if args.global_rank == 0 and self.writer is not None:
                    time_iter = time.time() - tic_iter
                    self.writer.avg_meter('time_iter', time_iter, step=step)
            ## -- log information after one epoch -- ##
            if args.global_rank == 0:
                epo_time = (time.time() - tic_epo) / 60.0  # min
                logger.info('[R%d] | E%d iter[%3d/%3d] | [Loss] %2.2f '
                    '| [exp] %s | [step] %5d | [url] %s | [time] %.1fm (~%dh) |'
                    '[best] %d %.3fx1e-2 ' % (
                    args.global_rank, epoch,  bidx, len(train_loader),
                    np.array(epoch_loss).mean(), 
                    cfg.save_dir, step, writer.url,
                    epo_time, epo_time * (cfg.trainer.epochs - epoch) / 60,
                    self.best_eval_epoch, self.best_eval_score*1e2
                    ))
                tic_log = time.time()  # reset tic_log

            ## -- save model -- ##
            if (epoch + 1) % int(cfg.viz.save_freq) == 0 and \
                    int(cfg.viz.save_freq) > 0 and args.global_rank == 0:
                self.save(epoch=epoch, step=step)
            if ((time.time() - tic0) / 60 > cfg.snapshot_min) and \
                    args.global_rank == 0:  # save every 30 min
                file_name = self.save(
                    save_name='snapshot_bak', epoch=epoch, step=step)
                if file_name is None:
                    file_name = os.path.join(
                        self.cfg.save_dir, "checkpoints", "snapshot_bak")
                os.rename(file_name, file_name.replace(
                    'snapshot_bak', 'snapshot'))
                tic0 = time.time()

            ## -- run eval -- ##
            if int(cfg.viz.val_freq) > 0 and (epoch + 1) % int(cfg.viz.val_freq) == 0 and \
                    args.global_rank == 0:
                eval_score = self.eval_nll(step=step, save_file=False)
                if eval_score < self.best_eval_score or self.best_eval_score < 0:
                    self.save(save_name='best_eval.pth',  # save_dir=snapshot_dir,
                              epoch=epoch, step=step)
                    self.best_eval_score = eval_score
                    self.best_eval_epoch = epoch

            ## -- Signal the trainer to cleanup now that an epoch has ended -- ##
            self.epoch_end(epoch, writer=writer, step=step)
        ### -- end of the training -- ###
        if args.global_rank == 0:
            self.eval_nll(step=step)
        if self.cfg.trainer.type == 'trainers.train_prior':  # and args.global_rank == 0:
            self.model.eval()
            self.eval_sample(step)
            logger.info('debugging eval-sample; exit now')

    @torch.no_grad()
    def log_loss(self, train_info, writer=None, step=None, **kwargs):
        """ write to tensorboard and visualize 
        """
        if writer is None:
            return

        # Log training information to tensorboard
        train_info = {
            k: (v.cpu() if not isinstance(v, float) else v)
            for k, v in train_info.items()
        }
        for k, v in train_info.items():
            if not ('loss' in k):
                continue
            if step is not None:
                writer.add_scalar('train/' + k, v, step)
            else:
                assert epoch is not None
                writer.add_scalar('train/' + k, v, epoch)

    # --------------------------------------------- #
    #   visulization function and sampling function #
    # --------------------------------------------- #
    @torch.no_grad()
    def vis_recont(self, output, writer, step, normalize_pts=False):
        """
        Args:
            x_0:  Input point cloud, (B, N, d).
        """
        if writer is None:
            return 0
        # x_0: target
        # x_0_pred: recont
        # x_t: intermidiate sample at t (if t is not None)
        x_0_pred, x_0, x_t = output.get('x_0_pred', None), \
            output.get('x_0', None), output.get('x_t', None)
        if x_0_pred is None or x_0 is None or x_t is None:
            logger.info('x_0_pred: None? {}; x_0: None? {}, x_t: None? {}',
                        x_0_pred is None, x_0 is None, x_t is None)
            return 0

        CHECK3D(x_0)
        CHECK3D(x_t)
        CHECK3D(x_0_pred)

        t = output.get('t', None)
        nvis = min(max(x_0.shape[0], 2), 5)
        img_list = []
        for b in range(nvis):
            x_list, name_list = [], []
            x_list.append(x_0_pred[b])
            name_list.append('pred')

            if t is not None and t[b] > 0:
                x_t_name = 'x_t%d' % t[b].item()
                name_list.append(x_t_name)
                x_list.append(x_t[b])

            x_list.append(x_0[b])
            name_list.append('target')

            for k, v in output.items():
                if 'vis/' in k:
                    x_list.append(v[b])
                    name_list.append(k)
            if normalize_pts:
                x_list = normalize_point_clouds(x_list)

            vis_order = self.cfg.viz.viz_order
            vis_args = {'vis_order': vis_order}

            img = visualize_point_clouds_3d(x_list, name_list, **vis_args)
            img_list.append(img)
        img_list = torchvision.utils.make_grid(
            [torch.as_tensor(a) for a in img_list], pad_value=0)
        writer.add_image('vis_out/recont-train', img_list, step)

    @torch.no_grad()
    def eval_sample(self, step=0):
        """ compute sample metric: MMD,COV,1-NNA  """
        writer = self.writer
        batch_size_test = self.cfg.data.batch_size_test
        input_dim = self.cfg.ddpm.input_dim
        ddim_step = self.cfg.eval_ddim_step
        device = model_helper.get_device(self.model)
        test_loader = self.test_loader
        test_size = batch_size_test * len(test_loader)
        sample_num_points = self.cfg.data.tr_max_sample_points
        cates = self.cfg.data.cates
        num_ref = get_ref_num(
            cates) if self.cfg.num_ref == 0 else self.cfg.num_ref

        # option for post-processing
        if self.cfg.data.recenter_per_shape or self.cfg.data.normalize_shape_box or self.cfg.data.normalize_range:
            norm_box = True
        else:
            norm_box = False
        logger.info('norm_box: {}, recenter : {}, shapebox: {}',
                    norm_box, self.cfg.data.recenter_per_shape,
                    self.cfg.data.normalize_shape_box)

        # get exp tag and output name
        tag = exp_helper.get_evalname(self.cfg)
        if not self.cfg.sde.ode_sample:
            tag += 'diet'
        else:
            tag += 'ode%d' % self.cfg.sde.ode_sample
        output_name = os.path.join(
            self.cfg.save_dir, f'samples_{step}{tag}.pt')
        logger.info('batch_size_test={}, test_size={}, saved output: {} ',
                    batch_size_test, test_size, output_name)
        gen_pcs = []

        ### ---- ref_pcs ---- #
        ##ref_pcs = []
        ##m_pcs, s_pcs = [], []
        # for i, data in enumerate(test_loader):
        ##    tr_points = data['tr_points']
        ##    m, s = data['mean'], data['std']
        # ref_pcs.append(tr_points) # B,N,3
        # m_pcs.append(m.float())
        # s_pcs.append(s.float())
        ##    sample_num_points = tr_points.shape[1]
        # assert(tr_points.shape[2] in [3,6]
        # ), f'expect B,N,3; get {tr_points.shape}'
        ##ref_pcs = torch.cat(ref_pcs, dim=0)
        ##m_pcs = torch.cat(m_pcs, dim=0)
        ##s_pcs = torch.cat(s_pcs, dim=0)
        # if VIS:
        ##    img_list = []
        # for i in range(4):
        ##        norm_ref, norm_gen = data_helper.normalize_point_clouds([ref_pcs[i], ref_pcs[-i]])
        ##        img = visualize_point_clouds_3d([norm_ref, norm_gen], [f'ref-{i}', f'ref-{-i}'])
        ##        img_list.append(torch.as_tensor(img) / 255.0)
        ##    path = output_name.replace('.pt', '_ref.png')
        # torchvision.utils.save_image(img_list, path)
        ##    grid = torchvision.utils.make_grid(img_list)
        # ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        ##    writer.add_image('ref', grid, 0)
        # logger.info(writer.url)
        ##    logger.info('save vis at {}', path)

        # ---- gen_pcs ---- #
        if True:
            len_test_loader = num_ref // batch_size_test + 1
            if self.args.distributed:
                num_gen_iter = max(1, len_test_loader // self.args.global_size)
                if num_gen_iter * batch_size_test * self.args.global_size < num_ref:
                    num_gen_iter = num_gen_iter + 1
            else:
                num_gen_iter = len_test_loader

            index_start = 0
            logger.info('Rank={}, num_gen_iter: {}; num_ref={}, batch_size_test={}',
                        self.args.global_rank, num_gen_iter, num_ref, batch_size_test)
            seed = self.cfg.trainer.seed
            for i in range(0, num_gen_iter):
                torch.manual_seed(seed + i)
                np.random.seed(seed + i)
                torch.cuda.manual_seed(seed + i)
                torch.cuda.manual_seed_all(seed + i)

                logger.info('#%d/%d; BS=%d' %
                            (i, num_gen_iter, batch_size_test))
                # ---- draw samples ---- #
                self.index_start = index_start
                x = self.sample(num_shapes=batch_size_test,
                                num_points=sample_num_points,
                                device_str=device.type,
                                for_vis=False,
                                ddim_step=ddim_step).permute(0, 2, 1).contiguous()  # B,3,N->B,N,3
                assert(
                    x.shape[-1] == input_dim), f'expect x: B,N,{input_dim}; get {x.shape}'
                index_start = index_start + batch_size_test
                gen_pcs.append(x.detach().cpu())

            gen_pcs = torch.cat(gen_pcs, dim=0)
        if self.args.distributed:
            gen_pcs = gen_pcs.to(torch.device(self.device_str))
            logger.info('before gather: {}, rank={}',
                        gen_pcs.shape, self.args.global_rank)
            gen_pcs_list = [torch.zeros_like(gen_pcs)
                            for _ in range(self.args.global_size)]
            dist.all_gather(gen_pcs_list, gen_pcs)
            gen_pcs = torch.cat(gen_pcs_list, dim=0).cpu()
            logger.info('after gather: {}, rank={}',
                        gen_pcs.shape, self.args.global_rank)
        logger.info('save as %s' % output_name)
        if self.args.global_rank == 0:
            torch.save(gen_pcs, output_name)
        else:
            logger.info('return for rank {}', self.args.global_rank)
            return  # only do eval on one gpu
        if writer is not None:
            img_list = []
            for i in range(1):
                gen_list = [gen_pcs[k] for k in range(len(gen_pcs))][:8]
                norm_ref = data_helper.normalize_point_clouds(gen_list)
                img = visualize_point_clouds_3d(norm_ref, [f'gen-{k}' for k in range(len(norm_ref))]
                                                )
                img_list.append(torch.as_tensor(img) / 255.0)
            grid = torchvision.utils.make_grid(img_list)
            logger.info('ndarr: {}, range: {} img list: {} ', grid.shape,
                        grid.max(), img_list[0].shape, img_list[0].max())
            writer.add_image('sample', grid, step)
            logger.info('\n\t' + writer.url)
        #logger.info('early exit')
        # exit()

        shape_str = '{}: gen_pcs: {}'.format(self.cfg.save_dir, gen_pcs.shape)
        logger.info(shape_str)

        ref = get_ref_pt(self.cfg.data.cates, self.cfg.data.type)
        if ref is None:
            logger.info('Not computing score')
            return 1
        step_str = '%dk' % (step / 1000.0)
        epoch_str = '%.1fk' % (self.epoch / 1000.0)
        print_kwargs = {'dataset': self.cfg.data.cates,
                        'hash': self.cfg.hash + tag,
                        'step': step_str,
                        'epoch': epoch_str+'-'+os.path.basename(ref).split('.')[0]}
        self.model = self.model.cpu()
        torch.cuda.empty_cache()
        # -- compute the generation metric -- #
        results = compute_score(output_name, ref_name=ref,
                                writer=writer,
                                batch_size_test=min(
                                    5, self.cfg.data.batch_size_test),
                                norm_box=norm_box,
                                **print_kwargs)

        self.model = self.model.to(device)
        # ---- write to logger ---- #
        writer.add_scalar('test/Coverage_CD', results['lgan_cov-CD'], step)
        writer.add_scalar('test/Coverage_EMD', results['lgan_cov-EMD'], step)
        writer.add_scalar('test/MMD_CD', results['lgan_mmd-CD'], step)
        writer.add_scalar('test/MMD_EMD', results['lgan_mmd-EMD'], step)
        writer.add_scalar('test/1NN_CD', results['1-NN-CD-acc'], step)
        writer.add_scalar('test/1NN_EMD', results['1-NN-EMD-acc'], step)
        writer.add_scalar('test/JSD', results['jsd'], step)
        msg = f'step={step}'
        msg += '\n[Test] MinMatDis | CD %.6f | EMD %.6f' % (
            results['lgan_mmd-CD'], results['lgan_mmd-EMD'])
        msg += '\n[Test] Coverage  | CD %.6f | EMD %.6f' % (
            results['lgan_cov-CD'], results['lgan_cov-EMD'])
        msg += '\n[Test] 1NN-Accur | CD %.6f | EMD %.6f' % (
            results['1-NN-CD-acc'], results['1-NN-EMD-acc'])
        msg += '\n[Test] JsnShnDis | %.6f ' % (results['jsd'])

        logger.info(msg)
        with open(os.path.join(self.cfg.save_dir, 'eval_out.txt'), 'a') as f:
            f.write(shape_str+'\n')
            f.write(msg+'\n')
        # self.cfg.data.cates, self.cfg.hash, step_str, epoch_str)
        msg = print_results(results, **print_kwargs)
        with open(os.path.join(self.cfg.save_dir, 'eval_out.txt'), 'a') as f:
            f.write(msg+'\n')
        logger.info('\n\t' + writer.url)

    def vis_sample(self, writer, num_vis=None, step=0, include_pred_x0=True,
                   save_file=None):
        if num_vis is None:
            num_vis = self.num_val_samples
        logger.info("Sampling.. train-step=%s | N=%d" % (step, num_vis))
        tic = time.time()
        # get three list with entry: [L,N,3]
        # traj, traj_x0, time_list
        traj, pred_x0 = self.sample(num_points=self.num_points,
                                    num_shapes=num_vis, for_vis=True, use_ddim=True,
                                    save_file=save_file)

        toc = time.time()
        logger.info('sampling take %.1f sec' % (toc-tic))

        # display only a few steps
        num_shapes = num_vis
        vis_num_steps = len(traj)
        vis_index = list(traj.keys())
        vis_index = vis_index[::-1]
        display_num_step = 5
        step_size = max(1, vis_num_steps // 5)
        display_num_step_list = []
        for k in range(0, vis_num_steps, step_size):
            display_num_step_list.append(vis_index[k])
        if self.num_steps not in display_num_step_list and self.num_steps in traj:
            display_num_step_list.append(self.num_steps)
        logger.info('saving vis with N={}', len(display_num_step_list))
        alltraj_list = []
        allpred_x0_list = []
        allstep_list = []
        for b in range(num_shapes):
            traj_list = []
            pred_x0_list = []
            step_list = []
            for k in display_num_step_list:
                v = traj[k]
                traj_list.append(v[b].permute(1, 0).contiguous())
                v = pred_x0[k]
                pred_x0_list.append(v[b].permute(1, 0).contiguous())
                step_list.append(k)
                # B3N -> 3,N -> N,3 use first sample only
            alltraj_list.append(traj_list)
            allpred_x0_list.append(pred_x0_list)
            allstep_list.append(step_list)
        traj, traj_x0, time_list = alltraj_list, allpred_x0_list, allstep_list

        # vis the final images,
        all_imgs = []
        all_imgs_torchvis = []  # no preconcat in the image, left to the torchvision
        all_imgs_torchvis_norm = []  # no preconcat in the image, left to the torchvision
        for idx in range(num_vis):
            pcs = traj[idx][0:1]  # 1,N,3
            img = []
            # vis the normalized point cloud
            title_list = ['#%d normed x_%d' % (idx, 0)]
            norm_pcs = data_helper.normalize_point_clouds(pcs)
            img.append(visualize_point_clouds_3d(norm_pcs, title_list,
                                                 self.cfg.viz.viz_order))
            all_imgs_torchvis_norm.append(img[-1] / 255.0)
            if include_pred_x0:
                title_list = ['#%d p(x_0|x_%d,t)' % (idx, 0)]
                img.append(visualize_point_clouds_3d(traj_x0[idx][0:1], title_list,
                                                     self.cfg.viz.viz_order))
            # concat along the height
            all_imgs.append(np.concatenate(img, axis=1))

        # concatenate along the width dimension
        img = np.concatenate(all_imgs, axis=2)
        writer.add_image('summary/sample', torch.as_tensor(img), step)

        path = os.path.join(self.cfg.save_dir, 'vis', 'sample%06d.png' % step)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        img_list = [torch.as_tensor(a) for a in all_imgs_torchvis_norm]
        grid = torchvision.utils.make_grid(img_list)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(
            1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        im.save(path)
        logger.info('save as {}; url: {} ', path, writer.url)

    def prepare_vis_data(self):
        device = torch.device(self.device_str)
        num_val_samples = self.num_val_samples
        c = 0
        val_x = []
        val_input = []
        val_cls = []
        prior_cond = []
        for val_batch in self.test_loader:
            val_x.append(val_batch['tr_points'])
            val_cls.append(val_batch['cate_idx'])
            if 'input_pts' in val_batch:  # this is the input_pts to the vae model
                val_input.append(val_batch['input_pts'])
            if 'prior_cond' in val_batch:
                prior_cond.append(val_batch['prior_cond'])
            c += val_x[-1].shape[0]
            if c >= num_val_samples:
                break
        self.val_x = torch.cat(val_x, dim=0)[:num_val_samples].to(device)
        # this line may trigger error, change dataset output cate_idx from string to int can fix this issue
        self.val_cls = torch.cat(val_cls, dim=0)[:num_val_samples].to(device)
        self.prior_cond = torch.cat(prior_cond, dim=0)[:num_val_samples].to(
            device) if len(prior_cond) else None
        self.val_input = torch.cat(val_input, dim=0)[:num_val_samples].to(
            device) if len(val_input) else None
        c = 0
        tr_x = []
        m_x = []
        s_x = []
        tr_cls = []
        logger.info('[prepare_vis_data] len of train_loader: {}',
                    len(self.train_loader))
        assert(len(self.train_loader) > 0), f'get zero length train_loader, it could be the batch_size > the number of training sample, and the train drop_last is turn off'
        for tr_batch in self.train_loader:
            tr_x.append(tr_batch['tr_points'])
            m_x.append(tr_batch['mean'])
            s_x.append(tr_batch['std'])
            tr_cls.append(tr_batch['cate_idx'].view(-1))
            c += tr_x[-1].shape[0]
            if c >= num_val_samples:
                break
        self.tr_cls = torch.cat(tr_cls, dim=0)[:num_val_samples].to(device)
        self.tr_x = torch.cat(tr_x, dim=0)[:num_val_samples].to(device)
        self.m_pcs = torch.cat(m_x, dim=0)[:num_val_samples].to(device)
        self.s_pcs = torch.cat(s_x, dim=0)[:num_val_samples].to(device)
        logger.info('tr_x: {}, m_pcs: {}, s_pcs: {}, val_x: {}',
                    self.tr_x.shape, self.m_pcs.shape, self.s_pcs.shape,  self.val_x.shape)

        self.w_prior = torch.randn(
            [num_val_samples, self.cfg.shapelatent.latent_dim]).to(device)
        if self.clip_feat_list is not None:
            self.clip_feat_test = []
            for k in range(len(self.clip_feat_list)):
                for i in range(num_val_samples // len(self.clip_feat_list)):
                    self.clip_feat_test.append(self.clip_feat_list[k])
            for i in range(num_val_samples - len(self.clip_feat_test)):
                self.clip_feat_test.append(self.clip_feat_list[-1])
            self.clip_feat_test = torch.stack(self.clip_feat_test, dim=0)
            logger.info('[VIS data] clip_feat_test: {}',
                        self.clip_feat_test.shape)
            if self.clip_feat_test.shape[0] > num_val_samples:
                self.clip_feat_test = self.clip_feat_test[:num_val_samples]
        else:
            self.clip_feat_test = None

    def build_other_module(self):
        logger.info('no other module to build')
        pass

    def swap_vae_param_if_need(self):
        if self.cfg.ddpm.ema:
            self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)

    # -- shared method for all model with vae component -- #
    @torch.no_grad()
    def eval_nll(self, step, ntest=None, save_file=False):
        loss_dict = {}
        cfg = self.cfg
        self.swap_vae_param_if_need() # if using EMA, load the ema weight
        args = self.args
        device = torch.device('cuda:%d' % args.local_rank)
        tag = exp_helper.get_evalname(self.cfg)
        eval_trainnll = 0
        if eval_trainnll:
            data_loader = self.train_loader
            tag += '-train'
        else:
            data_loader = self.test_loader
        gen_pcs, ref_pcs = [], []
        output_name = os.path.join(self.cfg.save_dir, f'recont_{step}{tag}.pt')
        output_name_metric = os.path.join(
            self.cfg.save_dir, f'recont_{step}{tag}_metric.pt')
        shape_id_start = 0
        batch_metric_all = {}

        for vid, val_batch in enumerate(data_loader):
            if vid % 30 == 1:
                logger.info('eval: {}/{}', vid, len(data_loader))
            val_x = val_batch['tr_points'].to(device)
            m, s = val_batch['mean'], val_batch['std']
            B, N, C = val_x.shape
            m = m.view(B, 1, -1)
            s = s.view(B, 1, -1)
            inputs = val_batch['input_pts'].to(
                device) if 'input_pts' in val_batch else None  # the noisy points
            model_kwargs = {}

            output = self.model.get_loss(
                val_x, it=step, is_eval_nll=1, noisy_input=inputs, **model_kwargs)

            # book-keeping
            for k, v in output.items():
                if 'print/' in k:
                    k = k.split('print/')[-1]
                    if k not in loss_dict:
                        loss_dict[k] = AvgrageMeter()
                    v = v.mean().item() if torch.is_tensor(v) else v
                    loss_dict[k].update(v)

            gen_x = output['final_pred']
            if gen_x.shape[1] > val_x.shape[1]:  # downsample points if needed
                tr_idxs = np.random.permutation(np.arange(gen_x.shape[1]))[
                    :val_x.shape[1]]
                gen_x = gen_x[:, tr_idxs]

            gen_x = gen_x.cpu()
            val_x = val_x.cpu()
            gen_x[:, :, :3] = gen_x[:, :, :3] * s + m
            val_x[:, :, :3] = val_x[:, :, :3] * s + m
            gen_pcs.append(gen_x.detach().cpu())
            ref_pcs.append(val_x.detach().cpu())
            if ntest is not None and shape_id_start >= int(ntest):
                logger.info('!! reach number of test={}; has test: {}',
                            ntest, shape_id_start)
                break
            shape_id_start += B
        # summarized batch-metric if any
        for k, v in batch_metric_all.items():
            if len(v) == 0:
                continue
            v = torch.cat(v, dim=0)
            logger.info('{}={}', k, v.mean())

        gen_pcs = torch.cat(gen_pcs, dim=0)
        ref_pcs = torch.cat(ref_pcs, dim=0)

        # Save
        if self.writer is not None:
            img_list = []
            for i in range(10):
                points = gen_pcs[i]
                points = normalize_point_clouds([points])[0]
                img = visualize_point_clouds_3d([points], bound=1.0)
                img_list.append(img)
            img = np.concatenate(img_list, axis=2)
            self.writer.add_image('nll/rec', torch.as_tensor(img), step)
        if save_file:
            logger.info('reconstruct point clouds..., output shape: {}, save as {}',
                        gen_pcs.shape, output_name)
            torch.save(gen_pcs, output_name)

        results = compute_NLL_metric(
            gen_pcs[:, :, :3], ref_pcs[:, :, :3], device, self.writer, output_name, batch_size=20, step=step)
        score = 0
        for n, v in results.items():
            if 'detail' in n:
                continue
            if self.writer is not None:
                logger.info('add: {}', n)
                self.writer.add_scalar('eval/%s' % (n), v, step)
            if 'CD' in n:
                score = v
        self.swap_vae_param_if_need()  # if using EMA, swap back to none-ema weight here
        return score

    def prepare_clip_model_data(self):
        cfg = self.cfg
        self.clip_model, self.clip_preprocess = clip.load(cfg.clipforge.clip_model,
                                                          device=self.device_str)
        self.test_img_path = []
        if cfg.data.cates == 'chair':
            input_t = [
                "an armchair in the shape of an avocado. an armchair imitating a avocado"]
            text = clip.tokenize(input_t).to(self.device_str)
        elif cfg.data.cates == 'car':
            input_t = ["a ford model T", "a pickup", "an off-road vehicle"]
            text = clip.tokenize(input_t).to(self.device_str)
        elif cfg.data.cates == 'all':
            input_t = ['a boeing', 'an f-16', 'an suv', 'a chunk', 'a limo',
                       'a square chair', 'a swivel chair', 'a sniper rifle']
            text = clip.tokenize(input_t).to(self.device_str)
        else:
            raise NotImplementedError
        if len(self.test_img_path):
            self.test_img = [Image.open(t).convert("RGB")
                             for t in self.test_img_path]
            self.test_img = [self.clip_preprocess(img).unsqueeze(
                0).to(self.device_str) for img in self.test_img]
            self.test_img = torch.cat(self.test_img, dim=0)
        else:
            self.test_img = []
        self.t2s_input = self.test_img_path + input_t
        clip_feat = []
        if len(self.test_img):
            clip_feat.append(
                self.clip_model.encode_image(self.test_img).float())
        clip_feat.append(self.clip_model.encode_text(text).float())
        self.clip_feat_list = torch.cat(clip_feat, dim=0)

