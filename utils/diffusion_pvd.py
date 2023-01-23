# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
"""copied and modified from https://github.com/NVlabs/LSGM/blob/5eae2f385c014f2250c3130152b6be711f6a3a5a/diffusion_discretized.py"""
import torch
from torch.cuda.amp import autocast
import numpy as np
from utils.diffusion import make_beta_schedule
from utils import utils
from loguru import logger


class DiffusionDiscretized(object):
    """
    This class constructs the diffusion process and provides all related methods and constants.
    """

    def __init__(self, args, var_fun, cfg):  # alpha_bars_fun
        self.cfg = cfg

        self._diffusion_steps = cfg.ddpm.num_steps # args.diffusion_steps
        self._denoising_stddevs = 'beta' # args.denoising_stddevs
        #self._var_fun = var_fun
        beta_start = cfg.ddpm.beta_1
        beta_end = cfg.ddpm.beta_T
        mode = cfg.ddpm.sched_mode
        num_steps = cfg.ddpm.num_steps
        self.p2_gamma = cfg.ddpm.p2_gamma
        self.p2_k = cfg.ddpm.p2_k
        self.use_p2_weight = self.cfg.ddpm.use_p2_weight

        logger.info(
            f'[Build Discrete Diffusion object] beta_start={beta_start}, beta_end={beta_end}, mode={mode}, num_steps={num_steps}')
        self.betas = make_beta_schedule(
            mode, beta_start, beta_end, num_steps).numpy()
        self._betas_init, self._alphas, self._alpha_bars, self._betas_post_init, self.snr = \
            self._generate_base_constants(
                diffusion_steps=self._diffusion_steps)

    def iw_quantities_t(self, B, timestep, *args):
        timestep = timestep.view(B)
        timestep = timestep + 1  # [1,T]
        alpha_bars = torch.gather(self._alpha_bars, 0, timestep-1)  # [0,T-1]
        weight_init = alpha_bars_sqrt = torch.sqrt(alpha_bars)
        weight_noise_power = 1.0 - alpha_bars

        weight_noise_power = weight_noise_power[:, None, None, None]
        weight_init = weight_init[:, None, None, None]
        if self.use_p2_weight:
            p2_weight = torch.gather(
                1 / (self.p2_k + self.snr)**self.p2_gamma, 0, timestep-1).view(B)
            loss_weight = p2_weight
        else:
            loss_weight = 1.0
        return timestep, weight_noise_power, weight_init, loss_weight, None, None

    def iw_quantities(self, B, *args):
        rho = torch.rand(size=[B], device='cuda') * self._diffusion_steps
        timestep = rho.type(torch.int64)  # [0, T-1]
        assert(timestep.max() <= self._diffusion_steps -
               1), f'get max at {timestep.max()}'
        timestep = timestep + 1  # [1,T]
        alpha_bars = torch.gather(self._alpha_bars, 0, timestep-1)  # [0,T-1]
        weight_init = alpha_bars_sqrt = torch.sqrt(alpha_bars)
        weight_noise_power = 1.0 - alpha_bars

        weight_noise_power = weight_noise_power[:, None, None, None]
        weight_init = weight_init[:, None, None, None]
        if self.use_p2_weight:
            p2_weight = torch.gather(
                1 / (self.p2_k + self.snr)**self.p2_gamma, 0, timestep-1).view(B)
            loss_weight = p2_weight
        else:
            loss_weight = 1.0

        return timestep, weight_noise_power, weight_init, loss_weight, None, None

    def debug_sheduler(self):
        rho = torch.range(0, 1000-1).cuda()  # / 1000.0 + time_eps
        timestep = rho.type(torch.int64)  # [0, T-1]
        assert(timestep.max() <= self._diffusion_steps -
               1), f'get max at {timestep.max()}'
        timestep = timestep + 1  # [1,T]
        alpha_bars = torch.gather(self._alpha_bars, 0, timestep-1)  # [0,T-1]
        weight_init = alpha_bars_sqrt = torch.sqrt(alpha_bars)
        weight_noise_power = 1.0 - alpha_bars

        weight_noise_power = weight_noise_power[:, None, None, None]
        weight_init = weight_init[:, None, None, None]
        return timestep, weight_noise_power, weight_init, 1, None, None

    def sample_q(self, x_init, noise, var_t, m_t):
        """ returns a sample from diffusion process at time t 
        x_init: [B,ND,1,1]
        noise: 
        vae_t: weight noise; [B,1,1,1]
        m_t: weight init; [B,1,1,1]
        """
        assert(len(x_init.shape) == 4)
        assert(len(var_t.shape) == 4)
        assert(len(m_t.shape) == 4)
        #CHECK4D(x_init)
        #CHECK4D(var_t)
        #CHECK4D(m_t)
        #CHECKEQ(x_init.shape[0], m_t.shape[0])
        assert(x_init.shape[0] == m_t.shape[0])
        output = m_t * x_init + torch.sqrt(var_t) * noise

        return output

    def cross_entropy_const(self, ode_eps):
        return 0

    def _generate_base_constants(self, diffusion_steps):
        """
        Generates torch tensors with basic constants for all timesteps.
        """
        betas_np = self.betas  # self._generate_betas_from_continuous_fun(diffusion_steps)

        alphas_np = 1.0 - betas_np
        alphas_cumprod = alpha_bars_np = np.cumprod(alphas_np)
        snr = 1.0 / (1 - alphas_cumprod) - 1

        # posterior variances only make sense for t>1, hence the array is short by 1
        betas_post_np = betas_np[1:] * \
            (1.0 - alpha_bars_np[:-1]) / (1.0 - alpha_bars_np[1:])
        # we add beta_post_2 to the beginning of both beta arrays, since this is used as final decoder variance and
        # requires special treatment (as in diffusion paper)
        betas_post_init_np = np.append(betas_post_np[0], betas_post_np)
        #betas_init_np = np.append(betas_post_np[0], betas_np[1:])

        betas_init = torch.from_numpy(betas_np).float().cuda()
        snr = torch.from_numpy(snr).float().cuda()
        alphas = torch.from_numpy(alphas_np).float().cuda()
        alpha_bars = torch.from_numpy(alpha_bars_np).float().cuda()
        betas_post_init = torch.from_numpy(betas_post_init_np).float().cuda()

        return betas_init, alphas, alpha_bars, betas_post_init, snr

    # def _generate_betas_from_continuous_fun(self, diffusion_steps):
    #    t = np.arange(1, diffusion_steps + 1, dtype=np.float64)
    #    t = t / diffusion_steps

    #    # alpha_bars = self._alpha_bars_fun(t)
    #    alpha_bars = 1.0 - self._var_fun(torch.tensor(t)).numpy()
    #    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    #    betas = np.hstack((1 - alpha_bars[0], betas))

    #    return betas

    def get_p_log_scales(self, timestep, stddev_type):
        """
        Grab log std devs. of backward denoising process p, if we decide to fix them.
        """
        if stddev_type == 'beta':
            # use diffusion variances, except for t=1, for which we use posterior variance beta_post_2
            return 0.5 * torch.log(torch.gather(self._betas_init, 0, timestep-1))
        elif stddev_type == 'beta_post':
            # use diffusion posterior variances, except for t=1, for which there is no posterior, so we use beta_post_2
            return 0.5 * torch.log(torch.gather(self._betas_post_init, 0, timestep-1))
        elif stddev_type == 'learn':
            return None
        else:
            raise ValueError('Unknown stddev_type: {}'.format(stddev_type))
    # @torch.no_grad()
    # def debug_run_denoising_diffusion(self, model, num_samples, shape, x_noisy, timestep,
    #        temp=1.0, enable_autocast=False, is_image=False, prior_var=1.0,
    #        condition_input=None):
    #    """
    #    Run the full denoising sampling loop.
    #    """
    #    # set model to eval mode
    #    # initialize sample
    #    #x_noisy_size = [num_samples] + shape
    #    #x_noisy = torch.randn(size=x_noisy_size, device='cuda') ## * np.sqrt(prior_var) * temp
    #    model.eval()
    #    x_noisy_size = x_noisy.shape

    #    x_noisy = x_noisy[0:1].expand(x_noisy.shape[0],-1,-1,-1) #
    #    timestep_start = timestep[0].item()
    #    output_list = []
    #    output_pred_list = []
    #    logger.info('timestep_start: {}', timestep_start)
    #    # denoising loop
    #    for t in reversed(range(0, self._diffusion_steps)):
    #        if t > timestep_start:
    #            continue
    #        if t % 100 == 0:
    #            logger.info('t={}', t)
    #        timestep = torch.ones(num_samples, dtype=torch.int64, device='cuda') * (t+1)  # the model uses (1 ... T) without 0
    #        fixed_log_scales = self.get_p_log_scales(timestep=timestep, stddev_type=self._denoising_stddevs)
    #        mixing_component = self.get_mixing_component(x_noisy, timestep, enabled=model.mixed_prediction)

    #        # run model
    #        with autocast(enable_autocast):
    #            pred_logits = model(x=x_noisy, t=timestep.float() , condition_input=condition_input)
    #            # pred_logits = model(x_noisy, timestep.float() / self._diffusion_steps)
    #            logits = utils.get_mixed_prediction(model.mixed_prediction, pred_logits, model.mixing_logit, mixing_component)

    #        output_dist = utils.decoder_output('place_holder', logits, fixed_log_scales=fixed_log_scales)
    #        noise = torch.randn(size=x_noisy_size, device='cuda')
    #        mean = self.get_q_posterior_mean(x_noisy, output_dist.means, t)

    #        _, var_t_p, m_t_p, _, _, _ = self.iw_quantities_t(
    #                num_samples, timestep)
    #        pred_eps_t0 = (x_noisy - torch.sqrt(var_t_p) * pred_logits) / m_t_p
    #        if t == 0:
    #            x_image = mean
    #        else:
    #            x_noisy = mean + torch.exp(output_dist.log_scales) * noise * temp
    #        output_list.append(x_noisy)
    #        output_pred_list.append(pred_eps_t0)
    #    if is_image:
    #        x_image = x_image.clamp(min=-1., max=1.)
    #        x_image = utils.unsymmetrize_image_data(x_image)
    #    model.train()
    #    return x_image, output_list, output_pred_list

    @torch.no_grad()
    def run_denoising_diffusion(self, model, num_samples, shape, temp=1.0,
                                enable_autocast=False, is_image=False, prior_var=1.0,
                                condition_input=None, given_noise=None, clip_feat=None, cls_emb=None, grid_emb=None):
        """
        Run the full denoising sampling loop.
        """
        # set model to eval mode
        model.eval()

        # initialize sample
        x_noisy_size = [num_samples] + shape
        if given_noise is None:
            # * np.sqrt(prior_var) * temp
            x_noisy = torch.randn(size=x_noisy_size, device='cuda')
        else:
            x_noisy = given_noise[0]
        output_list = {}
        output_list['pred_x'] = []
        # output_list['init_x_noisy'] = x_noisy
        # output_list['input_x'] = []
        # output_list['input_t'] = []
        # output_list['output_e'] = []
        # output_list['noise_t'] = []
        # output_list['condition_input'] = []
        # denoising loop
        kwargs = {}
        if grid_emb is not None:
            kwargs['grid_emb'] = grid_emb
        for t in reversed(range(0, self._diffusion_steps)):
            if t % 500 == 0:
                logger.info('t={}; shape={}, num_samples={}, sample shape: {}',
                            t, shape, num_samples, x_noisy.shape)
            # the model uses (1 ... T) without 0
            timestep = torch.ones(
                num_samples, dtype=torch.int64, device='cuda') * (t+1)
            fixed_log_scales = self.get_p_log_scales(
                timestep=timestep, stddev_type=self._denoising_stddevs)
            mixing_component = self.get_mixing_component(
                x_noisy, timestep, enabled=model.mixed_prediction)

            # run model
            with autocast(enable_autocast):
                if cls_emb is not None and condition_input is not None:
                    condition_input = torch.cat(
                        [condition_input, cls_emb], dim=1)
                elif cls_emb is not None and condition_input is None:
                    condition_input = cls_emb
                # output_list['input_x'].append(x_noisy)
                # output_list['input_t'].append(timestep)
                # output_list['condition_input'].append(condition_input)

                pred_logits = model(x=x_noisy, t=timestep.float(),
                                    condition_input=condition_input, clip_feat=clip_feat, **kwargs)
                # output_list['output_e'].append(pred_logits)

                # pred_logits = model(x_noisy, timestep.float() / self._diffusion_steps)
                logits = utils.get_mixed_prediction(
                    model.mixed_prediction, pred_logits, model.mixing_logit, mixing_component)

            output_dist = utils.decoder_output(
                'place_holder', logits, fixed_log_scales=fixed_log_scales)
            if given_noise is None:
                noise = torch.randn(size=x_noisy_size, device='cuda')
            else:
                # torch.randn(size=x_noisy_size, device='cuda')
                noise = given_noise[1][t]

            mean = self.get_q_posterior_mean(x_noisy, output_dist.means, t)
            if t == 0:
                x_image = mean
            else:
                x_noisy = mean + \
                    torch.exp(output_dist.log_scales) * noise * temp
            # output_list['noise_t'].append(noise)
            output_list['pred_x'].append(x_noisy)
        if is_image:
            x_image = x_image.clamp(min=-1., max=1.)
            x_image = utils.unsymmetrize_image_data(x_image)
        model.train()
        return x_image, output_list

    def run_ddim_forward(self, dae, eps, ddim_step, ddim_skip_type, condition_input=None, clip_feat=None):
        ## raise NotImplementedError
        """ calculates NLL based on ODE framework, assuming integration cutoff ode_eps """
        model.eval()

        # initialize sample
        x_noisy_size = [num_samples] + shape
        x_noisy = torch.randn(
            size=x_noisy_size, device='cuda') if x_noisy is None else x_noisy.cuda()
        output_list = []
        S = ddim_step

        # even spaced t
        if skip_type == 'uniform':
            c = (self._diffusion_steps - 1.0) / (S - 1.0)
            list_tau = [np.floor(i * c) for i in range(S)]
            list_tau = [int(s) for s in list_tau]
        elif skip_type == 'quad':
            seq = (np.linspace(
                   0, np.sqrt(self._diffusion_steps * 0.8), S
                   ) ** 2
                   )
            list_tau = [int(s) for s in list(seq)]

        user_defined_steps = sorted(list(list_tau), reverse=True)
        T_user = len(user_defined_steps)
        kwargs = {}
        if grid_emb is not None:
            kwargs['grid_emb'] = grid_emb

        def ode_func(t, x):
            """ the ode function (including log probability integration for NLL calculation) """
            global nfe_counter
            nfe_counter = nfe_counter + 1

            x = x.detach()
            x.requires_grad_(False)
            with torch.set_grad_enabled(False):
                with autocast(enabled=enable_autocast):
                    variance = self.var(t=t)
                    mixing_component = self.mixing_component(
                        x_noisy=x, var_t=variance, t=t, enabled=dae.mixed_prediction)
                    pred_params = dae(
                        x=x, t=t, condition_input=condition_input, clip_feat=clip_feat)
                    # Warning: here mixing_logit can be NOne
                    params = get_mixed_prediction(
                        dae.mixed_prediction, pred_params, dae.mixing_logit, mixing_component)
                    dx_dt = self.f(t=t) * x + 0.5 * self.g2(t=t) * \
                        params / torch.sqrt(variance)
                    # dx_dt = - 0.5 * self.g2(t=t) * (x - params / torch.sqrt(variance))

                # with autocast(enabled=False):
                #    dlogp_x_dt = -trace_df_dx_hutchinson(dx_dt, x, noise, no_autograd).view(x.shape[0], 1)

            return dx_dt

        # NFE counter
        global nfe_counter

        nll_all, nfe_all = [], []
        for i in range(num_samples):
            # integrated log probability
            # logp_diff_t0 = torch.zeros(eps.shape[0], 1, device='cuda')

            nfe_counter = 0

            # solve the ODE
            x_t = odeint(
                ode_func,
                eps,
                torch.tensor([ode_eps, 1.0], device='cuda'),
                atol=ode_solver_tol,  # 1e-5
                rtol=ode_solver_tol,  # 1e-5
                # 'dopri5' or 'dopri8' methods also seems good.
                method="scipy_solver",
                options={"solver": 'RK45'},  # only for scipy solvers
            )
            x_t0 = x_t[-1]

            nfe_all.append(nfe_counter)
            print('nfe_counter: ', nfe_counter)

        return x_t0

    @torch.no_grad()
    def run_ddim(self, model, num_samples, shape, temp=1.0, enable_autocast=False, is_image=True, prior_var=1.0,
                 condition_input=None, ddim_step=100, skip_type='uniform', kappa=1.0, clip_feat=None, grid_emb=None,
                 x_noisy=None, dae_index=-1):
        """
        Run the full denoising sampling loop.
        kappa = 1.0  # this one is the eta in DDIM algorithm 
        """
        # set model to eval mode
        model.eval()

        # initialize sample
        x_noisy_size = [num_samples] + shape
        x_noisy = torch.randn(
            size=x_noisy_size, device='cuda') if x_noisy is None else x_noisy.cuda()
        output_list = []
        S = ddim_step

        # even spaced t
        if skip_type == 'uniform':
            c = (self._diffusion_steps - 1.0) / (S - 1.0)
            list_tau = [np.floor(i * c) for i in range(S)]
            list_tau = [int(s) for s in list_tau]
        elif skip_type == 'quad':
            seq = (np.linspace(
                   0, np.sqrt(self._diffusion_steps * 0.8), S
                   ) ** 2
                   )
            list_tau = [int(s) for s in list(seq)]

        user_defined_steps = sorted(list(list_tau), reverse=True)
        T_user = len(user_defined_steps)
        kwargs = {}
        if grid_emb is not None:
            kwargs['grid_emb'] = grid_emb
        # denoising loop
        # for t in user_defined_steps: ## reversed(range(0, self._diffusion_steps)):
        Alpha_bar = self._alpha_bars  # self.var_sched.alphas_cumprod
        # the following computation is the same as the function in https://github.com/ermongroup/ddim/blob/51cb290f83049e5381b09a4cc0389f16a4a02cc9/functions/denoising.py#L10
        for i, t in enumerate(user_defined_steps):
            if i % 500 == 0:
                logger.info('t={} / {}, ori={}', i, S, self._diffusion_steps)
            tau = t
            # the model uses (1 ... T) without 0
            timestep = torch.ones(
                num_samples, dtype=torch.int64, device='cuda') * (t+1)
            fixed_log_scales = self.get_p_log_scales(
                timestep=timestep, stddev_type=self._denoising_stddevs)
            mixing_component = self.get_mixing_component(
                x_noisy, timestep, enabled=model.mixed_prediction)

            # --- copied --- #
            if i == T_user - 1:  # the next step is to generate x_0
                assert t == 0
                alpha_next = torch.tensor(1.0)
                sigma = torch.tensor(0.0)
            else:
                alpha_next = Alpha_bar[user_defined_steps[i+1]]
                sigma = kappa * \
                    torch.sqrt(
                        (1-alpha_next) / (1-Alpha_bar[tau]) * (1 - Alpha_bar[tau] / alpha_next))

            x = x_noisy * torch.sqrt(alpha_next / Alpha_bar[tau])
            c = torch.sqrt(1 - alpha_next - sigma ** 2) - torch.sqrt(1 -
                                                                     Alpha_bar[tau]) * torch.sqrt(alpha_next / Alpha_bar[tau])

            # --- run model forward --- #
            with autocast(enable_autocast):
                pred_logits = model(x=x_noisy, t=timestep.float(
                ), condition_input=condition_input, clip_feat=clip_feat, **kwargs)
                # pred_logits = model(x_noisy, timestep.float() / self._diffusion_steps)
                logits = utils.get_mixed_prediction(
                    model.mixed_prediction, pred_logits, model.mixing_logit, mixing_component)
            epsilon_theta = logits
            # xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            # x_{t-1} = c * et + sigma * randn + sqrt(alpha_next / alpha_bar_t) * x_t
            x += c * epsilon_theta + sigma * \
                torch.randn(x_noisy_size).to(x.device)
            x_noisy = x
            output_list.append(x_noisy)
        # if is_image:
        #     x_image = x_image.clamp(min=-1., max=1.)
        #     x_image = utils.unsymmetrize_image_data(x_image)
        model.train()
        return x_noisy, output_list

    def get_q_posterior_mean(self, x_noisy, prediction, t):
        # last step works differently (for better FIDs we NEVER sample in last conditional images output!)
        # Line 4 in algorithm 2 in DDPM:
        if t == 0:
            mean = 1.0 / torch.sqrt(self._alpha_bars[0]) * \
                (x_noisy - torch.sqrt(1.0 - self._alpha_bars[0]) * prediction)
        else:
            mean = 1.0 / torch.sqrt(self._alphas[t]) * \
                (x_noisy - self._betas_init[t] * prediction /
                 torch.sqrt(1.0 - self._alpha_bars[t]))

        return mean

    def get_mixing_component(self, x_noisy, timestep, enabled):
        size = x_noisy.size()
        alpha_bars = torch.gather(self._alpha_bars, 0, timestep-1)
        if enabled:
            one_minus_alpha_bars_sqrt = utils.view4D(
                torch.sqrt(1.0 - alpha_bars), size)
            mixing_component = one_minus_alpha_bars_sqrt * x_noisy
        else:
            mixing_component = None

        return mixing_component

    def mixing_component(self, eps, var, t, enabled):
        return self.get_mixing_component(eps, t, enabled)

    @torch.no_grad()
    def run_denoising_diffusion_from_t(self, model, num_samples, shape, time_start, x_noisy,
                                       temp=1.0, enable_autocast=False, is_image=False, prior_var=1.0,
                                       condition_input=None, given_noise=None):
        """
        Run the full denoising sampling loop.
        given_noise: Nstep,*x_noisy_size 
        """
        # set model to eval mode
        model.eval()

        # initialize sample
        x_noisy_size = [num_samples] + shape
        # if given_noise is None:
        ##    raise ValueError('given_noise is required')
        # raise NotImplementedError
        # x_noisy = torch.randn(size=x_noisy_size, device='cuda') ## * np.sqrt(prior_var) * temp
        # else:
        ##    x_noisy = given_noise[0]

        output_list = []
        # denoising loop
        for t in reversed(range(0, time_start)):  # self._diffusion_steps)):
            # if t % 100 == 0:
            #    logger.info('t={}', t)
            # the model uses (1 ... T) without 0
            timestep = torch.ones(
                num_samples, dtype=torch.int64, device='cuda') * (t+1)
            fixed_log_scales = self.get_p_log_scales(
                timestep=timestep, stddev_type=self._denoising_stddevs)
            mixing_component = self.get_mixing_component(
                x_noisy, timestep, enabled=model.mixed_prediction)

            # run model
            with autocast(enable_autocast):
                pred_logits = model(
                    x=x_noisy, t=timestep.float(), condition_input=condition_input)
                # pred_logits = model(x_noisy, timestep.float() / self._diffusion_steps)
                logits = utils.get_mixed_prediction(
                    model.mixed_prediction, pred_logits, model.mixing_logit, mixing_component)

            output_dist = utils.decoder_output(
                'place_holder', logits, fixed_log_scales=fixed_log_scales)
            if given_noise is None:
                noise = torch.randn(size=x_noisy_size, device='cuda')
            else:
                # torch.randn(size=x_noisy_size, device='cuda')
                noise = given_noise[1][t]

            mean = self.get_q_posterior_mean(x_noisy, output_dist.means, t)
            if t == 0:  # < 10:
                x_image = mean
            else:
                x_noisy = mean + \
                    torch.exp(output_dist.log_scales) * noise * temp
            output_list.append(x_noisy)
        # if is_image:
        #     x_image = x_image.clamp(min=-1., max=1.)
        #     x_image = utils.unsymmetrize_image_data(x_image)
        model.train()
        return x_image, output_list
