# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
"""modified from https://github.com/NVlabs/LSGM/blob/5eae2f385c014f2250c3130152b6be711f6a3a5a/diffusion_continuous.py"""
from abc import ABC, abstractmethod
import numpy as np
import torch
import gc
# import utils.distributions as distributions
from utils.utils import trace_df_dx_hutchinson, sample_gaussian_like, sample_rademacher_like, get_mixed_prediction
from third_party.torchdiffeq.torchdiffeq import odeint
from torch.cuda.amp import autocast
from timeit import default_timer as timer
from loguru import logger


def make_diffusion(args):
    """ simple diffusion factory function to return diffusion instances. Only use this to create continuous diffusions """
    if args.sde_type == 'geometric_sde':
        return DiffusionGeometric(args)
    elif args.sde_type == 'vpsde':
        return DiffusionVPSDE(args)
    elif args.sde_type == 'sub_vpsde':
        return DiffusionSubVPSDE(args)
    elif args.sde_type == 'power_vpsde':
        return DiffusionPowerVPSDE(args)
    elif args.sde_type == 'sub_power_vpsde':
        return DiffusionSubPowerVPSDE(args)
    elif args.sde_type == 'vesde':
        return DiffusionVESDE(args)
    else:
        raise ValueError("Unrecognized sde type: {}".format(args.sde_type))


class DiffusionBase(ABC):
    """
    Abstract base class for all diffusion implementations.
    """

    def __init__(self, args):
        super().__init__()
        self.sigma2_0 = args.sigma2_0
        self.sde_type = args.sde_type

    @abstractmethod
    def f(self, t):
        """ returns the drift coefficient at time t: f(t) """
        pass

    @abstractmethod
    def g2(self, t):
        """ returns the squared diffusion coefficient at time t: g^2(t) """
        pass

    @abstractmethod
    def var(self, t):
        """ returns variance at time t, \sigma_t^2
            q(zt|z0) = N(zt; \mu_t(z0), \sigma_t^2 I)
        """
        pass

    @abstractmethod
    def e2int_f(self, t):
        """ returns e^{\int_0^t f(s) ds} which corresponds to the coefficient of mean at time t. """
        pass

    @abstractmethod
    def inv_var(self, var):
        """ inverse of the variance function at input variance var. """
        pass

    @abstractmethod
    def mixing_component(self, x_noisy, var_t, t, enabled):
        """ returns mixing component which is the optimal denoising model assuming that q(z_0) is N(0, 1) """
        pass

    def sample_q(self, x_init, noise, var_t, m_t):
        """ returns a sample from diffusion process at time t """
        return m_t * x_init + torch.sqrt(var_t) * noise

    def cross_entropy_const(self, ode_eps):
        """ returns cross entropy factor with variance according to ode integration cutoff ode_eps """
        # _, c, h, w = x_init.shape
        return 0.5 * (1.0 + torch.log(2.0 * np.pi * self.var(t=torch.tensor(ode_eps, device='cuda'))))

    def compute_ode_nll(self, dae, eps, ode_eps, ode_solver_tol, enable_autocast=False,
                        no_autograd=False, num_samples=1, report_std=False,
                        condition_input=None, clip_feat=None):
        ## raise NotImplementedError
        """ calculates NLL based on ODE framework, assuming integration cutoff ode_eps """
        # ODE solver starts consuming the CPU memory without this on large models
        # https://github.com/scipy/scipy/issues/10070
        gc.collect()

        dae.eval()

        def ode_func(t, x):
            """ the ode function (including log probability integration for NLL calculation) """
            global nfe_counter
            nfe_counter = nfe_counter + 1

            # x = state[0].detach()
            x = x.detach()
            x.requires_grad_(False)
            # noise = sample_gaussian_like(x)  # could also use rademacher noise (sample_rademacher_like)
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
            # last output values
            x_t0 = x_t[-1]
            ## x_t0, logp_diff_t0 = x_t[-1], logp_diff_t[-1]

            # prior
            # if self.sde_type == 'vesde':
            #    logp_prior = torch.sum(distributions.log_p_var_normal(x_t0, var=self.sigma2_max), dim=[1, 2, 3])
            # else:
            #    logp_prior = torch.sum(distributions.log_p_standard_normal(x_t0), dim=[1, 2, 3])

            #log_likelihood = logp_prior - logp_diff_t0.view(-1)

            # nll_all.append(-log_likelihood)
            nfe_all.append(nfe_counter)
            print('nfe_counter: ', nfe_counter)

        #nfe_mean = np.mean(nfe_all)
        ##nll_all = torch.stack(nll_all, dim=1)
        #nll_mean = torch.mean(nll_all, dim=1)
        # if num_samples > 1 and report_std:
        #    nll_stddev = torch.std(nll_all ,dii=1)
        #    nll_stddev_batch = torch.mean(nll_stddev)
        #    nll_stderror_batch = nll_stddev_batch / np.sqrt(num_samples)
        # else:
        #    nll_stddev_batch = None
        #    nll_stderror_batch = None
        return x_t0  # nll_mean, nfe_mean, nll_stddev_batch, nll_stderror_batch

    def sample_model_ode(self, dae, num_samples, shape, ode_eps,
                         ode_solver_tol, enable_autocast, temp, noise=None,
                         condition_input=None, mixing_logit=None,
                         use_cust_ode_func=0, init_t=1.0, return_all_sample=False, clip_feat=None
                         ):
        """ generates samples using the ODE framework, assuming integration cutoff ode_eps """
        # ODE solver starts consuming the CPU memory without this on large models
        # https://github.com/scipy/scipy/issues/10070
        gc.collect()

        dae.eval()

        def cust_ode_func(t, x):
            """ the ode function (sampling only, no NLL stuff) """
            global nfe_counter
            nfe_counter = nfe_counter + 1
            if nfe_counter % 100 == 0:
                logger.info('nfe_counter={}', nfe_counter)
            with autocast(enabled=enable_autocast):
                variance = self.var(t=t)
                params = dae(x, x, t, condition_input=condition_input)
                dx_dt = self.f(t=t) * x + 0.5 * self.g2(t=t) * \
                    params / torch.sqrt(variance)
                # dx_dt = - 0.5 * self.g2(t=t) * (x - params / torch.sqrt(variance))

            return dx_dt

        def ode_func(t, x):
            """ the ode function (sampling only, no NLL stuff) """
            global nfe_counter
            nfe_counter = nfe_counter + 1
            if nfe_counter % 100 == 0:
                logger.info('nfe_counter={}', nfe_counter)
            with autocast(enabled=enable_autocast):
                variance = self.var(t=t)
                mixing_component = self.mixing_component(
                    x_noisy=x, var_t=variance, t=t, enabled=dae.mixed_prediction)
                pred_params = dae(
                    x=x, t=t, condition_input=condition_input, clip_feat=clip_feat)
                input_mixing_logit = mixing_logit if mixing_logit is not None else dae.mixing_logit
                params = get_mixed_prediction(
                    dae.mixed_prediction, pred_params, input_mixing_logit, mixing_component)
                dx_dt = self.f(t=t) * x + 0.5 * self.g2(t=t) * \
                    params / torch.sqrt(variance)
                # dx_dt = - 0.5 * self.g2(t=t) * (x - params / torch.sqrt(variance))

            return dx_dt

        # the initial noise
        if noise is None:
            noise = torch.randn(size=[num_samples] + shape, device='cuda')

        if self.sde_type == 'vesde':
            noise_init = temp * noise * np.sqrt(self.sigma2_max)
        else:
            noise_init = temp * noise

        # NFE counter
        global nfe_counter
        nfe_counter = 0

        # solve the ODE
        start = timer()
        samples_out = odeint(
            ode_func if not use_cust_ode_func else cust_ode_func,
            noise_init,
            torch.tensor([init_t, ode_eps], device='cuda'),
            atol=ode_solver_tol,  # 1e-5
            rtol=ode_solver_tol,  # 1e-5
            # 'dopri5' or 'dopri8' methods also seems good.
            method="scipy_solver",
            options={"solver": 'RK45'},  # only for scipy solvers
        )
        end = timer()
        ode_solve_time = end - start
        if return_all_sample:
            return samples_out[-1], samples_out, nfe_counter, ode_solve_time
        return samples_out[-1], nfe_counter, ode_solve_time

    # def compute_dsm_nll(self, dae, eps, time_eps, enable_autocast, num_samples, report_std):
    #    with torch.no_grad():
    #        neg_log_prob_all = []
    #        for i in range(num_samples):
    #            assert self.sde_type in ['geometric_sde', 'vpsde', 'power_vpsde'], "we don't support subVPSDE yet."
    #            t, var_t, m_t, obj_weight_t, _, _ = \
    #                self.iw_quantities(eps.shape[0], time_eps, iw_sample_mode='ll_iw', iw_subvp_like_vp_sde=False)

    #            noise = torch.randn(size=eps.size(), device='cuda')
    #            eps_t = self.sample_q(eps, noise, var_t, m_t)
    #            mixing_component = self.mixing_component(eps_t, var_t, t, enabled=dae.mixed_prediction)
    #            with autocast(enabled=enable_autocast):
    #                pred_params = dae(eps_t, t)
    #                params = get_mixed_prediction(dae.mixed_prediction, pred_params, dae.mixing_logit, mixing_component)
    #                l2_term = torch.square(params - noise)

    #            neg_log_prob_per_var = obj_weight_t * l2_term
    #            neg_log_prob_per_var += self.cross_entropy_const(time_eps)
    #            neg_log_prob = torch.sum(neg_log_prob_per_var, dim=[1, 2, 3])

    #            neg_log_prob_all.append(neg_log_prob)

    #        neg_log_prob_all = torch.stack(neg_log_prob_all, dim=1)
    #        nll = torch.mean(neg_log_prob_all, dim=1)
    #        if num_samples > 1 and report_std:
    #            nll_std = torch.std(neg_log_prob_all, dim=1)
    #            print('std nll:', nll_std)

    #    return nll

    def iw_quantities(self, size, time_eps, iw_sample_mode, iw_subvp_like_vp_sde):
        if self.sde_type in ['geometric_sde', 'vpsde', 'power_vpsde']:
            return self._iw_quantities_vpsdelike(size, time_eps, iw_sample_mode)
        elif self.sde_type in ['sub_vpsde', 'sub_power_vpsde']:
            return self._iw_quantities_subvpsdelike(size, time_eps, iw_sample_mode, iw_subvp_like_vp_sde)
        elif self.sde_type in ['vesde']:
            return self._iw_quantities_vesde(size, time_eps, iw_sample_mode)
        else:
            raise NotImplementedError

    def debug_sheduler(self, time_eps):
        # time_eps, 1-time_eps, 1000) ##-1) / 1000.0 + time_eps
        t = torch.linspace(0, 1, 1000)
        t = torch.range(1, 1000) / 1000.0
        t = t.cuda()
        var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
        obj_weight_t_p = torch.ones(1, device='cuda')
        obj_weight_t_q = g2_t / (2.0 * var_t)
        return t, var_t.view(-1, 1, 1, 1), m_t.view(-1, 1, 1, 1), \
            obj_weight_t_p.view(-1, 1, 1, 1), \
            obj_weight_t_q.view(-1, 1, 1, 1), g2_t.view(-1, 1, 1, 1)

    def _iw_quantities_vpsdelike(self, size, time_eps, iw_sample_mode):
        """
        For all SDEs where the underlying SDE is of the form dz = -0.5 * beta(t) * z * dt + sqrt{beta(t)} * dw, like
        for the VPSDE.
        """
        rho = torch.rand(size=[size], device='cuda')

        if iw_sample_mode == 'll_uniform':
            # uniform t sampling - likelihood obj. for both q and p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t_p = obj_weight_t_q = g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'll_iw':
            # importance sampling for likelihood obj. - likelihood obj. for both q and p
            ones = torch.ones_like(rho, device='cuda')
            sigma2_1, sigma2_eps = self.var(ones), self.var(time_eps * ones)
            log_sigma2_1, log_sigma2_eps = torch.log(
                sigma2_1), torch.log(sigma2_eps)
            var_t = torch.exp(rho * log_sigma2_1 + (1 - rho) * log_sigma2_eps)
            t = self.inv_var(var_t)
            m_t, g2_t = self.e2int_f(t), self.g2(t)
            obj_weight_t_p = obj_weight_t_q = 0.5 * \
                (log_sigma2_1 - log_sigma2_eps) / (1.0 - var_t)

        elif iw_sample_mode == 'drop_all_uniform':
            # uniform t sampling - likelihood obj. for q, all-prefactors-dropped obj. for p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t_p = torch.ones(1, device='cuda')
            obj_weight_t_q = g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'drop_all_iw':
            # importance sampling for all-pref.-dropped obj. - likelihood obj. for q, all-pref.-dropped obj. for p
            assert self.sde_type == 'vpsde', 'Importance sampling for fully unweighted objective is currently only ' \
                                             'implemented for the regular VPSDE.'
            t = torch.sqrt(1.0 / self.delta_beta_half) * torch.erfinv(rho *
                                                                      self.const_norm_2 + self.const_erf) - self.beta_frac
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t_p = self.const_norm / (1.0 - var_t)
            obj_weight_t_q = obj_weight_t_p * g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'drop_sigma2t_iw':
            # importance sampling for inv_sigma2_t-dropped obj. - likelihood obj. for q, inv_sigma2_t-dropped obj. for p
            ones = torch.ones_like(rho, device='cuda')
            sigma2_1, sigma2_eps = self.var(ones), self.var(time_eps * ones)
            var_t = rho * sigma2_1 + (1 - rho) * sigma2_eps
            t = self.inv_var(var_t)
            m_t, g2_t = self.e2int_f(t), self.g2(t)
            obj_weight_t_p = 0.5 * (sigma2_1 - sigma2_eps) / (1.0 - var_t)
            obj_weight_t_q = obj_weight_t_p / var_t

        elif iw_sample_mode == 'drop_sigma2t_uniform':
            # uniform sampling for inv_sigma2_t-dropped obj. - likelihood obj. for q, inv_sigma2_t-dropped obj. for p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t_p = g2_t / 2.0
            obj_weight_t_q = g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'rescale_iw':
            # importance sampling for 1/(1-sigma2_t) resc. obj. - likelihood obj. for q, 1/(1-sigma2_t) resc. obj. for p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t_p = 0.5 / (1.0 - var_t)
            obj_weight_t_q = g2_t / (2.0 * var_t)

        else:
            raise ValueError(
                "Unrecognized importance sampling type: {}".format(iw_sample_mode))

        return t, var_t.view(-1, 1, 1, 1), m_t.view(-1, 1, 1, 1), obj_weight_t_p.view(-1, 1, 1, 1), \
            obj_weight_t_q.view(-1, 1, 1, 1), g2_t.view(-1, 1, 1, 1)

    # def _iw_quantities_subvpsdelike(self, size, time_eps, iw_sample_mode, iw_subvp_like_vp_sde):
    #    """
    #    For all SDEs where the underlying SDE is of the form
    #    dz = -0.5 * beta(t) * z * dt + sqrt{beta(t) * (1 - exp[-2 * betaintegral])} * dw, like for the Sub-VPSDE.
    #    When iw_subvp_like_vp_sde is True, then we define the importance sampling distributions based on an analogous
    #    VPSDE, while stile using the Sub-VPSDE. The motivation is that deriving the correct importance sampling
    #    distributions for the Sub-VPSDE itself is hard, but the importance sampling distributions from analogous VPSDEs
    #    probably already significantly reduce the variance also for the Sub-VPSDE.
    #    """
    #    rho = torch.rand(size=[size], device='cuda')

    #    if iw_sample_mode == 'll_uniform':
    #        # uniform t sampling - likelihood obj. for both q and p
    #        t = rho * (1. - time_eps) + time_eps
    #        var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
    #        obj_weight_t_p = obj_weight_t_q = g2_t / (2.0 * var_t)

    #    elif iw_sample_mode == 'll_iw':
    #        if iw_subvp_like_vp_sde:
    #            # importance sampling for vpsde likelihood obj. - sub-vpsde likelihood obj. for both q and p
    #            ones = torch.ones_like(rho, device='cuda')
    #            sigma2_1, sigma2_eps = self.var_vpsde(ones), self.var_vpsde(time_eps * ones)
    #            log_sigma2_1, log_sigma2_eps = torch.log(sigma2_1), torch.log(sigma2_eps)
    #            var_t_vpsde = torch.exp(rho * log_sigma2_1 + (1 - rho) * log_sigma2_eps)
    #            t = self.inv_var_vpsde(var_t_vpsde)
    #            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
    #            obj_weight_t_p = obj_weight_t_q = g2_t / (2.0 * var_t) * \
    #                (log_sigma2_1 - log_sigma2_eps) * var_t_vpsde / (1 - var_t_vpsde) / self.beta(t)
    #        else:
    #            raise NotImplementedError

    #    elif iw_sample_mode == 'drop_all_uniform':
    #        # uniform t sampling - likelihood obj. for q, all-prefactors-dropped obj. for p
    #        t = rho * (1. - time_eps) + time_eps
    #        var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
    #        obj_weight_t_p = torch.ones(1, device='cuda')
    #        obj_weight_t_q = g2_t / (2.0 * var_t)

    #    elif iw_sample_mode == 'drop_all_iw':
    #        if iw_subvp_like_vp_sde:
    #            # importance sampling for all-pref.-dropped obj. - likelihood obj. for q, all-pref.-dropped obj. for p
    #            assert self.sde_type == 'sub_vpsde', 'Importance sampling for fully unweighted objective is ' \
    #                                                 'currently only implemented for the Sub-VPSDE.'
    #            t = torch.sqrt(1.0 / self.delta_beta_half) * torch.erfinv(rho * self.const_norm_2 + self.const_erf) - self.beta_frac
    #            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
    #            obj_weight_t_p = self.const_norm / (1.0 - self.var_vpsde(t))
    #            obj_weight_t_q = obj_weight_t_p * g2_t / (2.0 * var_t)
    #        else:
    #            raise NotImplementedError

    #    elif iw_sample_mode == 'drop_sigma2t_iw':
    #        if iw_subvp_like_vp_sde:
    #            # importance sampling for inv_sigma2_t-dropped obj. - likelihood obj. for q, inv_sigma2_t-dropped obj. for p
    #            ones = torch.ones_like(rho, device='cuda')
    #            sigma2_1, sigma2_eps = self.var_vpsde(ones), self.var_vpsde(time_eps * ones)
    #            var_t_vpsde = rho * sigma2_1 + (1 - rho) * sigma2_eps
    #            t = self.inv_var_vpsde(var_t_vpsde)
    #            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
    #            obj_weight_t_p = 0.5 * g2_t / self.beta(t) * (sigma2_1 - sigma2_eps) / (1.0 - var_t_vpsde)
    #            obj_weight_t_q = obj_weight_t_p / var_t
    #        else:
    #            raise NotImplementedError

    #    elif iw_sample_mode == 'drop_sigma2t_uniform':
    #        # uniform sampling for inv_sigma2_t-dropped obj. - likelihood obj. for q, inv_sigma2_t-dropped obj. for p
    #        t = rho * (1. - time_eps) + time_eps
    #        var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
    #        obj_weight_t_p = g2_t / 2.0
    #        obj_weight_t_q = g2_t / (2.0 * var_t)

    #    elif iw_sample_mode == 'rescale_iw':
    #        # importance sampling for 1/(1-sigma2_t) resc. obj. - likelihood obj. for q, 1/(1-sigma2_t) resc. obj. for p
    #        # Note that we use the sub-vpsde variance to scale the p objective! It's not clear what's optimal here!
    #        t = rho * (1. - time_eps) + time_eps
    #        var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
    #        obj_weight_t_p = 0.5 / (1.0 - var_t)
    #        obj_weight_t_q = g2_t / (2.0 * var_t)

    #    else:
    #        raise ValueError("Unrecognized importance sampling type: {}".format(iw_sample_mode))

    #    return t, var_t.view(-1, 1, 1, 1), m_t.view(-1, 1, 1, 1), obj_weight_t_p.view(-1, 1, 1, 1), \
    #        obj_weight_t_q.view(-1, 1, 1, 1), g2_t.view(-1, 1, 1, 1)

    # def _iw_quantities_vesde(self, size, time_eps, iw_sample_mode):
    #    """
    #    For the VESDE.
    #    """
    #    rho = torch.rand(size=[size], device='cuda')

    #    if iw_sample_mode == 'll_uniform':
    #        # uniform t sampling - likelihood obj. for both q and p
    #        t = rho * (1. - time_eps) + time_eps
    #        var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
    #        obj_weight_t_p = obj_weight_t_q = g2_t / (2.0 * var_t)

    #    elif iw_sample_mode == 'll_iw':
    #        # importance sampling for likelihood obj. - likelihood obj. for both q and p
    #        ones = torch.ones_like(rho, device='cuda')
    #        nsigma2_1, nsigma2_eps, sigma2_eps = self.var_N(ones), self.var_N(time_eps * ones), self.var(time_eps * ones)
    #        log_frac_sigma2_1, log_frac_sigma2_eps = torch.log(self.sigma2_max / nsigma2_1), torch.log(nsigma2_eps / sigma2_eps)
    #        var_N_t = (1.0 - self.sigma2_min) / (1.0 - torch.exp(rho * (log_frac_sigma2_1 + log_frac_sigma2_eps) - log_frac_sigma2_eps))
    #        t = self.inv_var_N(var_N_t)
    #        var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
    #        obj_weight_t_p = obj_weight_t_q = 0.5 * (log_frac_sigma2_1 + log_frac_sigma2_eps) * self.var_N(t) / (1.0 - self.sigma2_min)

    #    elif iw_sample_mode == 'drop_all_uniform':
    #        # uniform t sampling - likelihood obj. for q, all-prefactors-dropped obj. for p
    #        t = rho * (1. - time_eps) + time_eps
    #        var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
    #        obj_weight_t_p = torch.ones(1, device='cuda')
    #        obj_weight_t_q = g2_t / (2.0 * var_t)

    #    elif iw_sample_mode == 'drop_all_iw':
    #        # importance sampling for all-pref.-dropped obj. - likelihood obj. for q, all-pref.-dropped obj. for p
    #        ones = torch.ones_like(rho, device='cuda')
    #        nsigma2_1, nsigma2_eps, sigma2_eps = self.var_N(ones), self.var_N(time_eps * ones), self.var(time_eps * ones)
    #        log_frac_sigma2_1, log_frac_sigma2_eps = torch.log(self.sigma2_max / nsigma2_1), torch.log(nsigma2_eps / sigma2_eps)
    #        var_N_t = (1.0 - self.sigma2_min) / (1.0 - torch.exp(rho * (log_frac_sigma2_1 + log_frac_sigma2_eps) - log_frac_sigma2_eps))
    #        t = self.inv_var_N(var_N_t)
    #        var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
    #        obj_weight_t_q = 0.5 * (log_frac_sigma2_1 + log_frac_sigma2_eps) * self.var_N(t) / (1.0 - self.sigma2_min)
    #        obj_weight_t_p = 2.0 * obj_weight_t_q / np.log(self.sigma2_max / self.sigma2_min)

    #    elif iw_sample_mode == 'drop_sigma2t_iw':
    #        # importance sampling for inv_sigma2_t-dropped obj. - likelihood obj. for q, inv_sigma2_t-dropped obj. for p
    #        ones = torch.ones_like(rho, device='cuda')
    #        nsigma2_1, nsigma2_eps = self.var_N(ones), self.var_N(time_eps * ones)
    #        var_N_t = torch.exp(rho * torch.log(nsigma2_1) + (1 - rho) * torch.log(nsigma2_eps))
    #        t = self.inv_var_N(var_N_t)
    #        var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
    #        obj_weight_t_p = 0.5 * torch.log(nsigma2_1 / nsigma2_eps) * self.var_N(t)
    #        obj_weight_t_q = obj_weight_t_p / var_t

    #    elif iw_sample_mode == 'drop_sigma2t_uniform':
    #        # uniform sampling for inv_sigma2_t-dropped obj. - likelihood obj. for q, inv_sigma2_t-dropped obj. for p
    #        t = rho * (1. - time_eps) + time_eps
    #        var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
    #        obj_weight_t_p = g2_t / 2.0
    #        obj_weight_t_q = g2_t / (2.0 * var_t)

    #    elif iw_sample_mode == 'rescale_iw':
    #        # uniform sampling for 1/(1-sigma2_t) resc. obj. - likelihood obj. for q, 1/(1-sigma2_t) resc. obj. for p
    #        t = rho * (1. - time_eps) + time_eps
    #        var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
    #        obj_weight_t_p = 0.5 / (1.0 - var_t)
    #        obj_weight_t_q = g2_t / (2.0 * var_t)

    #    else:
    #        raise ValueError("Unrecognized importance sampling type: {}".format(iw_sample_mode))

    #    return t, var_t.view(-1, 1, 1, 1), m_t.view(-1, 1, 1, 1), obj_weight_t_p.view(-1, 1, 1, 1), \
    #        obj_weight_t_q.view(-1, 1, 1, 1), g2_t.view(-1, 1, 1, 1)


# class DiffusionGeometric(DiffusionBase):
#    """
#    Diffusion implementation with dz = -0.5 * beta(t) * z * dt + sqrt(beta(t)) * dW SDE and geometric progression of
#    variance. This is our new diffusion.
#    """
#    def __init__(self, args):
#        super().__init__(args)
#        self.sigma2_min = args.sigma2_min
#        self.sigma2_max = args.sigma2_max
#
#    def f(self, t):
#        return -0.5 * self.g2(t)
#
#    def g2(self, t):
#        sigma2_geom = self.sigma2_min * ((self.sigma2_max / self.sigma2_min) ** t)
#        log_term = np.log(self.sigma2_max / self.sigma2_min)
#        return sigma2_geom * log_term / (1.0 - self.sigma2_0 + self.sigma2_min - sigma2_geom)
#
#    def var(self, t):
#        return self.sigma2_min * ((self.sigma2_max / self.sigma2_min) ** t) - self.sigma2_min + self.sigma2_0
#
#    def e2int_f(self, t):
#        return torch.sqrt(1.0 + self.sigma2_min * (1.0 - (self.sigma2_max / self.sigma2_min) ** t) / (1.0 - self.sigma2_0))
#
#    def inv_var(self, var):
#        return torch.log((var + self.sigma2_min - self.sigma2_0) / self.sigma2_min) / np.log(self.sigma2_max / self.sigma2_min)
#
#    def mixing_component(self, x_noisy, var_t, t, enabled):
#        if enabled:
#            return torch.sqrt(var_t) * x_noisy
#        else:
#            return None
#

class DiffusionVPSDE(DiffusionBase):
    """
    Diffusion implementation of the VPSDE. This uses the same SDE like DiffusionGeometric but with linear beta(t).
    Note that we need to scale beta_start and beta_end by 1000 relative to JH's DDPM values, since our t is in [0,1].
    """

    def __init__(self, args):
        super().__init__(args)
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end
        logger.info('VPSDE: beta_start={}, beta_end={}, sigma2_0={}',
                    self.beta_start, self.beta_end, self.sigma2_0)
        # auxiliary constants (yes, this is not super clean...)
        self.time_eps = args.time_eps
        self.delta_beta_half = torch.tensor(
            0.5 * (self.beta_end - self.beta_start), device='cuda')
        self.beta_frac = torch.tensor(
            self.beta_start / (self.beta_end - self.beta_start), device='cuda')
        self.const_aq = (1.0 - self.sigma2_0) * torch.exp(0.5 *
                                                          self.beta_frac) * torch.sqrt(0.25 * np.pi / self.delta_beta_half)
        self.const_erf = torch.erf(torch.sqrt(
            self.delta_beta_half) * (self.time_eps + self.beta_frac))
        self.const_norm = self.const_aq * \
            (torch.erf(torch.sqrt(self.delta_beta_half)
             * (1.0 + self.beta_frac)) - self.const_erf)
        self.const_norm_2 = torch.erf(torch.sqrt(
            self.delta_beta_half) * (1.0 + self.beta_frac)) - self.const_erf

    def f(self, t):
        return -0.5 * self.g2(t)

    def g2(self, t):
        return self.beta_start + (self.beta_end - self.beta_start) * t

    def var(self, t):
        return 1.0 - (1.0 - self.sigma2_0) * torch.exp(-self.beta_start * t - 0.5 * (self.beta_end - self.beta_start) * t * t)

    def e2int_f(self, t):
        return torch.exp(-0.5 * self.beta_start * t - 0.25 * (self.beta_end - self.beta_start) * t * t)

    def inv_var(self, var):
        c = torch.log((1 - var) / (1 - self.sigma2_0))
        a = self.beta_end - self.beta_start
        t = (-self.beta_start + torch.sqrt(np.square(self.beta_start) - 2 * a * c)) / a
        return t

    def mixing_component(self, x_noisy, var_t, t, enabled):
        if enabled:
            return torch.sqrt(var_t) * x_noisy
        else:
            return None


# class DiffusionSubVPSDE(DiffusionBase):
#    """
#    Diffusion implementation of the sub-VPSDE. Note that this uses a different SDE compared to the above two diffusions.
#    """
#    def __init__(self, args):
#        super().__init__(args)
#        self.beta_start = args.beta_start
#        self.beta_end = args.beta_end
#
#        # auxiliary constants (assumes regular VPSDE... yes, this is not super clean...)
#        self.time_eps = args.time_eps
#        self.delta_beta_half = torch.tensor(0.5 * (self.beta_end - self.beta_start), device='cuda')
#        self.beta_frac = torch.tensor(self.beta_start / (self.beta_end - self.beta_start), device='cuda')
#        self.const_aq = (1.0 - self.sigma2_0) * torch.exp(0.5 * self.beta_frac) * torch.sqrt(0.25 * np.pi / self.delta_beta_half)
#        self.const_erf = torch.erf(torch.sqrt(self.delta_beta_half) * (self.time_eps + self.beta_frac))
#        self.const_norm = self.const_aq * (torch.erf(torch.sqrt(self.delta_beta_half) * (1.0 + self.beta_frac)) - self.const_erf)
#        self.const_norm_2 = torch.erf(torch.sqrt(self.delta_beta_half) * (1.0 + self.beta_frac)) - self.const_erf
#
#    def f(self, t):
#        return -0.5 * self.beta(t)
#
#    def g2(self, t):
#        return self.beta(t) * (1.0 - torch.exp(-2.0 * self.beta_start * t - (self.beta_end - self.beta_start) * t * t))
#
#    def var(self, t):
#        int_term = torch.exp(-self.beta_start * t - 0.5 * (self.beta_end - self.beta_start) * t * t)
#        return torch.square(1.0 - int_term) + self.sigma2_0 * int_term
#
#    def e2int_f(self, t):
#        return torch.exp(-0.5 * self.beta_start * t - 0.25 * (self.beta_end - self.beta_start) * t * t)
#
#    def beta(self, t):
#        """ auxiliary beta function """
#        return self.beta_start + (self.beta_end - self.beta_start) * t
#
#    def inv_var(self, var):
#        raise NotImplementedError
#
#    def mixing_component(self, x_noisy, var_t, t, enabled):
#        if enabled:
#            int_term = torch.exp(-self.beta_start * t - 0.5 * (self.beta_end - self.beta_start) * t * t).view(-1, 1, 1, 1)
#            return torch.sqrt(var_t) * x_noisy / (torch.square(1.0 - int_term) + int_term)
#        else:
#            return None
#
#    def var_vpsde(self, t):
#        return 1.0 - (1.0 - self.sigma2_0) * torch.exp(-self.beta_start * t - 0.5 * (self.beta_end - self.beta_start) * t * t)
#
#    def inv_var_vpsde(self, var):
#        c = torch.log((1 - var) / (1 - self.sigma2_0))
#        a = self.beta_end - self.beta_start
#        t = (-self.beta_start + torch.sqrt(np.square(self.beta_start) - 2 * a * c)) / a
#        return t
#
#
# class DiffusionPowerVPSDE(DiffusionBase):
#    """
#    Diffusion implementation of the power-VPSDE. This uses the same SDE like DiffusionGeometric but with beta function
#    that is a power function with user specified power. Note that for power=1, this reproduces the vanilla
#    DiffusionVPSDE above.
#    """
#    def __init__(self, args):
#        super().__init__(args)
#        self.beta_start = args.beta_start
#        self.beta_end = args.beta_end
#        self.power = args.vpsde_power
#
#    def f(self, t):
#        return -0.5 * self.g2(t)
#
#    def g2(self, t):
#        return self.beta_start + (self.beta_end - self.beta_start) * t ** self.power
#
#    def var(self, t):
#        return 1.0 - (1.0 - self.sigma2_0) * torch.exp(-self.beta_start * t - (self.beta_end - self.beta_start) * t ** (self.power + 1) / (self.power + 1.0))
#
#    def e2int_f(self, t):
#        return torch.exp(-0.5 * self.beta_start * t - 0.5 * (self.beta_end - self.beta_start) * t ** (self.power + 1) / (self.power + 1.0))
#
#    def inv_var(self, var):
#        if self.power == 2:
#            c = torch.log((1 - var) / (1 - self.sigma2_0))
#            p = 3.0 * self.beta_start / (self.beta_end - self.beta_start)
#            q = 3.0 * c / (self.beta_end - self.beta_start)
#            a = -0.5 * q + torch.sqrt(q ** 2 / 4.0 + p ** 3 / 27.0)
#            b = -0.5 * q - torch.sqrt(q ** 2 / 4.0 + p ** 3 / 27.0)
#            return torch.pow(a, 1.0 / 3.0) + torch.pow(b, 1.0 / 3.0)
#        else:
#            raise NotImplementedError
#
#    def mixing_component(self, x_noisy, var_t, t, enabled):
#        if enabled:
#            return torch.sqrt(var_t) * x_noisy
#        else:
#            return None
#
#
# class DiffusionSubPowerVPSDE(DiffusionBase):
#    """
#    Diffusion implementation of the sub-power-VPSDE. This uses the same SDE like DiffusionSubVPSDE but with beta
#    function that is a power function with user specified power. Note that for power=1, this reproduces the vanilla
#    DiffusionSubVPSDE above.
#    """
#    def __init__(self, args):
#        super().__init__(args)
#        self.beta_start = args.beta_start
#        self.beta_end = args.beta_end
#        self.power = args.vpsde_power
#
#    def f(self, t):
#        return -0.5 * self.beta(t)
#
#    def g2(self, t):
#        return self.beta(t) * (1.0 - torch.exp(-2.0 * self.beta_start * t - 2.0 * (self.beta_end - self.beta_start) * t ** (self.power + 1) / (self.power + 1.0)))
#
#    def var(self, t):
#        int_term = torch.exp(-self.beta_start * t - (self.beta_end - self.beta_start) * t ** (self.power + 1) / (self.power + 1.0))
#        return torch.square(1.0 - int_term) + self.sigma2_0 * int_term
#
#    def e2int_f(self, t):
#        return torch.exp(-0.5 * self.beta_start * t - 0.5 * (self.beta_end - self.beta_start) * t ** (self.power + 1) / (self.power + 1.0))
#
#    def beta(self, t):
#        """ internal auxiliary beta function """
#        return self.beta_start + (self.beta_end - self.beta_start) * t ** self.power
#
#    def inv_var(self, var):
#        raise NotImplementedError
#
#    def mixing_component(self, x_noisy, var_t, t, enabled):
#        if enabled:
#            int_term = torch.exp(-self.beta_start * t - (self.beta_end - self.beta_start) * t ** (self.power + 1) / (self.power + 1.0)).view(-1, 1, 1, 1)
#            return torch.sqrt(var_t) * x_noisy / (torch.square(1.0 - int_term) + int_term)
#        else:
#            return None
#
#    def var_vpsde(self, t):
#        return 1.0 - (1.0 - self.sigma2_0) * torch.exp(-self.beta_start * t - (self.beta_end - self.beta_start) * t ** (self.power + 1) / (self.power + 1.0))
#
#    def inv_var_vpsde(self, var):
#        if self.power == 2:
#            c = torch.log((1 - var) / (1 - self.sigma2_0))
#            p = 3.0 * self.beta_start / (self.beta_end - self.beta_start)
#            q = 3.0 * c / (self.beta_end - self.beta_start)
#            a = -0.5 * q + torch.sqrt(q ** 2 / 4.0 + p ** 3 / 27.0)
#            b = -0.5 * q - torch.sqrt(q ** 2 / 4.0 + p ** 3 / 27.0)
#            return torch.pow(a, 1.0 / 3.0) + torch.pow(b, 1.0 / 3.0)
#        else:
#            raise NotImplementedError
#
#
# class DiffusionVESDE(DiffusionBase):
#    """
#    Diffusion implementation of the VESDE with dz = sqrt(beta(t)) * dW
#    """
#    def __init__(self, args):
#        super().__init__(args)
#        self.sigma2_min = args.sigma2_min
#        self.sigma2_max = args.sigma2_max
#        assert self.sigma2_min == self.sigma2_0, "VESDE was proposed implicitly assuming sigma2_min = sigma2_0!"
#
#    def f(self, t):
#        return torch.zeros_like(t, device='cuda')
#
#    def g2(self, t):
#        return self.sigma2_min * np.log(self.sigma2_max / self.sigma2_min) * ((self.sigma2_max / self.sigma2_min) ** t)
#
#    def var(self, t):
#        return self.sigma2_min * ((self.sigma2_max / self.sigma2_min) ** t) - self.sigma2_min + self.sigma2_0
#
#    def e2int_f(self, t):
#        return torch.ones_like(t, device='cuda')
#
#    def inv_var(self, var):
#        return torch.log((var + self.sigma2_min - self.sigma2_0) / self.sigma2_min) / np.log(self.sigma2_max / self.sigma2_min)
#
#    def mixing_component(self, x_noisy, var_t, t, enabled):
#        if enabled:
#            return torch.sqrt(var_t) * x_noisy / (self.sigma2_min * ((self.sigma2_max / self.sigma2_min) ** t.view(-1, 1, 1, 1)) - self.sigma2_min + 1.0)
#        else:
#            return None
#
#    def var_N(self, t):
#        return 1.0 - self.sigma2_min + self.sigma2_min * ((self.sigma2_max / self.sigma2_min) ** t)
#
#    def inv_var_N(self, var):
#        return torch.log((var + self.sigma2_min - 1.0) / self.sigma2_min) / np.log(self.sigma2_max / self.sigma2_min)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    class Foo:
        def __init__(self):
            self.sde_type = 'vpsde'
            self.sigma2_0 = 0.01
            self.sigma2_min = 3e-5
            self.sigma2_max = 0.999
            self.beta_start = 0.1
            self.beta_end = 20

    # A unit test to check the implementation of e2intf and var_t
    diff = make_diffusion(Foo())

    print(diff.inv_var(diff.var(torch.tensor(0.5))))
    exit()

    delta = 1e-8
    t = np.arange(start=0.001, stop=0.999, step=delta)
    t = torch.tensor(t)

    f_t = diff.f(t)
    e2intf = diff.e2int_f(t)
    # compute finite differences for e2intf
    grad_fd = (e2intf[1:] - e2intf[:-1]) / delta
    grad_analytic = f_t[:-1] * e2intf[:-1]
    print(torch.max(torch.abs(grad_fd - grad_analytic)))

    var_t = diff.var(t)
    grad_fd = (var_t[1:] - var_t[:-1]) / delta
    grad_analytic = (2 * f_t * var_t + diff.g2(t))[:-1]
    print(torch.max(torch.abs(grad_fd - grad_analytic)))
