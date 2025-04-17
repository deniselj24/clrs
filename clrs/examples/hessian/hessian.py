# *
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICUnp.linalgR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
# *

import os
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import time
from datetime import datetime
import json
import matplotlib.pyplot as plt
import copy
import jax 
import jax.numpy as jnp 

# from pyhessian.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal

from scipy.stats import norm
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from scipy.stats import entropy
from scipy.stats import pearsonr

# from src.bound.compute_bounds import *

from clrs._src.baselines import BaselineModel


class Hessian_Calculator:
    def __init__(
        self,
        model,
        loss_fn,
        rng_key,
        feedback,
        algo_idx,
        batch_size,
        train_num,
        #p=97,
        #dataloader=None,
        #valid_dataloader=None,
        #external_load_batch_func=None,
        device="cpu",
    ):
        #self.p = p # unused 
        #self.num_classes = p + 2 # unused 
        
        self.model = model
        #self.model = model.eval()  # ensure model is in evaluation mode
        #self.loss_fn = loss_fn
        self.loss_fn = BaselineModel._loss
        self.aggregate_method = "mean"

        #if external_load_batch_func is not None:
        #    self.load_batch_func = external_load_batch_func
        #else:
        #    self.load_batch_func = load_batch_func

        #self.dataloader = dataloader
        #self.valid_dataloader = valid_dataloader
        self.device = device

        # JAX model 
        self.total_params = sum(x.size for x in jax.tree_util.tree_leaves(model.params))
        #self.total_params = sum(
        #    p.numel() for p in self.model.parameters() if p.requires_grad
        #)
        self.rng_key = rng_key
        self.feedback = feedback
        self.algorithm_index = algo_idx
        self.batch_size = batch_size
        self.train_num = train_num
        self.params = model.params

    ##############################################
    # Main algorithm 1: stochastic lanczos quadrature
    ##############################################

    def _get_train_spectrum(self, n_v, n_iter):
        if (
            not (hasattr(self, "train_values") and hasattr(self, "train_weights"))
            or self.train_values is None
            or self.train_weights is None
        ):
            self.train_values, self.train_weights = self.get_full_spectrum(
                n_v, n_iter #, self.dataloader
            )
        return self.train_values, self.train_weights

    def _get_valid_spectrum(self, n_v, n_iter):
        if (
            not (hasattr(self, "valid_values") and hasattr(self, "valid_weights"))
            or self.valid_values is None
            or self.valid_weights is None
        ):
            self.valid_values, self.valid_weights = self.get_full_spectrum(
                n_v, n_iter #, self.valid_dataloader
            )
        return self.valid_values, self.valid_weights

    def get_full_spectrum(self, n_v, n_iter):
        weights = np.zeros((n_v, n_iter))
        values = np.zeros((n_v, n_iter))

        for k in range(n_v):
            "wiki version"
            T = self.tridiagonalize_by_lanzcos(n_iter, k)
            eigenvalues, U = np.linalg.eigh(T)
            values[k, :] = eigenvalues
            weights[k, :] = U[0] ** 2

        all_values = np.concatenate(values)
        all_weights = np.concatenate(weights)
        return all_values, all_weights

        grid, curve = self.interpolate(weights, values)

    def tridiagonalize_by_lanzcos(self, n_iter, k):
        "set up"
        v_list = []
        T = np.zeros((n_iter, n_iter), dtype=np.float64)

        "initialization"
        v = torch.randn(self.total_params, dtype=torch.float64)
        v /= torch.norm(v)
        v_list.append(v.cpu())

        w_prime = self.hessian_vector_product_with_tensor_input(v_list[-1])
        w_prime = torch.from_numpy(np.array(w_prime))
        "orthogonalize wprime"
        alpha = torch.sum(w_prime * v_list[-1])
        w = w_prime - alpha * v_list[-1]
        T[0, 0] = alpha

        "iteration"
        # t_s = time.time()
        # print('runing lanczos')
        for j in range(1, n_iter):
            beta = torch.norm(w)
            if beta > 1e-8:
                v_list.append(w / beta)

            else:
                v_list.append(w / 1e-8)

                # print(f' since beta = {beta}, generate v that orthogonal to all previous v')
                # # Generate a random vector orthogonal to previous ones
                # v = torch.randn(self.total_params) *(1/self.total_params)**0.5
                # for i in range(j):
                #     vi = v_list[i]
                #     v -= torch.sum(vi * v) * vi
                # v /= torch.norm(v)
                if len(v_list) > 2:
                    del v_list[0]  # keep this list short to save memory

            w_prime = self.hessian_vector_product_with_tensor_input(
                v_list[-1]
            )
            w_prime = torch.from_numpy(np.array(w_prime))
            print("w_prime", w_prime)
            alpha = torch.sum(w_prime * v_list[-1])
            w = w_prime - alpha * v_list[-1] - beta * v_list[-2]
            T[j, j] = alpha
            T[j - 1, j] = beta
            T[j, j - 1] = beta

        return T

    # First pass 16/04/25 
    def hessian_vector_product_with_tensor_input(self, d_tensor):
        "compute hessian_vector product, takes a flattened tensors as input (with shape (total parameters, ) )"
        # if dataloader is None:
        #    dataloader = self.dataloader
        #d_tensor = d_tensor.cuda()
        d_tensor = jnp.array(d_tensor.cpu().detach().numpy())
        #self.model.eval()
        #self.model.zero_grad(set_to_none=True)

        t_hd = time.time()
        #(loss, grads) = jax.value_and_grad(self.loss_fn)(self.params, self.rng_key, self.feedback, self.length_and_algorithm_idx)
        #g_list = jax.tree_util.tree_leaves(grads)
        #g_tensor, _ = jax.flatten_util.ravel_pytree(grads)
        #g_tensor = torch.from_numpy(np.array(g_tensor))  # JAX -> NumPy -> PyTorch

        #def compute_second_order_loss(params, rng_key, feedback, algo_idx):
        #    (loss, grads) = jax.value_and_grad(self.loss_fn)(self, params, self.rng_key, self.feedback, self.algorithm_index)
        #    g_list = jax.tree_util.tree_leaves(grads)
        #    g_tensor = jnp.concatenate([jnp.ravel(g) for g in g_list])
        #    return jnp.sum(g_tensor * d_tensor)

        # Get Hessian-vector product
        #print("feedback", self.feedback)
        #print("algo index", self.algorithm_index)
        hd = jax.grad(self.model._compute_second_order_loss)(self.params, self.rng_key, self.feedback, self.algorithm_index, d_tensor)
        # Flatten Hessian-vector product
        hd_list = jax.tree_util.tree_leaves(hd)
        hd_tensor = jnp.concatenate([jnp.ravel(hd) for hd in hd_list])
        return hd_tensor 
        """

        for batch in dataloader:
            # Specific data process, in order to fit the loss input
            data, target, batch_size = self.load_batch_func(batch, self.device)
            output = self.model(data)
            # loss = self.loss_fn(output, target, "mean")
            loss = self.loss_fn(output, target)
            # self.model.zero_grad()

            loss.backward(create_graph=True)
            g_list = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    g_list.append(torch.flatten(param.grad.double()))

            g_tensor = torch.cat(g_list, dim=0)

            self.model.zero_grad(set_to_none=True)
            g_tensor = g_tensor.cuda()
            l = torch.sum(g_tensor * d_tensor)
            l.backward(retain_graph=True)

            hd_list = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    hd_list.append(torch.flatten(param.grad.double().data.clone()))

            hd_tensor = torch.cat(hd_list, dim=0)
            self.model.zero_grad(set_to_none=True)
            hd_tensor = hd_tensor.cpu()
            total_hd_tensor += hd_tensor * batch_size

        total_hd_tensor /= len(dataloader.dataset)
        # if self.use_minibatch == True and batch_idx == self.gradient_accumulation_steps-1:
        #    break
        return total_hd_tensor
        """

    ##############################################
    # Main algorithm 2: Hutchinson's Method
    ##############################################

    def approximate_lambda_max(self, loss, model, power_iter=20):
        """
        Approximates the largest eigenvalue of the Hessian with respect to 'weights'
        using power iteration.
        Only works well if the Hessian is PSD.

        Returns: float lambda_max
        """
        # Compute first-order gradient with create_graph=True for higher-order derivatives.
        gradients = torch.autograd.grad(
            loss, model.parameters(), retain_graph=True, create_graph=True
        )
        grad_vector = torch.cat([g.reshape(-1) for g in gradients])

        # Initialize a random vector v of same shape as grad_vector
        v = torch.randn_like(grad_vector)
        v = v / torch.norm(v)

        for _ in range(power_iter):
            # Compute dot product between grad_vector and v (a scalar)
            dot = torch.dot(grad_vector, v)
            # Compute Hessian-vector product; allow_unused=True avoids error if some weights don't affect dot.
            Hv_tuple = torch.autograd.grad(
                dot, model.parameters(), retain_graph=True, allow_unused=True
            )
            # Replace any None gradients with zero tensors of the same shape
            Hv_list = []
            for idx, h in enumerate(Hv_tuple):
                Hv_list.append(h)
            # Flatten the gradients to a single vector
            Hv = torch.cat([h.reshape(-1) for h in Hv_list])
            norm_Hv = torch.norm(Hv, p=2)
            if norm_Hv < 1e-8:
                return 0.0
            v = Hv / norm_Hv

        # Final Rayleigh quotient approximation for the eigenvalue
        # final_dot = torch.dot(grad_vector, v)
        # lambda_max_approx = final_dot.item()
        lambda_max_approx = norm_Hv.item()
        return lambda_max_approx

    def hessian_quadratic_form(self, model, loss, noise_vector):

        # Compute gradients with respect to model parameters.
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_vector = torch.cat([g.reshape(-1) for g in grads])

        # Compute the dot product between gradients and noise vector.
        grad_dot_noise = torch.dot(grad_vector, noise_vector)

        # Compute Hessian-vector product using the Pearlmutter trick.
        Hv = torch.autograd.grad(grad_dot_noise, model.parameters(), retain_graph=True)
        Hv_vector = torch.cat([h.reshape(-1) for h in Hv])

        # The quadratic form δ^T H δ.
        quad_form = torch.dot(noise_vector, Hv_vector)
        return quad_form

    def hessian_quadratic_2_form(self, model, loss, noise_vector):

        # Compute gradients with respect to model parameters.
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_vector = torch.cat([g.reshape(-1) for g in grads])

        noise_tuple = []
        idx = 0
        for g in grads:
            sz = g.numel()
            noise_tuple.append(noise_vector[idx : idx + sz].view_as(g))
            idx += sz

        # Compute the dot product between gradients and noise vector.
        # grad_dot_noise = torch.dot(grad_vector, noise_vector)

        # Compute Hessian-vector product using the Pearlmutter trick.
        # Hv = torch.autograd.grad(grad_dot_noise, model.parameters(), retain_graph=True)
        # HHv = torch.autograd.grad(Hv, model.parameters(), retain_graph=True)
        Hv = torch.autograd.grad(
            grads, model.parameters(), grad_outputs=noise_tuple, retain_graph=True
        )
        HHv = torch.autograd.grad(
            grads, model.parameters(), grad_outputs=Hv, retain_graph=True
        )
        HHv_vector = torch.cat([h.reshape(-1) for h in HHv])

        # The quadratic form δ^T H δ.
        quad_form = torch.dot(noise_vector, HHv_vector)
        return quad_form

    def compare_hessian(self, logger, log_i, train_num, valid_num):
        with sdpa_kernel(SDPBackend.MATH):
            device = self.device
            model = self.model
            loss_fn = self.loss_fn
            sample_num = 50
            train_hessian_list = []
            train_hessian_2_list = []
            valid_hessian_list = []
            valid_hessian_2_list = []
            for i in range(sample_num):
                noise_vector = None
                train_hessian = 0
                train_hessian_2 = 0
                for train_batch in self.dataloader:
                    data, target, batch_size = self.load_batch_func(train_batch, device)
                    output = model(data)
                    loss = loss_fn(output, target)

                    # Compute gradients to get the shape.
                    if noise_vector is None:
                        grads = torch.autograd.grad(
                            loss, model.parameters(), create_graph=True
                        )
                        grad_vector = torch.cat([g.reshape(-1) for g in grads])

                        # Sample the noise vector once.
                        # noise_vector = torch.randn_like(grad_vector)
                        noise_vector = torch.randint_like(grad_vector, high=2)
                        noise_vector[noise_vector == 0] = -1
                    train_quad = self.hessian_quadratic_form(model, loss, noise_vector)
                    train_H2 = self.hessian_quadratic_2_form(model, loss, noise_vector)
                    train_hessian += train_quad.item() * batch_size
                    train_hessian_2 += train_H2.item() * batch_size
                train_hessian /= train_num
                train_hessian_2 /= train_num

                valid_hessian = 0
                valid_hessian_2 = 0
                for valid_batch in self.valid_dataloader:
                    data, target, batch_size = self.load_batch_func(valid_batch, device)
                    output = model(data)
                    loss = loss_fn(output, target)

                    valid_quad = self.hessian_quadratic_form(model, loss, noise_vector)
                    valid_hessian += valid_quad.item() * batch_size
                    valid_H2 = self.hessian_quadratic_2_form(model, loss, noise_vector)
                    valid_hessian_2 += valid_H2.item() * batch_size
                valid_hessian /= valid_num
                valid_hessian_2 /= valid_num

                noise_vector = None
                train_hessian_list.append(train_hessian)
                train_hessian_2_list.append(train_hessian_2)
                valid_hessian_list.append(valid_hessian)
                valid_hessian_2_list.append(valid_hessian_2)
            train_hessian = np.mean(train_hessian_list)
            train_hessian_2 = np.mean(train_hessian_2_list)
            valid_hessian = np.mean(valid_hessian_list)
            valid_hessian_2 = np.mean(valid_hessian_2_list)

        self.train_hessian = train_hessian
        self.train_hessian_2 = train_hessian_2
        self.valid_hessian = valid_hessian
        self.valid_hessian_2 = valid_hessian_2

        logger.log("train_hessian", train_hessian, log_i)
        logger.log("train_hessian_2", train_hessian_2, log_i)
        logger.log("valid_hessian", valid_hessian, log_i)
        logger.log("valid_hessian_2", valid_hessian_2, log_i)

        plot_curves(
            logger,
            ["train_hessian", "train_hessian_2", "valid_hessian", "valid_hessian_2"],
            path_name="hessian",
        )
        # plot_curves(logger, ['train_hessian_2', 'valid_hessian_2'], path_name='hessian_2')

        return train_hessian, train_hessian_2, valid_hessian, valid_hessian_2

    ##############################################
    # Hessian values
    ##############################################

    def compute_sensitivity(self, loss, data, target, batch_size):
        # noise_sensitivity: Estimate the sensetivity of input. Save in ./results/input
        for i in range(50):
            # noisy_output, noise_norm = model.add_noise_forward(data)

            # noise_sensitivity = torch.norm(noisy_output[:, -1] - output[:, -1]) / noise_norm
            # noisy_loss = F.cross_entropy(noisy_output[:, -1], target[:, -1], reduction='none')
            # noise_sensitivity = (noisy_loss - loss) / noise_norm
            noisy_output_1, noisy_output_2, noise_norm = (
                self.model.add_bi_noise_forward(data)
            )
            noisy_loss_1 = F.cross_entropy(
                noisy_output_1[:, -1], target[:, -1], reduction="none"
            )
            noisy_loss_2 = F.cross_entropy(
                noisy_output_2[:, -1], target[:, -1], reduction="none"
            )
            noise_sensitivity = noisy_loss_1 + noisy_loss_2 - 2 * loss
            # noise_sensitivity = (noisy_output_1 + noisy_output_2 - output)[:, -1]

        self.noise_sensitivity += noise_sensitivity.mean().item() * batch_size

    def compute_spectral_entropy(self, eigen_list, weight_list, sigma=0.01, grid=1000):
        # Step 1: Filter near-zero eigenvalues
        filtered_eigen, filtered_weight = filter_eigenvalues(eigen_list, weight_list)

        # Step 2: Renormalize weights
        renormalized_weight = renormalize_weights(filtered_weight)
        # print("renorm: ", sum(renormalized_weight))

        # Step 3: Define lambda grid
        if len(filtered_eigen) == 0:
            raise ValueError(
                "No eigenvalues remain after filtering. Adjust the threshold."
            )
        lambda_min = min(filtered_eigen) - 1.0  # Adding padding
        lambda_max = max(filtered_eigen) + 1.0  # Adding padding
        lambdas = np.linspace(lambda_min, lambda_max, grid)
        delta_lambda = lambdas[1] - lambdas[0]

        # Step 4: Construct spectral density
        density = construct_spectral_density(
            filtered_eigen, renormalized_weight, lambdas, sigma
        )

        # Step 5: Compute spectral entropy
        epsilon = 1e-12
        p = density + epsilon  # Avoid log(0)
        # spectral_entropy = -np.sum(p * np.log(p))
        p = np.array(renormalized_weight) + epsilon
        spectral_entropy = -np.sum(p * np.log(p))
        weighted_entropy = -np.sum(p * np.log(p) * np.array(filtered_eigen))
        centroid = np.sum(np.array(renormalized_weight) * np.array(filtered_eigen))
        spread = np.sum(
            np.array(renormalized_weight) * (np.array(filtered_eigen) - centroid) ** 2
        )

        return spectral_entropy, weighted_entropy, centroid, spread

    def compute_effective_rank(
        self, value_tensor: torch.Tensor, weight_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the effective rank (R_eff) using Option A: weighting by eigenvalue magnitude.

        Args:
            value_tensor (torch.Tensor): 1D tensor of eigenvalue bin centers.
            weight_tensor (torch.Tensor): 1D tensor of density values corresponding to the eigenvalues.

        Returns:
            torch.Tensor: Effective rank (a scalar tensor).
        """
        # Compute the bin width. We assume equally spaced bins.
        if value_tensor.numel() > 1:
            d_lambda = (value_tensor[1] - value_tensor[0]).item()
        else:
            d_lambda = 1.0  # Fallback value if only one bin is available

        # Compute the trace approximation T = sum(λ * density * bin_width)
        T = torch.sum(value_tensor * weight_tensor * d_lambda)

        # Construct the probability distribution:
        # p(λ) = (λ * density * bin_width) / T
        p = (value_tensor * weight_tensor * d_lambda) / T

        # For numerical stability, clamp p to avoid log(0)
        epsilon = 1e-12
        p_clamped = p.clamp(min=epsilon)

        # Compute the entropy H = -sum(p * log(p))
        entropy = -torch.sum(p * torch.log(p_clamped))

        # The effective rank is the exponential of the entropy.
        effective_rank = torch.exp(entropy)

        return effective_rank

    def pac_bound(
        self,
        logger,
        log_i,
        train_num,
        valid_num,
        n_iter=10,
        n_v=5,
        delta=0.1,
        use_hessian_bound=True,
    ):
        """
        print("=======> lambda")
        with sdpa_kernel(SDPBackend.MATH):
            device = self.device
            model = self.model
            loss_fn = self.loss_fn

            train_lambda_max = 0
            for batch in self.dataloader:
                data, target, batch_size = self.load_batch_func(batch, device)
                output = model(data)
                loss = loss_fn(output, target, 'none')
                #model.zero_grad()

                batch_lambda_max = self.approximate_lambda_max(loss.mean(), model, power_iter=100)
                train_lambda_max += batch_lambda_max * batch_size
            train_lambda_max /= len(self.dataloader.dataset)

            valid_lambda_max = 0
            for batch in self.valid_dataloader:
                data, target, batch_size = self.load_batch_func(batch, device)
                output = model(data)
                loss = loss_fn(output, target, 'none')
                #model.zero_grad()

                batch_lambda_max = self.approximate_lambda_max(loss.mean(), model, power_iter=100)
                valid_lambda_max += batch_lambda_max * batch_size
            valid_lambda_max /= len(self.valid_dataloader.dataset)
        """
        print("=======> hessian")
        train_hessian, train_hessian_2, valid_hessian, valid_hessian_2 = (
            self.compare_hessian(logger, log_i, train_num, valid_num)
        )
        d = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        delta = torch.tensor(delta)

        print(
            "part 1",
            2
            * torch.sqrt(train_hessian_2 * torch.log(2 * d / delta))
            / torch.sqrt(torch.tensor(train_num)),
        )
        print("part 2", 4 * train_hessian * torch.log(2 / delta) / (3 * train_num))
        # hessian_part = 2 * torch.sqrt(train_hessian_2*torch.log(2*d/delta)) / torch.sqrt(torch.tensor(train_num)) + 4 * train_hessian * torch.log(2/delta) / (3 * train_num)
        hessian_part = torch.sqrt(
            2 * train_hessian_2 * torch.log(d / delta) / train_num
        ) + (train_hessian + torch.sqrt(torch.tensor(train_hessian_2))) * torch.log(
            2 / delta
        ) / (
            3 * train_num
        )

        hessian_item = abs(train_hessian - valid_hessian)

        kl_div = compute_kl_divergence_initial_state(
            self.model.state_dict(), self.model.init_state
        )
        pac_part = pac_bayes_term(kl_div=kl_div, n=train_num, delta=delta)

        pac_part = pac_part.item()
        hessian_part = hessian_part.item()
        result = pac_part + hessian_part
        real_result = pac_part + hessian_item.item()
        logger.log("pac_part", pac_part, log_i)
        logger.log("hessian_part", hessian_part, log_i)
        logger.log("pac_bound_result", result, log_i)
        logger.log("pac_real_result", real_result, log_i)
        plot_curves(
            log=logger,
            data_names=["pac_part", "hessian_part"],
            path_name="pac",
            file_name="pac",
        )

        logger.log("hessian_gap", abs(train_hessian - valid_hessian), log_i)
        plot_curves(
            log=logger,
            data_names=["hessian_gap", "hessian_part"],
            path_name="hessian_noise",
            file_name="hessian_gap",
        )

        plot_curves(
            log=logger,
            data_names=["pac_bound_result", "pac_real_result", "loss_gap"],
            path_name="gap",
            file_name="pac_bound",
        )

        return result, pac_part, hessian_part, real_result

    ##############################################
    # Usage
    ##############################################

    def check_slq(self, logger, i, train_num, valid_num, n_iter=100, n_v=1):
        with sdpa_kernel(SDPBackend.MATH):
            print("=======> SLQ for full model")
            # values_full, weights_full = self.get_full_spectrum(n_iter=n_iter, n_v=n_v, dataloader=self.dataloader)
            values_full, weights_full = self._get_train_spectrum(n_v, n_iter)
            self.values_full = values_full.tolist()
            self.weights_full = weights_full.tolist()
            d = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.slq_H_trace = np.sum(values_full * weights_full) * d
            self.slq_H2_trace = np.sum(values_full**2 * weights_full) * d
            self.hvp_H_trace, self.hvp_H2_trace, _, _ = self.compare_hessian(
                logger, i, train_num, valid_num
            )
            print(self.slq_H_trace, self.slq_H2_trace, self.hvp_H_trace)
            slq_lambda_max = max(values_full)

            device = self.device
            model = self.model
            loss_fn = self.loss_fn

            train_lambda_max = 0
            for batch in self.dataloader:
                data, target, batch_size = self.load_batch_func(batch, device)
                output = model(data)
                loss = loss_fn(output, target, "none")
                model.zero_grad()

                batch_lambda_max = self.approximate_lambda_max(
                    loss.mean(), model, power_iter=100
                )
                train_lambda_max += batch_lambda_max * batch_size
            train_lambda_max /= len(self.dataloader.dataset)

        logger.log("slq_H_trace", self.slq_H_trace, i)
        logger.log("slq_H2_trace", self.slq_H2_trace, i)
        logger.log("hvp_H_trace", self.hvp_H_trace, i)
        logger.log("hvp_H2_trace", self.hvp_H2_trace, i)
        data_names = ["slq_H_trace", "hvp_H_trace"]
        plot_curves(logger, data_names, path_name="check", file_name="hessian")
        data_names = ["slq_H2_trace", "hvp_H2_trace"]
        plot_curves(logger, data_names, path_name="check", file_name="hessian_2")

        logger.log("hvp_lambda_max", train_lambda_max, i)
        logger.log("slq_lambda_max", slq_lambda_max, i)
        plot_curves(
            logger,
            ["hvp_lambda_max", "slq_lambda_max"],
            path_name="check",
            file_name="lambda_max",
        )

    def noisy_loss(self, logger, log_i, train_loss, val_loss, train_num, valid_num):
        print("=======> hessian")
        train_hessian, train_hessian_2, valid_hessian, valid_hessian_2 = (
            self.compare_hessian(logger, log_i, train_num, valid_num)
        )
        print("=======> perturb model")
        noisy_model = copy.deepcopy(self.model)
        loss_fn = self.loss_fn

        noisy_train_loss_list = []
        noisy_valid_loss_list = []
        sigma = 0.01
        for i in range(50):
            for param in noisy_model.parameters():
                noise = torch.randn_like(param) * sigma
                param.data.add_(noise)

            noisy_train_loss = 0
            for batch in self.dataloader:
                data, target, batch_size = self.load_batch_func(batch, self.device)
                output = noisy_model(data)
                loss = self.loss_fn(output, target, "mean")
                noisy_train_loss += loss.item() * batch_size
            noisy_train_loss /= train_num
            noisy_train_loss_list.append(noisy_train_loss)

            noisy_valid_loss = 0
            for batch in self.valid_dataloader:
                data, target, batch_size = self.load_batch_func(batch, self.device)
                output = noisy_model(data)
                loss = self.loss_fn(output, target, "mean")
                noisy_valid_loss += loss.item() * batch_size
            noisy_valid_loss /= valid_num
            noisy_valid_loss_list.append(noisy_valid_loss)

        noisy_train_loss = np.mean(noisy_train_loss_list)
        noisy_valid_loss = np.mean(noisy_valid_loss_list)
        perturbation_loss_gap = noisy_valid_loss - noisy_train_loss
        logger.log("perturbation_loss_gap", perturbation_loss_gap, log_i)
        logger.log("noisy_train_loss", noisy_train_loss, log_i)
        logger.log("noisy_valid_loss", noisy_valid_loss, log_i)

        loss_gap = abs(train_loss - val_loss)
        hessian_gap = train_hessian - valid_hessian
        # perturbation_taylor_gap = loss_gap + (valid_hessian - train_hessian) * sigma**2 / 2
        perturbation_taylor_gap = (
            loss_gap + (train_hessian - valid_hessian) * sigma**2 / 2
        )
        train_taylor_loss = train_loss + train_hessian * sigma**2 / 2
        valid_taylor_loss = val_loss + valid_hessian * sigma**2 / 2
        logger.log("perturbation_taylor_gap", perturbation_taylor_gap, log_i)
        logger.log("train_taylor_loss", train_taylor_loss, log_i)
        logger.log("valid_taylor_loss", valid_taylor_loss, log_i)
        logger.log("hessian_gap", hessian_gap, log_i)
        plot_curves(
            log=logger,
            data_names=["noisy_train_loss", "train_taylor_loss"],
            path_name="perturbation",
            file_name="train",
        )
        plot_curves(
            log=logger,
            data_names=["noisy_valid_loss", "valid_taylor_loss"],
            path_name="perturbation",
            file_name="valid",
        )
        plot_curves(
            log=logger,
            data_names=["perturbation_loss_gap", "perturbation_taylor_gap"],
            path_name="perturbation",
            file_name="gap",
        )
        plot_curves(
            log=logger,
            data_names=["hessian_gap"],
            path_name="perturbation",
            file_name="hessian_gap",
            y_log=False,
        )

    def compute_compression_bound(
        self, logger, log_i, train_num, valid_num, train_loss
    ):
        # vector = self.model.weight.cpu().data.numpy()
        vector = (
            torch.cat([p.detach().view(-1) for p in self.model.parameters()])
            .cpu()
            .numpy()
        )
        quantized_vec, message_len = quantize_vector(vector)
        prefix_message_len = (
            message_len + 2 * np.log2(message_len) if message_len > 0 else 0
        )
        misc_extra_bits = 5  # TODO
        divergence = (prefix_message_len + misc_extra_bits) * np.log(2)
        total_sample_size = train_num + valid_num
        sample_size = train_num
        alpha = 0.2
        all_selected_probs = []
        for batch in self.valid_dataloader:
            data, target, batch_size = self.load_batch_func(batch, self.device)
            output = self.model(data)
            # loss = self.loss_fn(output, target, 'mean')
            logits = output[:, -1, :]
            softmax_matrix = torch.nn.functional.softmax(logits, dim=-1)
            selected_prob_scores = softmax_matrix[
                torch.arange(softmax_matrix.shape[0]), target[:, -1].view(-1)
            ]
            all_selected_probs.append(selected_prob_scores)
        all_selected_probs = torch.cat(all_selected_probs)

        vocab_size = self.p
        log_probs = [
            np.log2((1 - alpha) * x.item() + alpha / vocab_size)
            for x in selected_prob_scores
        ]
        bdp_alpha = -sum(log_probs) / len(log_probs)
        alpha = (bdp_alpha) / (valid_num)  # TODO

        delta = np.log2(1 + (1 - alpha) * vocab_size / alpha)
        compression_bound = llm_subsampling_bound(
            train_error=train_loss,
            div=divergence,
            data_size=total_sample_size,
            sample_size=sample_size,
            delta=delta,
        )
        compression_bound = compression_bound - train_loss
        logger.log("compression_bound", compression_bound, log_i)
        plot_curves(
            log=logger,
            data_names=["loss_gap", "compression_bound"],
            path_name="compression",
            file_name="compression_bound",
        )
        return compression_bound

    def compare_bound(
        self, logger, log_i, train_num, valid_num, train_loss, n_iter=10, n_v=5
    ):
        # compte compression bound baseline
        # compression_bound = self.compute_compression_bound(
        #     logger, log_i, train_num, valid_num, train_loss
        # )

        # approximate Hessian spectrum using SLQ, on training data and validation data
        print("=======> SLQ")
        with sdpa_kernel(SDPBackend.MATH):
            # train_values_full, train_weights_full = self.get_full_spectrum(n_v=n_v, n_iter=n_iter, dataloader=self.dataloader)
            # valid_values_full, valid_weights_full = self.get_full_spectrum(n_v=n_v, n_iter=n_iter, dataloader=self.valid_dataloader)
            train_values_full, train_weights_full = self._get_train_spectrum(
                n_v, n_iter
            )
            valid_values_full, valid_weights_full = self._get_valid_spectrum(
                n_v, n_iter
            )

        # model norm, use in the Hessian bound
        r = compute_model_norm(self.model)

        train_values_tensor = torch.tensor(train_values_full, dtype=torch.float32)
        train_weights_tensor = torch.tensor(train_weights_full, dtype=torch.float32)
        train_pos_mask = train_values_tensor > 0
        train_neg_mask = train_values_tensor < 0

        valid_values_tensor = torch.tensor(valid_values_full, dtype=torch.float32)
        valid_weights_tensor = torch.tensor(valid_weights_full, dtype=torch.float32)
        valid_pos_mask = valid_values_tensor > 0
        valid_neg_mask = valid_values_tensor < 0

        # Some spectrum metrics that might be useful: effective rank , entropy, weighted entropy
        train_effective_rank = self.compute_effective_rank(
            train_values_tensor[train_pos_mask], train_weights_tensor[train_pos_mask]
        )
        train_entropy, train_weighted_entropy, train_centroid, train_spread = (
            self.compute_spectral_entropy(
                train_values_full[train_pos_mask],
                train_weights_full[train_pos_mask],
                sigma=0.01,
                grid=1000,
            )
        )
        valid_effective_rank = self.compute_effective_rank(
            valid_values_tensor[valid_pos_mask], valid_weights_tensor[valid_pos_mask]
        )
        valid_entropy, valid_weighted_entropy, valid_centroid, valid_spread = (
            self.compute_spectral_entropy(
                valid_values_full[valid_pos_mask],
                valid_weights_full[valid_pos_mask],
                sigma=0.01,
                grid=1000,
            )
        )

        # Compute the trace bound and spectral bound
        d = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        C = np.sqrt(5)

        # trace and bound, computed on validation data
        valid_hessian_trace = torch.sum(valid_values_tensor * valid_weights_tensor) * d
        trace_bound = torch.sqrt(torch.tensor(valid_hessian_trace / valid_num)) * r
        spectral_bound = (
            torch.sqrt(torch.tensor(valid_weighted_entropy / valid_num)) * r
        )

        # log results to logger
        logger.log("train_effective_rank", train_effective_rank.item(), log_i)
        logger.log("train_entropy", train_entropy.item(), log_i)
        logger.log("train_weighted_entropy", train_weighted_entropy.item(), log_i)
        logger.log("valid_effective_rank", valid_effective_rank.item(), log_i)
        logger.log("valid_entropy", valid_entropy.item(), log_i)
        logger.log("valid_weighted_entropy", valid_weighted_entropy.item(), log_i)
        logger.log("trace_bound", trace_bound.item(), log_i)
        logger.log("spectral_bound", spectral_bound.item(), log_i)

        # plot results in logger
        plot_curves(
            log=logger,
            data_names=["train_effective_rank", "valid_effective_rank"],
            path_name="entropy",
            file_name="effective_rank",
        )
        plot_curves(
            log=logger,
            data_names=["train_entropy", "valid_entropy"],
            path_name="entropy",
            file_name="entropy",
        )
        plot_curves(
            log=logger,
            data_names=["train_weighted_entropy", "valid_weighted_entropy"],
            path_name="entropy",
            file_name="weighted_entropy",
        )
        plot_curves(
            log=logger,
            data_names=[
                "loss_gap",
                "trace_bound",
                "spectral_bound",
                # "compression_bound",
            ],
            path_name="bound",
            file_name="bound",
        )


def compute_model_norm(model, p=2):
    norm = torch.norm(
        torch.stack(
            [torch.norm(p.detach(), 2) for p in model.parameters() if p.requires_grad]
        ),
        2,
    )
    return norm


def load_batch_func(batch, device="cpu"):
    batch = batch[0].to(device)
    inputs = batch[:, :-1]
    targets = batch
    batch_size = batch.shape[0]
    return inputs, targets, batch_size


def filter_eigenvalues(eigen_list, weight_list, threshold=None):
    filtered_eigen = []
    filtered_weight = []
    # print(np.max(weight_list))
    for eig, w in zip(eigen_list, weight_list):
        if threshold is not None:
            if eig >= threshold and w >= 1e-7:
                filtered_eigen.append(eig)
                filtered_weight.append(w)
        else:
            if w >= 1e-10:
                filtered_eigen.append(eig)
                filtered_weight.append(w)
    # print(filtered_eigen)
    return filtered_eigen, filtered_weight


def renormalize_weights(filtered_weight, epsilon=1e-12):
    total = sum(filtered_weight)
    if total > 0:
        renormalized_weight = [w / (total + epsilon) for w in filtered_weight]
    else:
        # Handle case where all weights are zero
        renormalized_weight = [0.0 for _ in filtered_weight]
    return renormalized_weight


def construct_spectral_density(flat_eigen, flat_weight, lambdas, sigma=0.1):
    density = np.zeros_like(lambdas)
    for eig, w in zip(flat_eigen, flat_weight):
        density += w * norm.pdf(lambdas, loc=eig, scale=sigma)

    # Normalize the density
    density_sum = np.sum(density) * (lambdas[1] - lambdas[0])
    density /= density_sum + 1e-12  # Avoid division by zero
    return density


def sqrt_with_neg_handling(arr):
    result = np.where(arr < 0, 0, np.sqrt(arr))
    return result


def compute_eigenvalue(model, loss, device, maxIter=100, tol=1e-10, top_n=1):
    model.zero_grad()
    gradients = torch.autograd.grad(
        loss, model.parameters(), retain_graph=True, create_graph=True
    )

    topn_eigenvalues = []
    eigenvectors = []
    computed_dim = 0
    while computed_dim < top_n:

        eigenvalues = None
        # Compute gradients with respect to model parameters.
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_vector = torch.cat([g.reshape(-1) for g in grads])

        v = torch.randn_like(grad_vector)
        v = v / torch.norm(v)

        # Compute the dot product between gradients and noise vector.
        grad_dot_noise = torch.dot(grad_vector, v)

        # Compute Hessian-vector product using the Pearlmutter trick.
        Hv = torch.autograd.grad(grad_dot_noise, model.parameters(), retain_graph=True)

        for _ in range(maxIter):
            vs = orthnormal(vs, eigenvectors)
            model.zero_grad()

            Hvs = torch.autograd.grad(
                grad_dot_noise, model.parameters(), retain_graph=True
            )
            tmp_eigenvalues = torch.sum(Hv * v).cpu().item()

            vs = normalization(Hvs)

            if eigenvalues == None:
                eigenvalues = tmp_eigenvalues
            else:
                if (
                    abs(sum(eigenvalues) - sum(tmp_eigenvalues))
                    / (abs(sum(eigenvalues)) + 1e-6)
                    < tol
                ):
                    break
                else:
                    eigenvalues = tmp_eigenvalues
        topn_eigenvalues.append(eigenvalues)
        eigenvectors.append(vs)
        computed_dim += 1

    return topn_eigenvalues, eigenvectors


def compute_layer_eigenvalue(model, loss, device, maxIter=100, tol=1e-10, top_n=1):
    model.zero_grad()
    layers = model.get_layers()
    weights = [module.weight for name, module in layers.items()]

    model.zero_grad()
    gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)

    topn_eigenvalues = []
    eigenvectors = []
    computed_dim = 0
    while computed_dim < top_n:

        eigenvalues = None
        vs = [torch.randn_like(weight) for weight in weights]  # generate random vector
        vs = normalization(vs)  # normalize the vector

        for _ in range(maxIter):
            vs = orthnormal(vs, eigenvectors)
            model.zero_grad()

            Hvs = torch.autograd.grad(
                gradients, weights, grad_outputs=vs, retain_graph=True
            )
            tmp_eigenvalues = [
                torch.sum(Hv * v).cpu().item() for (Hv, v) in zip(Hvs, vs)
            ]

            vs = normalization(Hvs)

            if eigenvalues == None:
                eigenvalues = tmp_eigenvalues
            else:
                if (
                    abs(sum(eigenvalues) - sum(tmp_eigenvalues))
                    / (abs(sum(eigenvalues)) + 1e-6)
                    < tol
                ):
                    break
                else:
                    eigenvalues = tmp_eigenvalues
        topn_eigenvalues.append(eigenvalues)
        eigenvectors.append(vs)
        computed_dim += 1

    return topn_eigenvalues, eigenvectors


def get_layers(model):
    layers = OrderedDict()
    for name, module in model.named_modules():
        if hasattr(module, "weight") and any(
            p.requires_grad for p in module.parameters(recurse=False)
        ):
            # if (type(module) == torch.nn.Linear) and "LayerNorm" not in name and "ln" not in name and "embeddings" not in name and "pooler" not in name:
            if "LayerNorm" not in name and "ln" not in name and "pooler" not in name:
                # print(f"Layer: {name}, Module: {module}")
                layers[name] = module
    return layers


def weighted_quantile(values, weights, quantile):
    """
    Compute the weighted quantile of a tensor.
    Args:
      values: 1D tensor of eigenvalues.
      weights: 1D tensor of corresponding weights.
      quantile: desired quantile (between 0 and 1).
    Returns:
      The eigenvalue threshold corresponding to the weighted quantile.
    """
    # Sort values and weights in ascending order.
    sorted_vals, sorted_indices = torch.sort(values)
    sorted_weights = weights[sorted_indices]
    cumulative_weights = torch.cumsum(sorted_weights, dim=0)
    total_weight = sorted_weights.sum()
    normalized_cum_weights = cumulative_weights / total_weight
    # Find the first index where cumulative weight exceeds the quantile.
    idx = torch.nonzero(normalized_cum_weights >= quantile, as_tuple=False)[0]
    threshold = sorted_vals[idx]
    return threshold


def tail_mass_fraction(values, weights, quantile=0.9):
    """
    Compute the tail mass fraction: the fraction of the total weighted mass
    (∑ p_i λ_i) that comes from eigenvalues above the weighted quantile threshold.
    """
    weights = weights / weights.sum()
    tau = weighted_quantile(values, weights, quantile)
    mask = values >= tau
    numerator = torch.sum(weights[mask] * values[mask])
    denominator = torch.sum(weights * values)
    return (numerator / denominator).item()


def weighted_gini(values, weights):
    """
    Compute the weighted Gini coefficient for the eigenvalue distribution.
    First, normalize weights to obtain a probability distribution:
      q_i = weights_i / (∑_j weights_j).
    Then, compute:
      G = (∑_{i,j} q_i q_j |λ_i - λ_j|) / (2 μ),
    where μ = ∑_i q_i λ_i.
    """
    # Normalize the weights to form a probability distribution.
    q = weights / weights.sum()
    mu = (values * q).sum()
    # Compute pairwise absolute differences between eigenvalues.
    diff_matrix = torch.abs(values.unsqueeze(0) - values.unsqueeze(1))
    # Compute pairwise product of normalized weights.
    q_matrix = q.unsqueeze(0) * q.unsqueeze(1)
    gini = torch.sum(diff_matrix * q_matrix) / (2 * mu)
    return gini.item()


def weighted_skewness(values, weights, eps=1e-8):
    """
    Compute the weighted skewness for the eigenvalue distribution.
    Using normalized weights q_i = weights_i / (∑_j weights_j), we have:
      μ   = ∑_i q_i λ_i,
      σ²  = ∑_i q_i (λ_i - μ)²,
      skew = ∑_i q_i (λ_i - μ)³ / (σ³ + eps).
    """
    q = weights / weights.sum()
    mu = (values * q).sum()
    diff = values - mu
    variance = (q * diff**2).sum()
    std = torch.sqrt(variance + eps)
    skew = (q * diff**3).sum() / (std**3 + eps)
    return skew.item()


def compute_sigma_from_weights(state_dict, factor=1.0):
    """
    Compute sigma as a factor times the average standard deviation of the floating-point parameters.
    """
    sigmas = []
    for key, param in state_dict.items():
        if param.requires_grad:
            sigmas.append(param.std().item())
    if sigmas:
        return factor * (sum(sigmas) / len(sigmas))
    else:
        return factor


def compute_kl_divergence_initial_state(final_state_dict, init_state_dict):
    """
    Compute KL(Q||P) where
        Q = N(w_T, sigma^2 I) is the posterior (final weights),
        P = N(w_0, sigma0^2 I) is the prior (initial weights).
    """
    sigma = compute_sigma_from_weights(final_state_dict, factor=0.5)
    sigma0 = compute_sigma_from_weights(init_state_dict, factor=1.0)

    sigma2 = sigma**2
    sigma0_2 = sigma0**2
    kl_total = 0.0

    # Loop over parameters (assuming both state_dicts have the same keys)
    for key in final_state_dict:
        param_final = final_state_dict[key]
        param_init = init_state_dict[key].to(param_final.device)

        # Consider only floating point parameters (learnable weights)
        if not torch.is_floating_point(param_final) or not torch.is_floating_point(
            param_init
        ):
            continue

        d = param_final.numel()  # number of elements in this tensor
        # Compute squared difference between final and initial weights
        diff_norm_sq = torch.sum((param_final - param_init) ** 2)

        # KL divergence for this tensor:
        # KL = 0.5 * [d*log(sigma0^2/sigma^2) + ||w_T - w_0||^2/sigma0^2 + d*(sigma^2/sigma0^2) - d]
        kl_tensor = 0.5 * (
            d * math.log(sigma0_2 / sigma2)
            + diff_norm_sq / sigma0_2
            + d * (sigma2 / sigma0_2)
            - d
        )
        kl_total += kl_tensor

    return kl_total


def pac_bayes_term(kl_div, n, delta):
    """
    Computes the PAC-Bayes bound of the form:
        E[L(f)] <= E[hat{L}(f)] + sqrt((KL(Q||P) + log(1/delta_prime)) / (2n))

    Args:
        empirical_loss (float or torch.Tensor): Empirical loss (averaged over n samples).
        kl_div (float or torch.Tensor): KL(Q||P) already computed.
        n (int): Number of samples in the dataset.
        delta_prime (float): Confidence parameter (e.g., 0.05).

    Returns:
        torch.Tensor: The PAC-Bayes upper bound on the true (expected) loss.
    """
    # Ensure all inputs are torch.Tensor for consistency
    if not isinstance(kl_div, torch.Tensor):
        kl_div = torch.tensor(float(kl_div), dtype=torch.float32)

    # Convert n and delta_prime to Tensors if needed
    n_t = torch.tensor(float(n), dtype=torch.float32)
    delta_t = torch.tensor(float(delta), dtype=torch.float32)

    # Compute the PAC-Bayes complexity term
    # sqrt( (KL + log(1/delta)) / (2n) )
    complexity_term = torch.sqrt((kl_div + torch.log(1.0 / delta_t)) / (2.0 * n_t))

    # Final bound is empirical_loss + complexity
    return complexity_term


def plot_curves(
    log,
    data_names,
    path_name,
    file_name=None,
    save_dir="./results/",
    x_log=True,
    y_log=True,
):
    if file_name is None:
        file_name = path_name
    train_converge = log["train_converge"]["value"]
    val_converge = log["val_converge"]["value"]
    # grok_start = log["grok_start"]["value"]
    # print(train_converge, val_converge)
    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Plotting hessian, ", "temp")  # TODO JL 3/21/25 log.label
    for i, name in enumerate(data_names):
        plt.plot(log[name]["iter"], log[name]["value"], label=name)

    # Fix for finding convergence indices
    train_converge_index = -1
    val_converge_index = -1

    # Convert to numpy array if it's not already
    train_converge_array = np.array(train_converge)
    val_converge_array = np.array(val_converge)

    # Find the first occurrence of 1 in the arrays
    train_converge_indices = np.where(train_converge_array == 1)[0]
    val_converge_indices = np.where(val_converge_array == 1)[0]

    # Check if any indices were found
    if len(train_converge_indices) > 0:
        train_converge_index = train_converge_indices[0]

    if len(val_converge_indices) > 0:
        val_converge_index = val_converge_indices[0]

    if train_converge_index > 0:
        plt.axvline(
            x=train_converge_index,
            color="blue",
            linestyle="--",
            linewidth=1,
            label="train convergence",
        )
    if val_converge_index > 0:
        plt.axvline(
            x=val_converge_index,
            color="orange",
            linestyle="--",
            linewidth=1,
            label="val convergence",
        )
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Hessian")
    if x_log:
        plt.xscale("log", base=10)
    if y_log:
        plt.yscale("log", base=10)
    # plt.ylim(1e-7, 1e7)
    plt.grid()
    plt.annotate(
        time_str,
        xy=(0.2, 0.5),
        xycoords="axes fraction",
        fontsize=12,
        color="purple",
        ha="center",
    )
    os.makedirs(f"{save_dir}{path_name}", exist_ok=True)
    plt.savefig(
        f"{save_dir}{path_name}/{file_name}_temp.png", dpi=150
    )  # TODO JL 3/21/25 log.label
    plt.draw()
    plt.close()
