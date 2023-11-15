# Copyright 2021 Predicitve Intelligence Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.numpy as np
from jax import jit, vmap, grad, random, jacfwd
from jax.flatten_util import ravel_pytree

from jaxpinns.base import PINN, dtPINN
from jaxpinns.ntk import compute_ntk

from functools import partial

class Poisson1D(PINN):
    ''' 1D Poisson equation solver '''
    # Initialize the class
    def __init__(self, mu_X = 0.0, sigma_X = 1.0):
        super().__init__(mu_X, sigma_X)
        self.num_loss_terms = 2

    # Make sure this accepts a single input data-point and returns a scalar!
    # Then use vmap to vectorize
    def net_u(self, params, x):
        inputs = np.stack([x])
        u = self.net_apply(params, inputs)
        return u[0]

    def net_r(self, params, x):
        u_xx = grad(grad(self.net_u, 1), 1)(params, x)
        res = -u_xx/self.sigma_X**2
        return res

    @partial(jit, static_argnums=(0,))
    def loss_u(self, params, batch):
        # Fetch data
        inputs, targets = batch
        X_bc1, X_bc2, _ = inputs
        Y_bc1, Y_bc2, _ = targets
        # Evaluate model
        u_fn = lambda x: self.net_u(params, x)
        u_bc1 = vmap(u_fn)(X_bc1)
        u_bc2 = vmap(u_fn)(X_bc2)
        # Compute loss
        loss_bc1 = np.mean((Y_bc1 - u_bc1)**2)
        loss_bc2 = np.mean((Y_bc2 - u_bc2)**2)
        loss_u = loss_bc1 + loss_bc2
        return loss_u

    @partial(jit, static_argnums=(0,))
    def loss_r(self, params, batch):
        # Fetch data
        inputs, targets = batch
        _, _, X_res = inputs
        _, _, Y_res = targets
        # Evaluate residual
        r_fn = lambda x: self.net_r(params, x)
        res = vmap(r_fn)(X_res)
        # Compute loss
        loss_r = np.mean((Y_res - res)**2)
        return loss_r

    def loss(self, params, batch, weights=(1.0,1.0)):
        w_u, w_r = weights
        loss_u = self.loss_u(params, batch)
        loss_r = self.loss_r(params, batch)
        loss = w_u*loss_u + w_r*loss_r
        return loss

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict(self, params, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        u_fn = lambda x: self.net_u(params, x)
        u_star = vmap(u_fn)(X_star)
        return u_star

    @partial(jit, static_argnums=(0,))
    def residual(self, params, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        r_fn = lambda x: self.net_r(params, x)
        r_star = vmap(r_fn)(X_star)
        return r_star

    @partial(jit, static_argnums=(0,))
    def update_NTK_weights(self, params, batch):
        inputs, _ = batch
        X_bc1, X_bc2, X_res = inputs
        X_bcs = np.concatenate([X_bc1, X_bc2], axis = 0)
        diag_u = vmap(grad(self.net_u), in_axes = (None,0))(params, X_bcs)
        diag_r = vmap(grad(self.net_r), in_axes = (None,0))(params, X_res)
        diag_u, _ = ravel_pytree(diag_u)
        diag_r, _ = ravel_pytree(diag_r)
        trace_u = np.dot(diag_u, diag_u)
        trace_r = np.dot(diag_r, diag_r)
        T = trace_u + trace_r
        w_u = T/trace_u
        w_r = T/trace_r
        return w_u, w_r

    @partial(jit, static_argnums=(0,))
    def compute_NTK_spectrum(self, params, batch):
        inputs, _ = batch
        X_bc1, X_bc2, X_res = inputs
        X_bcs = np.concatenate([X_bc1, X_bc2], axis = 0)
        X_bcs = (X_bcs - self.mu_X)/self.sigma_X
        X_res = (X_res - self.mu_X)/self.sigma_X
        u_fn = lambda p, x: vmap(self.net_u, in_axes=(None,0))(p, x)[:,None]
        r_fn = lambda p, x: vmap(self.net_r, in_axes=(None,0))(p, x)[:,None]
        # Construct NKT operator
        K_uu = compute_ntk(u_fn, u_fn, (X_bcs,), (X_bcs,), params)
        K_ur = compute_ntk(u_fn, r_fn, (X_bcs,), (X_res,), params)
        K_rr = compute_ntk(r_fn, r_fn, (X_res,), (X_res,), params)
        K = np.concatenate([np.concatenate([K_uu, K_ur], axis = 1),
                            np.concatenate([K_ur.T, K_rr], axis = 1)], axis = 0)
        # Spectral decomposition
        v, w = np.linalg.eigh(K)
        # Sort eigenvalues
        idx = np.argsort(v)[::-1]
        evals = v[idx]
        evecs = w[:,idx]
        return evals, evecs


class Wave1D(PINN):
    ''' 1D Wave equation solver '''
    # Initialize the class
    def __init__(self, c, mu_X = 0.0, sigma_X = 1.0):
        super().__init__(mu_X, sigma_X)
        self.c = c
        self.num_loss_terms = 3

    # Make sure this accepts a single input data-point and returns a scalar!
    # Then use vmap to vectorize
    def net_u(self, params, x, t):
        inputs = np.stack([x, t])
        u = self.net_apply(params, inputs)
        return u[0]

    def net_u_t(self, params, x, t):
        u_t = grad(self.net_u, 2)(params, x, t)/self.sigma_X[1]
        return u_t

    def net_r(self, params, x, t):
        # Compute derivatives
        u_xx = grad(grad(self.net_u, 1), 1)(params, x, t)/self.sigma_X[0]**2
        u_tt = grad(grad(self.net_u, 2), 2)(params, x, t)/self.sigma_X[1]**2
        # Compute residual
        res = u_tt - self.c**2 * u_xx
        return res

    def loss_u(self, params, batch):
        # Fetch data
        inputs, targets = batch
        X_ic1, _, X_bc1, X_bc2, _ = inputs
        Y_ic1, _, Y_bc1, Y_bc2, _ = targets
        # Evaluate model
        u_fn = lambda x, t: self.net_u(params, x, t)
        ut_fn = lambda x, t: self.net_u_t(params, x, t)
        u_ic1 = vmap(u_fn, in_axes=(0,0))(X_ic1[:,0], X_ic1[:,1])
        u_bc1 = vmap(u_fn, in_axes=(0,0))(X_bc1[:,0], X_bc1[:,1])
        u_bc2 = vmap(u_fn, in_axes=(0,0))(X_bc2[:,0], X_bc2[:,1])
        # Compute loss
        loss_ic1 = np.mean((Y_ic1 - u_ic1)**2)
        loss_bc1 = np.mean((Y_bc1 - u_bc1)**2)
        loss_bc2 = np.mean((Y_bc2 - u_bc2)**2)
        loss_u = loss_ic1 + loss_bc1 + loss_bc2
        return loss_u

    def loss_u_t(self, params, batch):
        # Fetch data
        inputs, targets = batch
        _, X_ic2, _, _, _ = inputs
        _, Y_ic2, _, _, _ = targets
        # Evaluate model
        ut_fn = lambda x, t: self.net_u_t(params, x, t)
        u_ic2 = vmap(ut_fn, in_axes=(0,0))(X_ic2[:,0], X_ic2[:,1])
        # Compute loss
        loss_u_t = np.mean((Y_ic2 - u_ic2)**2)
        return loss_u_t

    def loss_r(self, params, batch):
        # Fetch data
        inputs, targets = batch
        _, _, _, _, X_res = inputs
        _, _, _, _, Y_res = targets
        # Evaluate residual
        r_fn = lambda x, t: self.net_r(params, x, t)
        res = vmap(r_fn, in_axes=(0,0))(X_res[:,0], X_res[:,1])
        # Compute loss
        loss_r = np.mean((Y_res - res)**2)
        return loss_r

    def loss(self, params, batch, weights):
        w_u, w_u_t, w_r = weights
        loss_u = self.loss_u(params, batch)
        loss_u_t = self.loss_u_t(params, batch)
        loss_r = self.loss_r(params, batch)
        loss = w_u*loss_u + w_u_t*loss_u_t + w_r*loss_r
        return loss

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict(self, params, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        u_fn = lambda x, t: self.net_u(params, x, t)
        u_star = vmap(u_fn, in_axes=(0,0))(X_star[:,0], X_star[:,1])
        return u_star

    @partial(jit, static_argnums=(0,))
    def residual(self, params, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        r_fn = lambda x, t: self.net_r(params, x, t)
        r_star = vmap(r_fn, in_axes=(0,0))(X_star[:,0], X_star[:,1])
        return r_star

    @partial(jit, static_argnums=(0,))
    def update_NTK_weights(self, params, batch):
        inputs, _ = batch
        X_ic1, X_ic2, X_bc1, X_bc2, X_res = inputs
        X_bcs = np.concatenate([X_ic1, X_bc1, X_bc2], axis = 0)
        # Compute gradients
        diag_u = vmap(grad(self.net_u), in_axes = (None,0,0))(params,
                                                              X_bcs[:,0],
                                                              X_bcs[:,1])
        diag_u_t = vmap(grad(self.net_u_t), in_axes = (None,0,0))(params,
                                                                  X_ic2[:,0],
                                                                  X_ic2[:,1])
        diag_r = vmap(grad(self.net_r), in_axes = (None,0,0))(params,
                                                              X_res[:,0],
                                                              X_res[:,1])
        # Flatten parameter pytrees
        diag_u, _ = ravel_pytree(diag_u)
        diag_u_t, _ = ravel_pytree(diag_u_t)
        diag_r, _ = ravel_pytree(diag_r)
        # Compute traces of the Jacobian sub-blocks
        trace_u = np.dot(diag_u, diag_u)
        trace_u_t = np.dot(diag_u_t, diag_u_t)
        trace_r = np.dot(diag_r, diag_r)
        T = trace_u + trace_u_t + trace_r
        # Compute weights
        w_u = T/trace_u
        w_u_t = T/trace_u_t
        w_r = T/trace_r
        return w_u, w_u_t, w_r

    @partial(jit, static_argnums=(0,))
    def compute_NTK_spectrum(self, params, batch):
        inputs, _ = batch
        X_ic1, X_ic2, X_bc1, X_bc2, X_res = inputs
        X_bcs = np.concatenate([X_ic1, X_bc1, X_bc2], axis = 0)
        X_bcs = (X_bcs - self.mu_X)/self.sigma_X
        X_ic2 = (X_ic2 - self.mu_X)/self.sigma_X
        X_res = (X_res - self.mu_X)/self.sigma_X
        # Helper functions
        u_fn = lambda p, x, t: vmap(self.net_u, in_axes=(None,0,0))(p, x, t)[:,None]
        u_t_fn = lambda p, x, t: vmap(self.net_u_t, in_axes=(None,0,0))(p, x, t)[:,None]
        r_fn = lambda p, x, t: vmap(self.net_r, in_axes=(None,0,0))(p, x, t)[:,None]
        # Row 1
        K_uu = compute_ntk(params, u_fn, u_fn,
                           (X_bcs[:,0], X_bcs[:,1]),
                           (X_bcs[:,0], X_bcs[:,1]))
        k_uut = compute_ntk(params, u_fn, u_t_fn,
                            (X_bcs[:,0], X_bcs[:,1]),
                            (X_ic2[:,0], X_ic2[:,1]))
        K_ur = compute_ntk(params, u_fn, r_fn,
                            (X_bcs[:,0], X_bcs[:,1]),
                            (X_res[:,0], X_res[:,1]))
        # Row 2
        K_utut = compute_ntk(params, u_t_fn, u_t_fn,
                            (X_ic2[:,0], X_ic2[:,1]),
                            (X_ic2[:,0], X_ic2[:,1]))
        K_utr = compute_ntk(params, u_t_fn, r_fn,
                            (X_ic2[:,0], X_ic2[:,1]),
                            (X_res[:,0], X_res[:,1]))
        # Row 3
        K_rr = compute_ntk(params, r_fn, r_fn,
                           (X_res[:,0], X_res[:,1]),
                           (X_res[:,0], X_res[:,1]))
        # Assemble NTK blocks
        K = np.concatenate([np.concatenate([K_uu, k_uut, K_ur], axis = 1),
                            np.concatenate([k_uut.T, K_utut, K_utr], axis = 1),
                            np.concatenate([K_ur.T, K_utr.T, K_rr], axis = 1)], axis = 0)
        # Spectral decomposition
        v, w = np.linalg.eigh(K)
        # Sort eigenvalues
        idx = np.argsort(v)[::-1]
        evals = v[idx]
        evecs = w[:,idx]
        return evals, evecs


class IncNavierStokes4DFlowMRI(PINN):
    ''' 3D incompressible Navier-Stokes solver '''
    # Initialize the class
    def __init__(self, Re, mu_X = 0.0, sigma_X = 1.0):
        super().__init__(mu_X, sigma_X)
        self.Re = Re
        self.num_loss_terms = 2

    # Make sure this accepts a single input data-point and returns a scalar!
    # Then use vmap to vectorize
    def net_psi(self, params, x, y, z, t):
        inputs = np.stack([x, y, z, t])
        out = self.net_apply(params, inputs)
        return out[0]

    def net_chi(self, params, x, y, z, t):
        inputs = np.stack([x, y, z, t])
        out = self.net_apply(params, inputs)
        return out[1]

    def net_p(self, params, x, y, z, t):
        inputs = np.stack([x, y, z, t])
        out = self.net_apply(params, inputs)
        return out[2]

    def net_u(self, params, x, y, z, t):
        psi_y = grad(self.net_psi, 2)(params, x, y, z, t)/self.sigma_X[1]
        psi_z = grad(self.net_psi, 3)(params, x, y, z, t)/self.sigma_X[2]
        chi_y = grad(self.net_chi, 2)(params, x, y, z, t)/self.sigma_X[1]
        chi_z = grad(self.net_chi, 3)(params, x, y, z, t)/self.sigma_X[2]
        u = psi_y*chi_z - psi_z*chi_y
        return u

    def net_v(self, params, x, y, z, t):
        psi_z = grad(self.net_psi, 3)(params, x, y, z, t)/self.sigma_X[2]
        psi_x = grad(self.net_psi, 1)(params, x, y, z, t)/self.sigma_X[0]
        chi_x = grad(self.net_chi, 1)(params, x, y, z, t)/self.sigma_X[0]
        chi_z = grad(self.net_chi, 3)(params, x, y, z, t)/self.sigma_X[2]
        v = psi_z*chi_x - psi_x*chi_z
        return v

    def net_w(self, params, x, y, z, t):
        psi_x = grad(self.net_psi, 1)(params, x, y, z, t)/self.sigma_X[0]
        psi_y = grad(self.net_psi, 2)(params, x, y, z, t)/self.sigma_X[1]
        chi_y = grad(self.net_chi, 2)(params, x, y, z, t)/self.sigma_X[1]
        chi_x = grad(self.net_chi, 1)(params, x, y, z, t)/self.sigma_X[0]
        w = psi_x*chi_y - psi_y*chi_x
        return w

    def net_div(self, params, x, y, z, t):
        u_x = grad(self.net_u, 1)(params, x, y, z, t)/self.sigma_X[0]
        v_y = grad(self.net_v, 2)(params, x, y, z, t)/self.sigma_X[1]
        w_z = grad(self.net_w, 3)(params, x, y, z, t)/self.sigma_X[2]
        div = u_x + v_y + w_z
        return div

    def net_r(self, params, x, y, z, t):
        # State variables
        u = self.net_u(params, x, y, z, t)
        v = self.net_v(params, x, y, z, t)
        w = self.net_w(params, x, y, z, t)
        p = self.net_p(params, x, y, z, t)

        # Gradients
        u_x = grad(self.net_u, 1)(params, x, y, z, t)/self.sigma_X[0]
        u_y = grad(self.net_u, 2)(params, x, y, z, t)/self.sigma_X[1]
        u_z = grad(self.net_u, 3)(params, x, y, z, t)/self.sigma_X[2]
        u_t = grad(self.net_u, 4)(params, x, y, z, t)/self.sigma_X[3]

        v_x = grad(self.net_v, 1)(params, x, y, z, t)/self.sigma_X[0]
        v_y = grad(self.net_v, 2)(params, x, y, z, t)/self.sigma_X[1]
        v_z = grad(self.net_v, 3)(params, x, y, z, t)/self.sigma_X[2]
        v_t = grad(self.net_v, 4)(params, x, y, z, t)/self.sigma_X[3]

        w_x = grad(self.net_w, 1)(params, x, y, z, t)/self.sigma_X[0]
        w_y = grad(self.net_w, 2)(params, x, y, z, t)/self.sigma_X[1]
        w_z = grad(self.net_w, 3)(params, x, y, z, t)/self.sigma_X[2]
        w_t = grad(self.net_w, 4)(params, x, y, z, t)/self.sigma_X[3]

        p_x = grad(self.net_p, 1)(params, x, y, z, t)/self.sigma_X[0]
        p_y = grad(self.net_p, 2)(params, x, y, z, t)/self.sigma_X[1]
        p_z = grad(self.net_p, 3)(params, x, y, z, t)/self.sigma_X[2]

        u_xx = grad(grad(self.net_u, 1), 1)(params, x, y, z, t)/self.sigma_X[0]**2
        u_yy = grad(grad(self.net_u, 2), 2)(params, x, y, z, t)/self.sigma_X[1]**2
        u_zz = grad(grad(self.net_u, 3), 3)(params, x, y, z, t)/self.sigma_X[2]**2

        v_xx = grad(grad(self.net_v, 1), 1)(params, x, y, z, t)/self.sigma_X[0]**2
        v_yy = grad(grad(self.net_v, 2), 2)(params, x, y, z, t)/self.sigma_X[1]**2
        v_zz = grad(grad(self.net_v, 3), 3)(params, x, y, z, t)/self.sigma_X[2]**2

        w_xx = grad(grad(self.net_w, 1), 1)(params, x, y, z, t)/self.sigma_X[0]**2
        w_yy = grad(grad(self.net_w, 2), 2)(params, x, y, z, t)/self.sigma_X[1]**2
        w_zz = grad(grad(self.net_w, 3), 3)(params, x, y, z, t)/self.sigma_X[2]**2

        # Residuals
        res_u = u_t + (u * u_x + v * u_y + w * u_z) + p_x - (u_xx + u_yy + u_zz) / self.Re
        res_v = v_t + (u * v_x + v * v_y + w * v_z) + p_y - (v_xx + v_yy + v_zz) / self.Re
        res_w = w_t + (u * w_x + v * w_y + w * w_z) + p_z - (w_xx + w_yy + w_zz) / self.Re

        return res_u, res_v, res_w

    @partial(jit, static_argnums=(0,))
    def loss_u(self, params, batch):
        # Fetch data
        inputs, targets = batch
        X_dat, _ = inputs
        Y_dat, _ = targets
        # Evaluate model
        u_fn = lambda x, y, z, t: self.net_u(params, x, y, z, t)
        v_fn = lambda x, y, z, t: self.net_v(params, x, y, z, t)
        w_fn = lambda x, y, z, t: self.net_w(params, x, y, z, t)
        u = vmap(u_fn, in_axes=(0,0,0,0))(X_dat[:,0], X_dat[:,1], X_dat[:,2], X_dat[:,3])
        v = vmap(v_fn, in_axes=(0,0,0,0))(X_dat[:,0], X_dat[:,1], X_dat[:,2], X_dat[:,3])
        w = vmap(w_fn, in_axes=(0,0,0,0))(X_dat[:,0], X_dat[:,1], X_dat[:,2], X_dat[:,3])
        # Compute loss
        loss_u = np.mean((Y_dat[:,0] - u)**2) + \
                 np.mean((Y_dat[:,1] - v)**2) + \
                 np.mean((Y_dat[:,2] - w)**2)
        return loss_u

    @partial(jit, static_argnums=(0,))
    def loss_r(self, params, batch):
        # Fetch data
        inputs, targets = batch
        _, X_res = inputs
        _, Y_res = targets
        # Evaluate model
        r_fn = lambda x, y, z, t: self.net_r(params, x, y, z, t)
        res_u, res_v, res_w = vmap(r_fn, in_axes=(0,0,0,0))(X_res[:,0], X_res[:,1], X_res[:,2], X_res[:,3])
        # Compute loss
        loss_r = np.mean((Y_res[:,0] - res_u)**2) + \
                 np.mean((Y_res[:,1] - res_v)**2) + \
                 np.mean((Y_res[:,2] - res_w)**2)
        return loss_r

    def loss(self, params, batch, weights):
        w_u, w_r = weights
        loss_u = self.loss_u(params, batch)
        loss_r = self.loss_r(params, batch)
        loss = w_u*loss_u + w_r*loss_r
        return loss

    # @partial(jit, static_argnums=(0,))
    def update_NTK_weights(self, params, batch):
        raise NotImplementedError

    # @partial(jit, static_argnums=(0,))
    def compute_NTK_spectrum(self, params, batch):
        raise NotImplementedError

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict(self, params, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        u_fn = lambda x, y, z, t: self.net_u(params, x, y, z, t)
        v_fn = lambda x, y, z, t: self.net_v(params, x, y, z, t)
        w_fn = lambda x, y, z, t: self.net_w(params, x, y, z, t)
        p_fn = lambda x, y, z, t: self.net_p(params, x, y, z, t)
        u_star = vmap(u_fn, in_axes=(0,0,0,0))(X_star[:,0],
                                               X_star[:,1],
                                               X_star[:,2],
                                               X_star[:,3])
        v_star = vmap(v_fn, in_axes=(0,0,0,0))(X_star[:,0],
                                               X_star[:,1],
                                               X_star[:,2],
                                               X_star[:,3])
        w_star = vmap(w_fn, in_axes=(0,0,0,0))(X_star[:,0],
                                               X_star[:,1],
                                               X_star[:,2],
                                               X_star[:,3])
        p_star = vmap(p_fn, in_axes=(0,0,0,0))(X_star[:,0],
                                               X_star[:,1],
                                               X_star[:,2],
                                               X_star[:,3])
        return u_star, v_star, w_star, p_star

    @partial(jit, static_argnums=(0,))
    def residual(self, params, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        r_fn = lambda x, y, z, t: self.net_r(params, x, y, z, t)
        res_u, res_v, res_w = vmap(r_fn, in_axes=(0,0,0,0))(X_star[:,0],
                                                            X_star[:,1],
                                                            X_star[:,2],
                                                            X_star[:,3])
        return res_u, res_v, res_w

    @partial(jit, static_argnums=(0,))
    def divergence(self, params, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        d_fn = lambda x, y, z, t: self.net_div(params, x, y, z, t)
        div = vmap(d_fn, in_axes=(0,0,0,0))(X_star[:,0],
                                            X_star[:,1],
                                            X_star[:,2],
                                            X_star[:,3])
        return div

    def velocity_gradients(self, params, x, y, z, t):
        # State variables
        u = self.net_u(params, x, y, z, t)
        v = self.net_v(params, x, y, z, t)
        w = self.net_w(params, x, y, z, t)
        # u gradients
        u_x = grad(self.net_u, 1)(params, x, y, z, t)/self.sigma_X[0]
        u_y = grad(self.net_u, 2)(params, x, y, z, t)/self.sigma_X[1]
        u_z = grad(self.net_u, 3)(params, x, y, z, t)/self.sigma_X[2]
        # v gradients
        v_x = grad(self.net_v, 1)(params, x, y, z, t)/self.sigma_X[0]
        v_y = grad(self.net_v, 2)(params, x, y, z, t)/self.sigma_X[1]
        v_z = grad(self.net_v, 3)(params, x, y, z, t)/self.sigma_X[2]
        # w gradients
        w_x = grad(self.net_w, 1)(params, x, y, z, t)/self.sigma_X[0]
        w_y = grad(self.net_w, 2)(params, x, y, z, t)/self.sigma_X[1]
        w_z = grad(self.net_w, 3)(params, x, y, z, t)/self.sigma_X[2]
        return u_x, v_x, w_x, u_y, v_y, w_y, u_z, v_z, w_z

    @partial(jit, static_argnums=(0,))
    def predict_wss(self, params, X_star, nx, ny, nz):
        # Normalize inputs
        X_star = (X_star - self.mu_X) / self.sigma_X
        vg_fn = lambda x, y, z, t: self.velocity_gradients(params, x, y, z, t)
        # Compute velocity gradients
        u_x, v_x, w_x, \
        u_y, v_y, w_y, \
        u_z, v_z, w_z = vmap(vg_fn, in_axes=(0,0,0,0))(X_star[:,0],
                                                       X_star[:,1],
                                                       X_star[:,2],
                                                       X_star[:,3])
        # Compute WSS
        uu = u_x + u_x
        uv = u_y + v_x
        uw = u_z + w_x
        vv = v_y + v_y
        vw = v_z + w_y
        ww = w_z + w_z

        sx = (uu * nx + uv * ny + uw * nz)/self.Re
        sy = (uv * nx + vv * ny + vw * nz)/self.Re
        sz = (uw * nx + vw * ny + ww * nz)/self.Re

        s_t = sx * nx + sy * ny + sz * nz

        sx = sx - s_t * nx
        sy = sy - s_t * ny
        sz = sz - s_t * nz

        return sx, sy, sz

    @partial(jit, static_argnums=(0,))
    def predict_vorticity(self, params, X_star):
        # Normalize inputs
        X_star = (X_star - self.mu_X) / self.sigma_X
        vg_fn = lambda x, y, z, t: self.velocity_gradients(params, x, y, z, t)
        # Compute velocity gradients
        u_x, v_x, w_x, \
        u_y, v_y, w_y, \
        u_z, v_z, w_z = vmap(vg_fn, in_axes=(0,0,0,0))(X_star[:,0],
                                                       X_star[:,1],
                                                       X_star[:,2],
                                                       X_star[:,3])
        # Compute vorticity
        Vor_x = w_y - v_z
        Vor_y = u_z - w_x
        Vor_z = v_x - u_y
        return Vor_x, Vor_y, Vor_z



class AllenCahn1D(dtPINN):
    ''' 1D AllenCahn equation discrete-time solver '''
    # Initialize the class
    def __init__(self, dt, q, mu_X = 0.0, sigma_X = 1.0):
        super().__init__(dt, q, mu_X, sigma_X)
        self.num_loss_terms = 2

    def net_u(self, params, x):
        inputs = np.stack([x])
        u = self.net_apply(params, inputs)
        return u

    def net_U0(self, params, x):
        u_fn = lambda x: self.net_u(params, x)
        U1 = u_fn(x)
        U = U1[:-1]
        U_xx = jacfwd(jacfwd(u_fn))(x)[:-1]/self.sigma_X**2
        F = 5.0*U - 5.0*U**3 + 0.0001*U_xx
        U0 = U1 - self.dt*np.dot(self.IRK_weights, F)
        return U0

    def net_U1(self, params, x):
        u_fn = lambda x: self.net_u(params, x)
        U1 = u_fn(x)
        U1_x = jacfwd(u_fn)(x)/self.sigma_X
        return U1, U1_x

    def loss_U0(self, params, batch):
        # Fetch data
        inputs, targets = batch
        X0, _, _ = inputs
        Y0, _, _ = targets
        Y0 = np.tile(Y0[:,None], (1, self.q+1))
        # Evaluate model
        u_fn = lambda x: self.net_U0(params, x)
        U0 = vmap(u_fn)(X0)
        loss_U0 = np.mean((Y0 - U0)**2)
        return loss_U0

    def loss_U1(self, params, batch):
        # Fetch data
        inputs, _ = batch
        _, X1_bc1, X1_bc2 = inputs
        # Evaluate model
        u_fn = lambda x: self.net_U1(params, x)
        U1_bc1, U1_x_bc1 = vmap(u_fn)(X1_bc1)
        U1_bc2, U1_x_bc2 = vmap(u_fn)(X1_bc2)
        loss_bc1 = np.mean((U1_bc1 - U1_bc2)**2)
        loss_bc2 = np.mean((U1_x_bc1 - U1_x_bc2)**2)
        loss_U1 = loss_bc1 + loss_bc2
        return loss_U1

    def loss(self, params, batch, weights):
        w_0, w_1 = weights
        loss_U0 = self.loss_U0(params, batch)
        loss_U1 = self.loss_U1(params, batch)
        loss = w_0*loss_U0 + w_1*loss_U1
        return loss

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict(self, params, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        u_fn = lambda x: self.net_U1(params, x)
        U1_star, _ = vmap(u_fn)(X_star)
        return U1_star[:,-1]

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict_all(self, params, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        u_fn = lambda x: self.net_U1(params, x)
        U1_star, _ = vmap(u_fn)(X_star)
        return U1_star

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict_u0(self, params, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        u_fn = lambda x: self.net_U0(params, x)
        U0_star = vmap(u_fn)(X_star)
        return U0_star

    # @partial(jit, static_argnums=(0,))
    def update_NTK_weights(self, params, batch):
        raise NotImplementedError

    # @partial(jit, static_argnums=(0,))
    def compute_NTK_spectrum(self, params, batch):
        raise NotImplementedError



# class Beltrami3D(PINN):
#     ''' 3D incompressible Beltrami flow solver '''
#     # Initialize the class
#     def __init__(self, Re, mu_X = 0.0, sigma_X = 1.0):
#         super().__init__(mu_X, sigma_X)
#         self.Re = Re
#
#     # Make sure this accepts a single input data-point and returns a scalar!
#     # Then use vmap to vectorize
#     def net_psi(self, params, x, y, z, t):
#         inputs = np.stack([x, y, z, t])
#         out = self.net_apply(params, inputs)
#         return out[0]
#
#     def net_chi(self, params, x, y, z, t):
#         inputs = np.stack([x, y, z, t])
#         out = self.net_apply(params, inputs)
#         return out[1]
#
#     def net_p(self, params, x, y, z, t):
#         inputs = np.stack([x, y, z, t])
#         out = self.net_apply(params, inputs)
#         return out[2]
#
#     def net_u(self, params, x, y, z, t):
#         psi_y = grad(self.net_psi, 2)(params, x, y, z, t)/self.sigma_X[1]
#         psi_z = grad(self.net_psi, 3)(params, x, y, z, t)/self.sigma_X[2]
#         chi_y = grad(self.net_chi, 2)(params, x, y, z, t)/self.sigma_X[1]
#         chi_z = grad(self.net_chi, 3)(params, x, y, z, t)/self.sigma_X[2]
#         u = psi_y*chi_z - psi_z*chi_y
#         return u
#
#     def net_v(self, params, x, y, z, t):
#         psi_z = grad(self.net_psi, 3)(params, x, y, z, t)/self.sigma_X[2]
#         psi_x = grad(self.net_psi, 1)(params, x, y, z, t)/self.sigma_X[0]
#         chi_x = grad(self.net_chi, 1)(params, x, y, z, t)/self.sigma_X[0]
#         chi_z = grad(self.net_chi, 3)(params, x, y, z, t)/self.sigma_X[2]
#         v = psi_z*chi_x - psi_x*chi_z
#         return v
#
#     def net_w(self, params, x, y, z, t):
#         psi_x = grad(self.net_psi, 1)(params, x, y, z, t)/self.sigma_X[0]
#         psi_y = grad(self.net_psi, 2)(params, x, y, z, t)/self.sigma_X[1]
#         chi_y = grad(self.net_chi, 2)(params, x, y, z, t)/self.sigma_X[1]
#         chi_x = grad(self.net_chi, 1)(params, x, y, z, t)/self.sigma_X[0]
#         w = psi_x*chi_y - psi_y*chi_x
#         return w
#
#     def net_div(self, params, x, y, z, t):
#         u_x = grad(self.net_u, 1)(params, x, y, z, t)/self.sigma_X[0]
#         v_y = grad(self.net_v, 2)(params, x, y, z, t)/self.sigma_X[1]
#         w_z = grad(self.net_w, 3)(params, x, y, z, t)/self.sigma_X[2]
#         div = u_x + v_y + w_z
#         return div
#
#     def net_r(self, params, x, y, z, t):
#         # State variables
#         u = self.net_u(params, x, y, z, t)
#         v = self.net_v(params, x, y, z, t)
#         w = self.net_w(params, x, y, z, t)
#         p = self.net_p(params, x, y, z, t)
#
#         # Gradients
#         u_x = grad(self.net_u, 1)(params, x, y, z, t)/self.sigma_X[0]
#         u_y = grad(self.net_u, 2)(params, x, y, z, t)/self.sigma_X[1]
#         u_z = grad(self.net_u, 3)(params, x, y, z, t)/self.sigma_X[2]
#         u_t = grad(self.net_u, 4)(params, x, y, z, t)/self.sigma_X[3]
#
#         v_x = grad(self.net_v, 1)(params, x, y, z, t)/self.sigma_X[0]
#         v_y = grad(self.net_v, 2)(params, x, y, z, t)/self.sigma_X[1]
#         v_z = grad(self.net_v, 3)(params, x, y, z, t)/self.sigma_X[2]
#         v_t = grad(self.net_v, 4)(params, x, y, z, t)/self.sigma_X[3]
#
#         w_x = grad(self.net_w, 1)(params, x, y, z, t)/self.sigma_X[0]
#         w_y = grad(self.net_w, 2)(params, x, y, z, t)/self.sigma_X[1]
#         w_z = grad(self.net_w, 3)(params, x, y, z, t)/self.sigma_X[2]
#         w_t = grad(self.net_w, 4)(params, x, y, z, t)/self.sigma_X[3]
#
#         p_x = grad(self.net_p, 1)(params, x, y, z, t)/self.sigma_X[0]
#         p_y = grad(self.net_p, 2)(params, x, y, z, t)/self.sigma_X[1]
#         p_z = grad(self.net_p, 3)(params, x, y, z, t)/self.sigma_X[2]
#
#         u_xx = grad(grad(self.net_u, 1), 1)(params, x, y, z, t)/self.sigma_X[0]**2
#         u_yy = grad(grad(self.net_u, 2), 2)(params, x, y, z, t)/self.sigma_X[1]**2
#         u_zz = grad(grad(self.net_u, 3), 3)(params, x, y, z, t)/self.sigma_X[2]**2
#
#         v_xx = grad(grad(self.net_v, 1), 1)(params, x, y, z, t)/self.sigma_X[0]**2
#         v_yy = grad(grad(self.net_v, 2), 2)(params, x, y, z, t)/self.sigma_X[1]**2
#         v_zz = grad(grad(self.net_v, 3), 3)(params, x, y, z, t)/self.sigma_X[2]**2
#
#         w_xx = grad(grad(self.net_w, 1), 1)(params, x, y, z, t)/self.sigma_X[0]**2
#         w_yy = grad(grad(self.net_w, 2), 2)(params, x, y, z, t)/self.sigma_X[1]**2
#         w_zz = grad(grad(self.net_w, 3), 3)(params, x, y, z, t)/self.sigma_X[2]**2
#
#         # Residuals
#         res_u = u_t + self.Re*(u*u_x + v*u_y + w*u_z) + self.Re*p_x - (u_xx + u_yy + u_zz)
#         res_v = v_t + self.Re*(u*v_x + v*v_y + w*v_z) + self.Re*p_y - (v_xx + v_yy + v_zz)
#         res_w = w_t + self.Re*(u*w_x + v*w_y + w*w_z) + self.Re*p_z - (w_xx + w_yy + w_zz)
#
#         res = (res_u + res_v + res_w)/3.0
#         return res
#
#
#     def loss_u(self, params, batch):
#         # Fetch data
#         inputs, targets = batch
#         X_ics, X_bc1, X_bc2, X_bc3, X_bc4, X_bc5, X_bc6, _ = inputs
#         Y_ics, Y_bc1, Y_bc2, Y_bc3, Y_bc4, Y_bc5, Y_bc6, _ = targets
#         # Evaluate model
#         u_fn = lambda x, y, z, t: self.net_u(params, x, y, z, t)
#         v_fn = lambda x, y, z, t: self.net_v(params, x, y, z, t)
#         w_fn = lambda x, y, z, t: self.net_w(params, x, y, z, t)
#         # ICS
#         u_ics = vmap(u_fn, in_axes=(0,0,0,0))(X_ics[:,0], X_ics[:,1], X_ics[:,2], X_ics[:,3])
#         v_ics = vmap(v_fn, in_axes=(0,0,0,0))(X_ics[:,0], X_ics[:,1], X_ics[:,2], X_ics[:,3])
#         w_ics = vmap(w_fn, in_axes=(0,0,0,0))(X_ics[:,0], X_ics[:,1], X_ics[:,2], X_ics[:,3])
#         loss_ics = np.mean((u_ics - Y_ics[:,0])**2) + \
#                    np.mean((v_ics - Y_ics[:,1])**2) + \
#                    np.mean((w_ics - Y_ics[:,2])**2)
#         # BC1
#         u_bc1 = vmap(u_fn, in_axes=(0,0,0,0))(X_bc1[:,0], X_bc1[:,1], X_bc1[:,2], X_bc1[:,3])
#         v_bc1 = vmap(v_fn, in_axes=(0,0,0,0))(X_bc1[:,0], X_bc1[:,1], X_bc1[:,2], X_bc1[:,3])
#         w_bc1 = vmap(w_fn, in_axes=(0,0,0,0))(X_bc1[:,0], X_bc1[:,1], X_bc1[:,2], X_bc1[:,3])
#         loss_bc1 = np.mean((u_bc1 - Y_bc1[:,0])**2) + \
#                    np.mean((v_bc1 - Y_bc1[:,1])**2) + \
#                    np.mean((w_bc1 - Y_bc1[:,2])**2)
#         # BC2
#         u_bc2 = vmap(u_fn, in_axes=(0,0,0,0))(X_bc2[:,0], X_bc2[:,1], X_bc2[:,2], X_bc2[:,3])
#         v_bc2 = vmap(v_fn, in_axes=(0,0,0,0))(X_bc2[:,0], X_bc2[:,1], X_bc2[:,2], X_bc2[:,3])
#         w_bc2 = vmap(w_fn, in_axes=(0,0,0,0))(X_bc2[:,0], X_bc2[:,1], X_bc2[:,2], X_bc2[:,3])
#         loss_bc2 = np.mean((u_bc2 - Y_bc2[:,0])**2) + \
#                    np.mean((v_bc2 - Y_bc2[:,1])**2) + \
#                    np.mean((w_bc2 - Y_bc2[:,2])**2)
#         # BC3
#         u_bc3 = vmap(u_fn, in_axes=(0,0,0,0))(X_bc3[:,0], X_bc3[:,1], X_bc3[:,2], X_bc3[:,3])
#         v_bc3 = vmap(v_fn, in_axes=(0,0,0,0))(X_bc3[:,0], X_bc3[:,1], X_bc3[:,2], X_bc3[:,3])
#         w_bc3 = vmap(w_fn, in_axes=(0,0,0,0))(X_bc3[:,0], X_bc3[:,1], X_bc3[:,2], X_bc3[:,3])
#         loss_bc3 = np.mean((u_bc3 - Y_bc3[:,0])**2) + \
#                    np.mean((v_bc3 - Y_bc3[:,1])**2) + \
#                    np.mean((w_bc3 - Y_bc3[:,2])**2)
#         # BC4
#         u_bc4 = vmap(u_fn, in_axes=(0,0,0,0))(X_bc4[:,0], X_bc4[:,1], X_bc4[:,2], X_bc4[:,3])
#         v_bc4 = vmap(v_fn, in_axes=(0,0,0,0))(X_bc4[:,0], X_bc4[:,1], X_bc4[:,2], X_bc4[:,3])
#         w_bc4 = vmap(w_fn, in_axes=(0,0,0,0))(X_bc4[:,0], X_bc4[:,1], X_bc4[:,2], X_bc4[:,3])
#         loss_bc4 = np.mean((u_bc4 - Y_bc4[:,0])**2) + \
#                    np.mean((v_bc4 - Y_bc4[:,1])**2) + \
#                    np.mean((w_bc4 - Y_bc4[:,2])**2)
#         # BC5
#         u_bc5 = vmap(u_fn, in_axes=(0,0,0,0))(X_bc5[:,0], X_bc5[:,1], X_bc5[:,2], X_bc5[:,3])
#         v_bc5 = vmap(v_fn, in_axes=(0,0,0,0))(X_bc5[:,0], X_bc5[:,1], X_bc5[:,2], X_bc5[:,3])
#         w_bc5 = vmap(w_fn, in_axes=(0,0,0,0))(X_bc5[:,0], X_bc5[:,1], X_bc5[:,2], X_bc5[:,3])
#         loss_bc5 = np.mean((u_bc5 - Y_bc5[:,0])**2) + \
#                    np.mean((v_bc5 - Y_bc5[:,1])**2) + \
#                    np.mean((w_bc5 - Y_bc5[:,2])**2)
#         # BC6
#         u_bc6 = vmap(u_fn, in_axes=(0,0,0,0))(X_bc6[:,0], X_bc6[:,1], X_bc6[:,2], X_bc6[:,3])
#         v_bc6 = vmap(v_fn, in_axes=(0,0,0,0))(X_bc6[:,0], X_bc6[:,1], X_bc6[:,2], X_bc6[:,3])
#         w_bc6 = vmap(w_fn, in_axes=(0,0,0,0))(X_bc6[:,0], X_bc6[:,1], X_bc6[:,2], X_bc6[:,3])
#         loss_bc6 = np.mean((u_bc6 - Y_bc6[:,0])**2) + \
#                    np.mean((v_bc6 - Y_bc6[:,1])**2) + \
#                    np.mean((w_bc6 - Y_bc6[:,2])**2)
#         # Compute total loss
#         loss_u = loss_ics + loss_bc1 + loss_bc2 + loss_bc3 + loss_bc4 + loss_bc5 + loss_bc6
#         return loss_u
#
#     def loss_r(self, params, batch):
#         # Fetch data
#         inputs, targets = batch
#         _, _, _, _, _, _, _, X_res = inputs
#         _, _, _, _, _, _, _, Y_res = targets
#         # Evaluate model
#         r_fn = lambda x, y, z, t: self.net_r(params, x, y, z, t)
#         res = vmap(r_fn, in_axes=(0,0,0,0))(X_res[:,0], X_res[:,1], X_res[:,2], X_res[:,3])
#         # Compute loss
#         loss_r = np.mean((Y_res - res)**2)
#         return loss_r
#
#     def loss(self, params, batch, weights):
#         w_u, w_r = weights
#         loss_u = self.loss_u(params, batch)
#         loss_r = self.loss_r(params, batch)
#         loss = w_u*loss_u + w_r*loss_r
#         return loss
#
#     # Evaluates predictions at test points
#     @partial(jit, static_argnums=(0,))
#     def predict(self, params, X_star):
#         X_star = (X_star - self.mu_X) / self.sigma_X
#         u_fn = lambda x, y, z, t: self.net_u(params, x, y, z, t)
#         v_fn = lambda x, y, z, t: self.net_v(params, x, y, z, t)
#         w_fn = lambda x, y, z, t: self.net_w(params, x, y, z, t)
#         p_fn = lambda x, y, z, t: self.net_p(params, x, y, z, t)
#         u_star = vmap(u_fn, in_axes=(0,0,0,0))(X_star[:,0], X_star[:,1],X_star[:,2], X_star[:,3])
#         v_star = vmap(v_fn, in_axes=(0,0,0,0))(X_star[:,0], X_star[:,1],X_star[:,2], X_star[:,3])
#         w_star = vmap(w_fn, in_axes=(0,0,0,0))(X_star[:,0], X_star[:,1],X_star[:,2], X_star[:,3])
#         p_star = vmap(p_fn, in_axes=(0,0,0,0))(X_star[:,0], X_star[:,1],X_star[:,2], X_star[:,3])
#         return u_star, v_star, w_star, p_star
#
#     @partial(jit, static_argnums=(0,))
#     def residual(self, params, X_star):
#         X_star = (X_star - self.mu_X) / self.sigma_X
#         r_fn = lambda x, y, z, t: self.net_r(params, x, y, z, t)
#         res = vmap(r_fn, in_axes=(0,0,0,0))(X_star[:,0], X_star[:,1],X_star[:,2], X_star[:,3])
#         return res
#
#     @partial(jit, static_argnums=(0,))
#     def divergence(self, params, X_star):
#         X_star = (X_star - self.mu_X) / self.sigma_X
#         d_fn = lambda x, y, z, t: self.net_div(params, x, y, z, t)
#         div = vmap(d_fn, in_axes=(0,0,0,0))(X_star[:,0], X_star[:,1],X_star[:,2], X_star[:,3])
#         return div
#
#     @partial(jit, static_argnums=(0,))
#     def update_NTK_weights(self, params, batch):
#         # Fetch data
#         inputs, _ = batch
#         X_ics, X_bc1, X_bc2, X_bc3, X_bc4, X_bc5, X_bc6, X_res = inputs
#         X_bcs = np.concatenate([X_ics, X_bc1, X_bc2, X_bc3, X_bc4, X_bc5, X_bc6], axis = 0)
#         # Compute gradients
#         diag_u = vmap(grad(self.net_u), in_axes = (None,0,0,0,0))(params,
#                                                                   X_bcs[:,0],
#                                                                   X_bcs[:,1],
#                                                                   X_bcs[:,2],
#                                                                   X_bcs[:,3])
#         diag_r = vmap(grad(self.net_r), in_axes = (None,0,0,0,0))(params,
#                                                                   X_res[:,0],
#                                                                   X_res[:,1],
#                                                                   X_res[:,2],
#                                                                   X_res[:,3])
#         # Flatten parameter pytrees
#         diag_u, _ = ravel_pytree(diag_u)
#         diag_r, _ = ravel_pytree(diag_r)
#         # Compute traces of the Jacobian sub-blocks
#         trace_u = np.dot(diag_u, diag_u)
#         trace_r = np.dot(diag_r, diag_r)
#         T = trace_u + trace_r
#         # Compute weights
#         w_u = T/trace_u
#         w_r = T/trace_r
#         return w_u, w_r
#
#     @partial(jit, static_argnums=(0,))
#     def compute_NTK_spectrum(self, params, batch):
#         inputs, _ = batch
#         X_ics, X_bc1, X_bc2, X_bc3, X_bc4, X_bc5, X_bc6, X_res = inputs
#         X_bcs = np.concatenate([X_ics, X_bc1, X_bc2, X_bc3, X_bc4, X_bc5, X_bc6], axis = 0)
#         X_bcs = (X_bcs - self.mu_X)/self.sigma_X
#         X_res = (X_res - self.mu_X)/self.sigma_X
#         # Helper functions
#         u_fn = lambda p, x, y, z, t: vmap(self.net_u, in_axes=(None,0,0,0,0))(p, x, y, z, t)[:,None]
#         r_fn = lambda p, x, y, z, t: vmap(self.net_r, in_axes=(None,0,0,0,0))(p, x, y, z, t)[:,None]
#         # Row 1
#         K_uu = compute_ntk(params, u_fn, u_fn,
#                            (X_bcs[:,0], X_bcs[:,1], X_bcs[:,2], X_bcs[:,3]),
#                            (X_bcs[:,0], X_bcs[:,1], X_bcs[:,2], X_bcs[:,3]))
#
#         K_ur = compute_ntk(params, u_fn, r_fn,
#                             (X_bcs[:,0], X_bcs[:,1], X_bcs[:,2], X_bcs[:,3]),
#                             (X_res[:,0], X_res[:,1], X_res[:,2], X_res[:,3]))
#         # Row 2
#         K_rr = compute_ntk(params, r_fn, r_fn,
#                            (X_res[:,0], X_res[:,1], X_res[:,2], X_res[:,3]),
#                            (X_res[:,0], X_res[:,1], X_res[:,2], X_res[:,3]))
#         # Assemble NTK blocks
#         K = np.concatenate([np.concatenate([K_uu, K_ur], axis = 1),
#                             np.concatenate([K_ur.T, K_rr], axis = 1)], axis = 0)
#         # Spectral decomposition
#         v, w = np.linalg.eigh(K)
#         # Sort eigenvalues
#         idx = np.argsort(v)[::-1]
#         evals = v[idx]
#         evecs = w[:,idx]
#         return evals, evecs
