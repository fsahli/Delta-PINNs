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
from jax import grad
from jax.example_libraries.optimizers import make_schedule
from jax import lax
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree
from jax.nn import relu

def momentum(step_size, f_fn, mass=0.0):
    # Callable learning rate schedule
    step_size = make_schedule(step_size)
    # Init function
    def init(x0):
        v0 = tree_map(np.zeros_like, x0)
        cnt = 0
        return x0, v0
    # Update function
    def update(i, state, batch, weights):
        x, v = state
        # Define update rules
        v_update = lambda v, g: mass*v + g
        x_update = lambda x, v: x - step_size(i)*v
        # Perform update
        g = grad(f_fn)(x, batch, weights)
        v = tree_map(v_update, v, g)
        x = tree_map(x_update, x, v)
        return x, v
    # Get parameters function
    def get_params(state):
        x, _ = state
        return x
    return init, update, get_params


def nesterov(step_size, f_fn, mass=0.0):
    # Callable learning rate schedule
    step_size = make_schedule(step_size)
    # Init function
    def init(x0):
        v0 = tree_map(np.zeros_like, x0)
        return x0, v0
    # Update function
    def update(i, state, batch, weights):
        x, v = state
        # Define update rules
        v_update = lambda v, g: mass*v + g
        x_update = lambda x, v, g: x - step_size(i)*(mass*v + g)
        # Perform update
        g = grad(f_fn)(x, batch, weights)
        v = tree_map(v_update, v, g)
        x = tree_map(x_update, x, v, g)
        return x, v
    # Get parameters function
    def get_params(state):
        x, _ = state
        return x
    return init, update, get_params


def rmsprop(step_size, f_fn, gamma=0.9, eps=1e-8, momentum=0.0):
    # Callable learning rate schedule
    step_size = make_schedule(step_size)
    # Init function
    def init(x0):
        avg_sq_grad = tree_map(np.zeros_like, x0)
        mom = tree_map(np.zeros_like, x0)
        return x0, a0, m0
    # Update function
    def update(i, state, batch, weights):
        x, a, m = state
        # Define update rules
        a_update = lambda a, g: a * gamma + np.square(g) * (1.0 - gamma)
        m_update = lambda m, g, a: momentum * m + step_size(i) * g / np.sqrt(a + eps)
        x_update = lambda x, m: x - m
        # Perform update
        g = grad(f_fn)(x, batch, weights)
        a = tree_map(a_update, a, g)
        m = tree_map(m_update, m, g, a)
        x = tree_map(x_update, x, m)
        return x, a, m
    # Get parameters function
    def get_params(state):
        x, _, _ = state
        return x
    return init, update, get_params


def adam(step_size, f_fn,
         b1=0.9, b2=0.999, eps=1e-8):
    # Callable learning rate schedule
    step_size = make_schedule(step_size)
    # Init function
    def init(x0):
        m0 = tree_map(np.zeros_like, x0)
        v0 = tree_map(np.zeros_like, x0)
        cnt = 0
        return x0, m0, v0, cnt
    # Update function
    def update(i, state, batch, weights):
        x, m, v, cnt = state
        # Define update rules
        x_update = lambda x, g: x - step_size(i)*g
        m_update = lambda m, g: (1.0 - b1) * g + b1 * m
        v_update = lambda v, g: (1.0 - b2) * np.square(g) + b2 * v
        mhat_upate = lambda m: m / (1.0 - np.asarray(b1, m.dtype) ** (cnt+1))
        vhat_upate = lambda v: v / (1.0 - np.asarray(b2, v.dtype) ** (cnt+1))
        g_update = lambda m, v: m / (np.sqrt(v) + eps)
        # Perform update
        g = grad(f_fn)(x, batch, weights)
        m = tree_map(m_update, m, g)
        v = tree_map(v_update, v, g)
        mhat = tree_map(mhat_upate, m)
        vhat = tree_map(vhat_upate, v)
        g = tree_map(g_update, mhat, vhat)
        x = tree_map(x_update, x, g)
        cnt += 1
        return x, m, v, cnt
    # Get parameters function
    def get_params(state):
        x, _, _, _ = state
        return x
    return init, update, get_params


def mdmm_adam(step_size, objective, constraints, epsilon,
              lam_lr = 1, damp_const = 10,
              b1=0.9, b2=0.999, eps=1e-8):
    # Callable learning rate schedule
    step_size = make_schedule(step_size)
    f_fn = lambda x, batch, weights: weights[0]*objective(x, batch) + np.dot(constraints(x, batch), weights[1:])
    # Init function
    def init(x0):
        m0 = tree_map(np.zeros_like, x0)
        v0 = tree_map(np.zeros_like, x0)
        lam = tree_map(np.zeros_like, epsilon)
        cnt = 0
        return x0, m0, v0, lam, cnt
    # Update function
    def update(i, state, batch, weights):
        x, m, v, lam, cnt = state
        # Define update rules
        d_update = lambda c, e: damp_const * lax.stop_gradient(e - c)
        w_update = lambda l, d: np.array([1., *(l - d)])
        x_update = lambda x, g: x - step_size(i)*g
        m_update = lambda m, g: (1.0 - b1) * g + b1 * m
        v_update = lambda v, g: (1.0 - b2) * np.square(g) + b2 * v
        mhat_upate = lambda m: m / (1.0 - np.asarray(b1, m.dtype) ** (cnt+1))
        vhat_upate = lambda v: v / (1.0 - np.asarray(b2, v.dtype) ** (cnt+1))
        g_update = lambda m, v: m / (np.sqrt(v) + eps)
        l_update = lambda l, c, e: relu(l + lam_lr * (c - e))
        # Perform update
        cons = constraints(x, batch)
        damp = tree_map(d_update, cons, epsilon)
        weights = tree_map(w_update, lam, damp)
        g = grad(f_fn)(x, batch, weights)
        m = tree_map(m_update, m, g)
        v = tree_map(v_update, v, g)
        mhat = tree_map(mhat_upate, m)
        vhat = tree_map(vhat_upate, v)
        g = tree_map(g_update, mhat, vhat)
        x = tree_map(x_update, x, g)
        lam = tree_map(l_update, lam, cons, epsilon)
        cnt += 1
        return x, m, v, lam, cnt
    # Get parameters function
    def get_params(state):
        x, _, _, _, _ = state
        return x
    return init, update, get_params



def admm_adam(step_size_f, step_size_g, f_fn, g_fn, rho = 1.0,
              num_inner_iter_f = 1, num_inner_iter_g = 1,
              b1=0.9, b2=0.999, eps=1e-8):
    # Create learning rate schedules
    step_size_f = make_schedule(step_size_f)
    step_size_g = make_schedule(step_size_g)
    # Define regularization term
    def regularizer(x, y, z):
        x, _ = ravel_pytree(x)
        y, _ = ravel_pytree(y)
        z, _ = ravel_pytree(z)
        reg = np.sum((x-y+z)**2)
        return reg
    # Define regularized objectives
    f = lambda x, y, w, batch, weight: weight*f_fn(x, batch) + 0.5*rho*regularizer(x, y, w)
    g = lambda x, y, w, batch, weight: weight*g_fn(y, batch) + 0.5*rho*regularizer(x, y, w)
    # Init function
    def init(x0):
        y0 = tree_map(lambda x: x, x0)
        w0 = tree_map(np.zeros_like, x0)
        m0_x = tree_map(np.zeros_like, x0)
        v0_x = tree_map(np.zeros_like, x0)
        m0_y = tree_map(np.zeros_like, y0)
        v0_y = tree_map(np.zeros_like, y0)
        cnt_x = 0
        cnt_y = 0
        return x0, y0, w0, m0_x, v0_x, m0_y, v0_y, cnt_x, cnt_y
    # Update function
    def update(i, state, batch, weights=(1.0, 1.0)):
        x, y, w, mx, vx, my, vy, cnt_x, cnt_y = state
        w_u, w_r = weights
        # Define update rules
        x_update = lambda x, gx: x - step_size_f(i)*gx
        y_update = lambda y, gy: y - step_size_g(i)*gy
        m_update = lambda m, g: (1.0 - b1) * g + b1 * m
        v_update = lambda v, g: (1.0 - b2) * np.square(g) + b2 * v
        g_update = lambda m, v: m / (np.sqrt(v) + eps)
        w_update = lambda x, y, w: w + (x-y)
        # f update
        def inner_step_f(k, inputs):
            x, mx, vx, cnt_x = inputs
            mhat_upate = lambda m: m / (1.0 - np.asarray(b1, m.dtype) ** (cnt_x+1))
            vhat_upate = lambda v: v / (1.0 - np.asarray(b2, v.dtype) ** (cnt_x+1))
            gx = grad(f, 0)(x, y, w, batch, w_u)
            mx = tree_map(m_update, mx, gx)
            vx = tree_map(v_update, vx, gx)
            mhatx = tree_map(mhat_upate, mx)
            vhatx = tree_map(vhat_upate, vx)
            gx = tree_map(g_update, mhatx, vhatx)
            x = tree_map(x_update, x, gx)
            cnt_x += 1
            return x, mx, vx, cnt_x
        x, mx, vx, cnt_x = lax.fori_loop(0, num_inner_iter_f, inner_step_f, (x, mx, vx, cnt_x))
        # g update
        def inner_step_g(k, inputs):
            y, my, vy, cnt_y = inputs
            mhat_upate = lambda m: m / (1.0 - np.asarray(b1, m.dtype) ** (cnt_y+1))
            vhat_upate = lambda v: v / (1.0 - np.asarray(b2, v.dtype) ** (cnt_y+1))
            gy = grad(g, 1)(x, y, w, batch, w_r)
            my = tree_map(m_update, my, gy)
            vy = tree_map(v_update, vy, gy)
            mhaty = tree_map(mhat_upate, my)
            vhaty = tree_map(vhat_upate, vy)
            gy = tree_map(g_update, mhaty, vhaty)
            y = tree_map(y_update, y, gy)
            cnt_y += 1
            return y, my, vy, cnt_y
        y, my, vy, cnt_y = lax.fori_loop(0, num_inner_iter_g, inner_step_g, (y, my, vy, cnt_y))
        # Update w
        w = tree_map(w_update, x, y, w)
        return x, y, w, mx, vx, my, vy, cnt_x, cnt_y
    # Get parameters function
    def get_params(state):
        x, _, _, _, _, _, _, _, _ = state
        return x
    return init, update, get_params
