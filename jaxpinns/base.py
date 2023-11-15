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
from jax import jit, grad, random, device_put
from jax.flatten_util import ravel_pytree

from numpy import loadtxt
from urllib.request import urlopen
from functools import partial
import itertools
from tqdm import trange

class PINN():
    ''' Base class for a general PINN model '''
    # Initialize the class
    def __init__(self, mu_X = 0.0, sigma_X = 1.0):
        # Normalization constants
        self.mu_X = mu_X
        self.sigma_X = sigma_X

    def architecture(self, neural_net, *args, init_key = random.PRNGKey(0)):
        ''' Network initialization and evaluation functions '''
        self.net_init, self.net_apply = neural_net(*args)
        self.net_params = self.net_init(init_key)

    def optimizer(self, opt, *args, saved_state=None):
        ''' Optimizer initialization and update functions '''
        self.opt_init, \
        self.opt_update, \
        self.get_params = opt(*args)
        if saved_state == None:
            self.opt_state = self.opt_init(self.net_params)
        else:
            state  = np.load(saved_state, allow_pickle=True)
            self.opt_state = [device_put(s) for s in state]
            self.net_params = self.get_params(self.opt_state)
        self.itercount = itertools.count()

    def logger(self, log, *args, io_step = 50, chkpt_step = np.inf):
        ''' Logger initialization and update functions '''
        self.io_step = io_step
        self.chkpt_step = chkpt_step
        # Training log
        self.training_log, self.training_log_update = log(*args)
        # Validation log
        io_keys = ['l2_error']
        log_keys = ['l2_error']
        log_funs = [self.L2error]
        args = (io_keys, log_keys, log_funs)
        self.validation_log, self.validation_log_update = log(*args)

    # Define a jit-compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch, weights):
        ''' Runs a single step of gradient descent '''
        return self.opt_update(i, opt_state, batch, weights)

    # Optimize parameters in a loop
    def train(self, dataset, nIter=10000, ntk_weights=True, validation_data=None, weights = None):
        ''' Main training loop '''
        data = iter(dataset)
        pbar = trange(nIter)
        # Initialize NTK weights
        if ntk_weights:
            params = self.get_params(self.opt_state)
            weights = self.update_NTK_weights(params, next(data))
        else:
            if weights is None:
                weights = tuple(1.0 for i in range(self.num_loss_terms))

        # Main training loop
        for it in pbar:
            batch = next(data)
            self.opt_state = self.step(next(self.itercount),
                                       self.opt_state,
                                       batch,
                                       weights)
            # Logger
            if it % self.io_step == 0:
                params = self.get_params(self.opt_state)
                if ntk_weights:
                    weights = self.update_NTK_weights(params, batch)
                self.training_log, io_dict = self.training_log_update(self.training_log, params, batch)
                if validation_data is not None:
                    self.validation_log, v_dict = self.validation_log_update(self.validation_log, params, validation_data)
                    io_dict.update(v_dict)
                pbar.set_postfix(io_dict)

            # Save model
            if (it+1) % self.chkpt_step == 0:
                filename = 'chkpt_it_{}.npy'.format(it+1)
                np.save(filename, self.opt_state)

    def L2error(self, params, batch):
        inputs, targets = batch
        outputs = self.predict(params, inputs)
        error = np.linalg.norm(targets-outputs,2)/np.linalg.norm(targets,2)
        return error

    def params_diff(self, p1, p2):
        p1_flat, _ = ravel_pytree(p1)
        p2_flat, _ = ravel_pytree(p2)
        diff = np.sum((p1_flat-p2_flat)**2)
        return diff


class dtPINN(PINN):
    ''' Base class for a discrete-time PINN model with IRK time-stepping '''
    # Initialize the class
    def __init__(self, dt, q, mu_X = 0.0, sigma_X = 1.0):
        super().__init__(mu_X, sigma_X)
        self.dt = dt
        self.q = max(q,1)
        url = 'https://raw.githubusercontent.com/PredictiveIntelligenceLab/PINNs/master/Utilities/IRK_weights/Butcher_IRK%d.txt' % (q)
        buf = np.float32(loadtxt(urlopen(url)))
        self.IRK_weights = np.reshape(buf[0:q**2+q], (q+1,q))
        self.IRK_times = buf[q**2+q:]
