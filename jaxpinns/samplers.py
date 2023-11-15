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
import numpy as onp
from jax import random, vmap

class Sampler:
    # Initialize the class
    def __init__(self, dim, coords, func, name = None):
        self.dim = (dim,) if dim > 1 else ()
        self.coords = coords
        self.func = func
        self.name = name

    def sample(self, N, key = random.PRNGKey(1234)):
        x = self.coords.min(1) + (self.coords.max(1)-self.coords.min(1))*random.uniform(key, (N,) + self.dim)
        y = vmap(self.func)(x)
        return x, y

class ResidualSampler:
    # Initialize the class
    def __init__(self, X, name = None):
        self.X = X
        self.N = self.X.shape[0]
        self.t_min = X[:,-1].min()
        self.t_max = X[:,-1].max()

    def sample(self, N, key = random.PRNGKey(1234)):
        idx = random.choice(key, self.N, (N,), replace=False)
        t_batch = self.t_min + (self.t_max - self.t_min) * random.uniform(key, (N,1))
        x_batch = self.X[idx,:-1]
        X_batch = np.concatenate([x_batch, t_batch], axis = 1)
        Y_batch = np.zeros((N,3))
        return X_batch, Y_batch


class DataSampler:
    # Initialize the class
    def __init__(self, X, Y, name = None):
        self.X = X
        self.Y = Y
        self.N = self.X.shape[0]

    def sample(self, N, key = random.PRNGKey(1234)):
        idx = random.choice(key, self.N, (N,), replace=False)
        X_batch = self.X[idx,:]
        Y_batch = self.Y[idx,:]
        return X_batch, Y_batch
