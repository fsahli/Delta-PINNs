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
from jax import jacobian, eval_shape
from jax.tree_util import tree_map, tree_reduce
#import neural_tangents as nt
import operator

# Helper function for computing the Jacobians dot product with PyTrees
def sum_and_contract(j1, j2, output_ndim):
    _diagonal_axes = nt.utils.utils.canonicalize_axis((), output_ndim)
    _trace_axes = nt.utils.utils.canonicalize_axis((-1,), output_ndim)
    def contract(x, y):
        param_axes = list(range(x.ndim))[output_ndim:]
        contract_axes = _trace_axes + param_axes
        return nt.utils.utils.dot_general(x, y, contract_axes, _diagonal_axes)
    return tree_reduce(operator.add, tree_map(contract, j1, j2))

# computes the NTK: <jac(f1)(x1), jac(f2)(x2)>
def compute_ntk(f1, f2, x1, x2, params):
    j1 = jacobian(f1)(params, *x1)
    j2 = jacobian(f2)(params, *x2)
    fx1 = eval_shape(f1, params, *x1)
    ntk = sum_and_contract(j1, j2, fx1.ndim)
    return ntk
