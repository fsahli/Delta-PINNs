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
# from jax import jit

def logger(io_keys, log_keys, log_funs):
    log = {key: [] for key in log_keys}
    def update(log, params, data):
        for i, key in enumerate(log_keys):
            log[key].append(log_funs[i](params, data))
        io_dict = {key: log[key][-1] for key in io_keys}
        return log, io_dict
    return log, update
