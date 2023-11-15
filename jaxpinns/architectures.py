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
from jax import random, vmap

def MLP(layers):
    ''' Vanilla MLP'''
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev*random.normal(k1, (d_in, d_out))
            b = np.zeros(d_out)
            return W, b
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params
    def apply(params, inputs):
        for W, b in params[:-1]:
            outputs = np.dot(inputs, W) + b
            inputs = np.tanh(outputs)
        W, b = params[-1]
        outputs = np.dot(inputs, W) + b
        return outputs
    return init, apply

def MLPsmallinit(layers):
    ''' Vanilla MLP'''
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            glorot_stddev = 0.01*1. / np.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev*random.normal(k1, (d_in, d_out))
            b = np.zeros(d_out)
            return W, b
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params
    def apply(params, inputs):
        for W, b in params[:-1]:
            outputs = np.dot(inputs, W) + b
            inputs = np.tanh(outputs)
        W, b = params[-1]
        outputs = np.dot(inputs, W) + b
        return outputs
    return init, apply


def ResNet(layers, depth):
    ''' MLP blocks with residual connections'''
    def init(rng_key):
        # Initialize neural net params
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            W = random.normal(k1, (d_in, d_out))
            b = random.normal(k2, (d_out,))
            return W, b
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params
    def mlp(params, inputs):
        for W, b in params:
            outputs = np.dot(inputs, W) + b
            inputs = np.tanh(outputs)
        return outputs
    def apply(params, inputs):
        for i in range(depth):
            outputs = mlp(params, inputs) + inputs
        return outputs
    return init, apply


def WN_MLP(layers):
    ''' MLP with weight normalization'''
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            W = random.normal(k1, (d_in, d_out))
            b = np.zeros(d_out)
            g = np.ones(d_out)
            return W, b, g
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params
    def apply(params, inputs):
        for W, b, g in params[:-1]:
            V = W/np.linalg.norm(W, axis = 0, keepdims=True)
            outputs = np.dot(inputs, V)*g + b
            inputs = np.tanh(outputs)
        W, b, g = params[-1]
        V = W/np.linalg.norm(W, axis = 0, keepdims=True)
        outputs = np.dot(inputs, V)*g + b
        return outputs
    return init, apply


def mFF_MLP(layers, freqs):
    ''' Multi-scale Fourier features MLP '''
    # Define input encoding function
    def input_encoding(x, w):
        out = np.hstack([np.sin(np.dot(x, w)),
                         np.cos(np.dot(x, w))])
        return out
    # Define activation function
    def activation(x, w, b):
        return np.tanh(np.dot(x, w) + b)
    # Initialize embedding weights (non-trainable)
    key, *keys = random.split(random.PRNGKey(0), len(freqs)+1)
    init_W = lambda key, freq: freq*random.normal(key, (layers[0], layers[1]//2))
    wFF = vmap(init_W)(*(np.array(keys), np.array(freqs)))
    # Define init function
    def init(rng_key):
        # Initialize neural net params
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev*random.normal(k1, (d_in, d_out))
            b = np.zeros(d_out)
            return W, b
        key, *keys = random.split(rng_key, len(layers))
        # All layers but last
        params = list(map(init_layer, keys[:-1], layers[1:-2], layers[2:-1]))
        # Append last layer
        params.append(init_layer(keys[-1], len(freqs)*layers[-2], layers[-1]))
        return params
    # Define apply function
    def apply(params, inputs):
        H = vmap(input_encoding, in_axes=(None,0))(inputs, wFF)
        for W, b in params[:-1]:
            H = vmap(activation, in_axes=(0,None,None))(H, W, b)
        H = H.flatten()
        W, b = params[-1]
        H = np.dot(H, W) + b
        return H
    return init, apply


def ST_mFF_MLP(layers, freqs_x, freqs_t):
    ''' Spatio-temporal multi-scale Fourier features MLP '''
    # Define input encoding function
    def input_encoding(x, w):
        out = np.hstack([np.sin(np.dot(x, w)),
                         np.cos(np.dot(x, w))])
        return out
    # Define activation function
    def activation(x, w, b):
        return np.tanh(np.dot(x, w) + b)
    # Initialize spatial embedding weights (non-trainable)
    init_Wx = lambda key, freq: freq*random.normal(key, (layers[0]-1, layers[1]//2))
    key, *keys = random.split(random.PRNGKey(0), len(freqs_x)+1)
    wFFx = vmap(init_Wx)(*(np.array(keys), np.array(freqs_x)))
    # Initialize temporal embedding weights (non-trainable)
    init_Wt = lambda key, freq: freq*random.normal(key, (1, layers[1]//2))
    key, *keys = random.split(random.PRNGKey(1), len(freqs_t)+1)
    wFFt = vmap(init_Wt)(*(np.array(keys), np.array(freqs_t)))
    # Define init function
    def init(rng_key):
        # Initialize neural net params
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev*random.normal(k1, (d_in, d_out))
            b = np.zeros(d_out)
            return W, b
        key, *keys = random.split(rng_key, len(layers))
        # All layers but last
        params = list(map(init_layer, keys[:-1], layers[1:-2], layers[2:-1]))
        # Append last layer
        params.append(init_layer(keys[-1], (len(freqs_x)*len(freqs_t))*layers[-2], layers[-1]))
        return params
    # Define apply function
    def apply(params, inputs):
        X = inputs[:-1]
        t = inputs[-1:]
        Hx = vmap(input_encoding, in_axes=(None,0))(X, wFFx)
        Ht = vmap(input_encoding, in_axes=(None,0))(t, wFFt)
        for W, b in params[:-1]:
            Hx = vmap(activation, in_axes=(0,None,None))(Hx, W, b)
            Ht = vmap(activation, in_axes=(0,None,None))(Ht, W, b)
        H = np.multiply(Hx[:,None], Ht).flatten()
        W, b = params[-1]
        H = np.dot(H, W) + b
        return H
    return init, apply
